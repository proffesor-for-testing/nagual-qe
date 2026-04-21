//! Self-improvement cycle implementation for pattern learning.
//!
//! This module provides the core self-improvement capabilities:
//! - Pattern analysis by domain
//! - Improvement opportunity identification
//! - Recommendation generation
//! - Consolidation triggers
//!
//! # Example
//!
//! ```ignore
//! use nagual::learning::{SelfImprover, ImprovementConfig};
//!
//! let improver = SelfImprover::new(patterns, config);
//! let plan = improver.self_improve("rust.async")?;
//!
//! for rec in &plan.recommendations {
//!     println!("Recommendation: {:?}", rec.recommendation_type);
//!     println!("  Target: {} patterns", rec.target_patterns.len());
//!     println!("  Rationale: {}", rec.rationale);
//!     println!("  Expected impact: {:.2}", rec.expected_impact);
//! }
//! ```

use std::collections::HashMap;

use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};

use crate::reasoning_bank::pattern::{Pattern, PatternId};

/// Configuration for the self-improvement process.
#[derive(Debug, Clone)]
pub struct ImprovementConfig {
    /// Minimum reward threshold for considering a pattern "high performing".
    pub high_reward_threshold: f32,

    /// Maximum reward threshold for considering a pattern "low performing".
    pub low_reward_threshold: f32,

    /// Minimum similarity score to consider patterns for consolidation.
    pub consolidation_similarity_threshold: f32,

    /// Minimum number of patterns in a domain before analysis is useful.
    pub min_patterns_for_analysis: usize,

    /// Days after which a pattern is considered stale without updates.
    pub stale_pattern_days: i64,

    /// Minimum usage count before considering effectiveness metrics reliable.
    pub min_usage_for_reliable_metrics: u32,

    /// Maximum number of recommendations to generate per improvement cycle.
    pub max_recommendations: usize,

    /// Weight for recency in scoring (0.0 - 1.0).
    pub recency_weight: f32,

    /// Weight for usage frequency in scoring (0.0 - 1.0).
    pub usage_weight: f32,

    /// Threshold for pattern complexity (solution length) to suggest splitting.
    pub complexity_threshold: usize,
}

impl Default for ImprovementConfig {
    fn default() -> Self {
        Self {
            high_reward_threshold: 0.8,
            low_reward_threshold: 0.4,
            consolidation_similarity_threshold: 0.85,
            min_patterns_for_analysis: 5,
            stale_pattern_days: 90,
            min_usage_for_reliable_metrics: 3,
            max_recommendations: 20,
            recency_weight: 0.3,
            usage_weight: 0.4,
            complexity_threshold: 1000,
        }
    }
}

impl ImprovementConfig {
    /// Create a new configuration with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the high reward threshold.
    pub fn with_high_reward_threshold(mut self, threshold: f32) -> Self {
        self.high_reward_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Set the low reward threshold.
    pub fn with_low_reward_threshold(mut self, threshold: f32) -> Self {
        self.low_reward_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Set the consolidation similarity threshold.
    pub fn with_consolidation_threshold(mut self, threshold: f32) -> Self {
        self.consolidation_similarity_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Set maximum recommendations per cycle.
    pub fn with_max_recommendations(mut self, max: usize) -> Self {
        self.max_recommendations = max;
        self
    }
}

/// Type of recommendation for pattern improvement.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RecommendationType {
    /// Consolidate similar patterns into a single, more general pattern.
    Consolidate,

    /// Archive low-performing or obsolete patterns.
    Archive,

    /// Improve an existing pattern based on usage feedback.
    Improve,

    /// Split a complex pattern into smaller, focused patterns.
    Split,

    /// Review patterns that may need manual attention.
    Review,

    /// Promote a high-performing pattern for wider use.
    Promote,
}

impl std::fmt::Display for RecommendationType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RecommendationType::Consolidate => write!(f, "consolidate"),
            RecommendationType::Archive => write!(f, "archive"),
            RecommendationType::Improve => write!(f, "improve"),
            RecommendationType::Split => write!(f, "split"),
            RecommendationType::Review => write!(f, "review"),
            RecommendationType::Promote => write!(f, "promote"),
        }
    }
}

/// A recommendation for pattern improvement.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Recommendation {
    /// Unique identifier for this recommendation.
    pub id: String,

    /// Type of recommendation.
    pub recommendation_type: RecommendationType,

    /// Pattern IDs targeted by this recommendation.
    pub target_patterns: Vec<PatternId>,

    /// Human-readable rationale for this recommendation.
    pub rationale: String,

    /// Expected impact score (0.0 - 1.0).
    pub expected_impact: f32,

    /// Priority level (higher = more important).
    pub priority: u8,

    /// Domain this recommendation applies to.
    pub domain: String,

    /// When this recommendation was generated.
    pub generated_at: DateTime<Utc>,

    /// Additional context or suggestions.
    pub details: Option<String>,
}

impl Recommendation {
    /// Create a new recommendation.
    pub fn new(
        recommendation_type: RecommendationType,
        target_patterns: Vec<PatternId>,
        rationale: impl Into<String>,
        expected_impact: f32,
        domain: impl Into<String>,
    ) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            recommendation_type,
            target_patterns,
            rationale: rationale.into(),
            expected_impact: expected_impact.clamp(0.0, 1.0),
            priority: Self::calculate_priority(recommendation_type, expected_impact),
            domain: domain.into(),
            generated_at: Utc::now(),
            details: None,
        }
    }

    /// Set additional details.
    pub fn with_details(mut self, details: impl Into<String>) -> Self {
        self.details = Some(details.into());
        self
    }

    /// Calculate priority based on type and impact.
    fn calculate_priority(rec_type: RecommendationType, impact: f32) -> u8 {
        let base_priority = match rec_type {
            RecommendationType::Archive => 3,      // Low priority
            RecommendationType::Review => 4,       // Medium-low
            RecommendationType::Consolidate => 5,  // Medium
            RecommendationType::Improve => 6,      // Medium-high
            RecommendationType::Split => 7,        // High
            RecommendationType::Promote => 8,      // Highest
        };

        // Adjust by impact (add 0-2 based on impact)
        let impact_bonus = (impact * 2.0).round() as u8;
        (base_priority + impact_bonus).min(10)
    }
}

/// Type of improvement opportunity identified.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OpportunityType {
    /// Similar patterns that could be merged.
    SimilarPatterns,

    /// Patterns with consistently low performance.
    LowPerformance,

    /// Patterns that haven't been used or updated recently.
    Stale,

    /// Patterns that are overly complex.
    OverlyComplex,

    /// High-performing patterns worth promoting.
    HighPerformer,

    /// Patterns with inconsistent metrics.
    InconsistentMetrics,

    /// Domain with insufficient pattern coverage.
    CoverageGap,
}

/// An identified opportunity for improvement.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImprovementOpportunity {
    /// Type of opportunity.
    pub opportunity_type: OpportunityType,

    /// Related pattern IDs.
    pub pattern_ids: Vec<PatternId>,

    /// Confidence in this opportunity (0.0 - 1.0).
    pub confidence: f32,

    /// Potential improvement value (0.0 - 1.0).
    pub potential_value: f32,

    /// Description of the opportunity.
    pub description: String,

    /// Supporting metrics.
    pub metrics: HashMap<String, f64>,
}

impl ImprovementOpportunity {
    /// Create a new improvement opportunity.
    pub fn new(
        opportunity_type: OpportunityType,
        pattern_ids: Vec<PatternId>,
        confidence: f32,
        potential_value: f32,
        description: impl Into<String>,
    ) -> Self {
        Self {
            opportunity_type,
            pattern_ids,
            confidence: confidence.clamp(0.0, 1.0),
            potential_value: potential_value.clamp(0.0, 1.0),
            description: description.into(),
            metrics: HashMap::new(),
        }
    }

    /// Add a supporting metric.
    pub fn with_metric(mut self, name: impl Into<String>, value: f64) -> Self {
        self.metrics.insert(name.into(), value);
        self
    }
}

/// The complete improvement plan generated by self_improve().
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImprovementPlan {
    /// Domain this plan applies to (or "all" for global analysis).
    pub domain: String,

    /// Generated recommendations.
    pub recommendations: Vec<Recommendation>,

    /// Identified opportunities.
    pub opportunities: Vec<ImprovementOpportunity>,

    /// Summary statistics.
    pub summary: ImprovementSummary,

    /// When this plan was generated.
    pub generated_at: DateTime<Utc>,

    /// Configuration used to generate this plan.
    pub config_snapshot: ImprovementConfigSnapshot,
}

/// Summary of the improvement plan.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ImprovementSummary {
    /// Total patterns analyzed.
    pub total_patterns: usize,

    /// Number of patterns by recommendation type.
    pub patterns_by_action: HashMap<String, usize>,

    /// Total expected impact if all recommendations are applied.
    pub total_expected_impact: f32,

    /// Highest priority recommendation type.
    pub highest_priority_action: Option<String>,

    /// Average pattern quality in the domain.
    pub average_quality: f32,

    /// Number of high-performing patterns.
    pub high_performers: usize,

    /// Number of low-performing patterns.
    pub low_performers: usize,
}

/// Snapshot of configuration for auditing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImprovementConfigSnapshot {
    pub high_reward_threshold: f32,
    pub low_reward_threshold: f32,
    pub consolidation_similarity_threshold: f32,
}

impl From<&ImprovementConfig> for ImprovementConfigSnapshot {
    fn from(config: &ImprovementConfig) -> Self {
        Self {
            high_reward_threshold: config.high_reward_threshold,
            low_reward_threshold: config.low_reward_threshold,
            consolidation_similarity_threshold: config.consolidation_similarity_threshold,
        }
    }
}

/// Trigger types for pattern consolidation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConsolidationTrigger {
    /// Trigger consolidation after a time interval (hours).
    TimeBased(u64),

    /// Trigger consolidation after N new patterns.
    CountBased(usize),

    /// Manual trigger.
    Manual,

    /// Trigger on quality degradation.
    QualityDrop(u8), // Percentage drop

    /// Trigger when storage threshold is reached (percentage).
    StorageThreshold(u8),
}

impl ConsolidationTrigger {
    /// Create a time-based trigger (default: 24 hours).
    pub fn time_based() -> Self {
        Self::TimeBased(24)
    }

    /// Create a time-based trigger with custom hours.
    pub fn every_hours(hours: u64) -> Self {
        Self::TimeBased(hours)
    }

    /// Create a count-based trigger (default: 100 patterns).
    pub fn count_based() -> Self {
        Self::CountBased(100)
    }

    /// Create a count-based trigger with custom count.
    pub fn every_n_patterns(n: usize) -> Self {
        Self::CountBased(n)
    }

    /// Create a manual trigger.
    pub fn manual() -> Self {
        Self::Manual
    }
}

impl std::fmt::Display for ConsolidationTrigger {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConsolidationTrigger::TimeBased(hours) => write!(f, "time_based({}h)", hours),
            ConsolidationTrigger::CountBased(count) => write!(f, "count_based({})", count),
            ConsolidationTrigger::Manual => write!(f, "manual"),
            ConsolidationTrigger::QualityDrop(pct) => write!(f, "quality_drop({}%)", pct),
            ConsolidationTrigger::StorageThreshold(pct) => write!(f, "storage({}%)", pct),
        }
    }
}

/// Configuration for consolidation triggers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsolidationConfig {
    /// Active triggers.
    pub triggers: Vec<ConsolidationTrigger>,

    /// Last time consolidation was run.
    pub last_run: Option<DateTime<Utc>>,

    /// Pattern count at last consolidation.
    pub last_pattern_count: usize,

    /// Minimum similarity for automatic consolidation.
    pub auto_consolidate_threshold: f32,

    /// Whether to auto-archive patterns below threshold.
    pub auto_archive_low_performers: bool,

    /// Threshold for auto-archiving (reward).
    pub auto_archive_threshold: f32,

    /// Minimum age (days) before auto-archiving.
    pub auto_archive_min_age_days: i64,
}

impl Default for ConsolidationConfig {
    fn default() -> Self {
        Self {
            triggers: vec![
                ConsolidationTrigger::TimeBased(24),
                ConsolidationTrigger::CountBased(100),
            ],
            last_run: None,
            last_pattern_count: 0,
            auto_consolidate_threshold: 0.9,
            auto_archive_low_performers: false,
            auto_archive_threshold: 0.3,
            auto_archive_min_age_days: 30,
        }
    }
}

impl ConsolidationConfig {
    /// Check if any trigger condition is met.
    pub fn check_trigger(&self, current_pattern_count: usize) -> Option<ConsolidationTrigger> {
        for trigger in &self.triggers {
            if self.is_trigger_active(trigger, current_pattern_count) {
                return Some(*trigger);
            }
        }
        None
    }

    /// Check if a specific trigger is active.
    fn is_trigger_active(&self, trigger: &ConsolidationTrigger, current_count: usize) -> bool {
        match trigger {
            ConsolidationTrigger::TimeBased(hours) => {
                if let Some(last) = self.last_run {
                    let elapsed = Utc::now().signed_duration_since(last);
                    elapsed >= Duration::hours(*hours as i64)
                } else {
                    true // Never run before
                }
            }
            ConsolidationTrigger::CountBased(threshold) => {
                current_count >= self.last_pattern_count + threshold
            }
            ConsolidationTrigger::Manual => false, // Manual is never auto-triggered
            ConsolidationTrigger::QualityDrop(_) => false, // Requires metric comparison
            ConsolidationTrigger::StorageThreshold(_) => false, // Requires storage info
        }
    }

    /// Mark consolidation as run.
    pub fn mark_run(&mut self, pattern_count: usize) {
        self.last_run = Some(Utc::now());
        self.last_pattern_count = pattern_count;
    }
}

/// Result of a consolidation operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsolidationResult {
    /// Number of patterns analyzed.
    pub patterns_analyzed: usize,

    /// Number of consolidation groups found.
    pub groups_found: usize,

    /// Number of patterns consolidated.
    pub patterns_consolidated: usize,

    /// Number of patterns archived.
    pub patterns_archived: usize,

    /// Duration of the consolidation process.
    pub duration_ms: u64,

    /// Trigger that initiated this consolidation.
    pub trigger: ConsolidationTrigger,

    /// Timestamp of completion.
    pub completed_at: DateTime<Utc>,

    /// Any warnings or issues encountered.
    pub warnings: Vec<String>,
}

impl ConsolidationResult {
    /// Create a new consolidation result.
    pub fn new(trigger: ConsolidationTrigger) -> Self {
        Self {
            patterns_analyzed: 0,
            groups_found: 0,
            patterns_consolidated: 0,
            patterns_archived: 0,
            duration_ms: 0,
            trigger,
            completed_at: Utc::now(),
            warnings: Vec::new(),
        }
    }

    /// Check if the consolidation had any effect.
    pub fn had_effect(&self) -> bool {
        self.patterns_consolidated > 0 || self.patterns_archived > 0
    }
}

/// The main self-improver struct that coordinates improvement cycles.
pub struct SelfImprover {
    /// Configuration for improvement analysis.
    config: ImprovementConfig,

    /// Consolidation configuration.
    consolidation_config: ConsolidationConfig,
}

impl SelfImprover {
    /// Create a new SelfImprover with the given configuration.
    pub fn new(config: ImprovementConfig) -> Self {
        Self {
            config,
            consolidation_config: ConsolidationConfig::default(),
        }
    }

    /// Create with both improvement and consolidation configs.
    pub fn with_consolidation_config(
        config: ImprovementConfig,
        consolidation_config: ConsolidationConfig,
    ) -> Self {
        Self {
            config,
            consolidation_config,
        }
    }

    /// Run a self-improvement cycle on patterns in the specified domain.
    ///
    /// Analyzes patterns, identifies opportunities, and generates recommendations.
    pub fn self_improve(
        &self,
        patterns: &[Pattern],
        domain: Option<&str>,
    ) -> ImprovementPlan {
        let domain_str = domain.unwrap_or("all").to_string();

        // Filter patterns by domain if specified
        let filtered_patterns: Vec<&Pattern> = if let Some(d) = domain {
            patterns
                .iter()
                .filter(|p| p.category().to_string().starts_with(d))
                .collect()
        } else {
            patterns.iter().collect()
        };

        // Analyze patterns and identify opportunities
        let opportunities = self.identify_opportunities(&filtered_patterns);

        // Generate recommendations from opportunities
        let recommendations = self.generate_recommendations(&opportunities, &domain_str);

        // Calculate summary
        let summary = self.calculate_summary(&filtered_patterns, &recommendations);

        ImprovementPlan {
            domain: domain_str,
            recommendations,
            opportunities,
            summary,
            generated_at: Utc::now(),
            config_snapshot: ImprovementConfigSnapshot::from(&self.config),
        }
    }

    /// Identify improvement opportunities in the given patterns.
    fn identify_opportunities(&self, patterns: &[&Pattern]) -> Vec<ImprovementOpportunity> {
        let mut opportunities = Vec::new();

        // Find low performers
        opportunities.extend(self.find_low_performers(patterns));

        // Find stale patterns
        opportunities.extend(self.find_stale_patterns(patterns));

        // Find overly complex patterns
        opportunities.extend(self.find_complex_patterns(patterns));

        // Find high performers
        opportunities.extend(self.find_high_performers(patterns));

        // Find patterns with inconsistent metrics
        opportunities.extend(self.find_inconsistent_patterns(patterns));

        // Sort by potential value descending
        opportunities.sort_by(|a, b| {
            b.potential_value
                .partial_cmp(&a.potential_value)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        opportunities
    }

    /// Find patterns with consistently low performance.
    fn find_low_performers(&self, patterns: &[&Pattern]) -> Vec<ImprovementOpportunity> {
        let low_performers: Vec<_> = patterns
            .iter()
            .filter(|p| {
                p.reward() < self.config.low_reward_threshold
                    && p.reuse_count() >= self.config.min_usage_for_reliable_metrics
            })
            .collect();

        if low_performers.is_empty() {
            return Vec::new();
        }

        let avg_reward: f32 = low_performers.iter().map(|p| p.reward()).sum::<f32>()
            / low_performers.len() as f32;

        vec![ImprovementOpportunity::new(
            OpportunityType::LowPerformance,
            low_performers.iter().map(|p| p.id().clone()).collect(),
            0.9, // High confidence
            0.6, // Medium-high value
            format!(
                "Found {} patterns with reward below {:.2} (avg: {:.2})",
                low_performers.len(),
                self.config.low_reward_threshold,
                avg_reward
            ),
        )
        .with_metric("count", low_performers.len() as f64)
        .with_metric("avg_reward", avg_reward as f64)]
    }

    /// Find patterns that haven't been used or updated recently.
    fn find_stale_patterns(&self, patterns: &[&Pattern]) -> Vec<ImprovementOpportunity> {
        let cutoff = Utc::now() - Duration::days(self.config.stale_pattern_days);

        let stale: Vec<_> = patterns
            .iter()
            .filter(|p| p.updated_at() < cutoff && p.reuse_count() < 2)
            .collect();

        if stale.is_empty() {
            return Vec::new();
        }

        vec![ImprovementOpportunity::new(
            OpportunityType::Stale,
            stale.iter().map(|p| p.id().clone()).collect(),
            0.85,
            0.4, // Lower value since they might still be useful
            format!(
                "Found {} patterns not updated in {} days with minimal usage",
                stale.len(),
                self.config.stale_pattern_days
            ),
        )
        .with_metric("count", stale.len() as f64)
        .with_metric("stale_days", self.config.stale_pattern_days as f64)]
    }

    /// Find patterns that are overly complex and might benefit from splitting.
    fn find_complex_patterns(&self, patterns: &[&Pattern]) -> Vec<ImprovementOpportunity> {
        let complex: Vec<_> = patterns
            .iter()
            .filter(|p| p.solution().len() > self.config.complexity_threshold)
            .collect();

        if complex.is_empty() {
            return Vec::new();
        }

        let avg_length: f64 = complex.iter().map(|p| p.solution().len()).sum::<usize>() as f64
            / complex.len() as f64;

        vec![ImprovementOpportunity::new(
            OpportunityType::OverlyComplex,
            complex.iter().map(|p| p.id().clone()).collect(),
            0.7, // Lower confidence - complexity is subjective
            0.5,
            format!(
                "Found {} patterns exceeding complexity threshold of {} chars (avg: {:.0})",
                complex.len(),
                self.config.complexity_threshold,
                avg_length
            ),
        )
        .with_metric("count", complex.len() as f64)
        .with_metric("avg_length", avg_length)]
    }

    /// Find high-performing patterns worth promoting.
    fn find_high_performers(&self, patterns: &[&Pattern]) -> Vec<ImprovementOpportunity> {
        let high_performers: Vec<_> = patterns
            .iter()
            .filter(|p| {
                p.reward() >= self.config.high_reward_threshold
                    && p.reuse_count() >= self.config.min_usage_for_reliable_metrics
            })
            .collect();

        if high_performers.is_empty() {
            return Vec::new();
        }

        let avg_reward: f32 = high_performers.iter().map(|p| p.reward()).sum::<f32>()
            / high_performers.len() as f32;

        vec![ImprovementOpportunity::new(
            OpportunityType::HighPerformer,
            high_performers.iter().map(|p| p.id().clone()).collect(),
            0.95, // High confidence
            0.8,  // High value
            format!(
                "Found {} high-performing patterns (reward >= {:.2}, avg: {:.2})",
                high_performers.len(),
                self.config.high_reward_threshold,
                avg_reward
            ),
        )
        .with_metric("count", high_performers.len() as f64)
        .with_metric("avg_reward", avg_reward as f64)]
    }

    /// Find patterns with inconsistent metrics.
    fn find_inconsistent_patterns(&self, patterns: &[&Pattern]) -> Vec<ImprovementOpportunity> {
        let inconsistent: Vec<_> = patterns
            .iter()
            .filter(|p| {
                // High confidence but low reward suggests inconsistency
                let confidence = p.confidence();
                let reward = p.reward();
                let effectiveness = p.effectiveness();

                // Check for large discrepancies
                (confidence - reward).abs() > 0.4 || (effectiveness - reward).abs() > 0.4
            })
            .collect();

        if inconsistent.is_empty() {
            return Vec::new();
        }

        vec![ImprovementOpportunity::new(
            OpportunityType::InconsistentMetrics,
            inconsistent.iter().map(|p| p.id().clone()).collect(),
            0.75,
            0.45,
            format!(
                "Found {} patterns with inconsistent metrics (large gaps between confidence/effectiveness and reward)",
                inconsistent.len()
            ),
        )
        .with_metric("count", inconsistent.len() as f64)]
    }

    /// Generate recommendations from identified opportunities.
    fn generate_recommendations(
        &self,
        opportunities: &[ImprovementOpportunity],
        domain: &str,
    ) -> Vec<Recommendation> {
        let mut recommendations = Vec::new();

        for opp in opportunities {
            let rec = match opp.opportunity_type {
                OpportunityType::LowPerformance => Recommendation::new(
                    RecommendationType::Archive,
                    opp.pattern_ids.clone(),
                    format!(
                        "Consider archiving {} low-performing patterns. {}",
                        opp.pattern_ids.len(),
                        opp.description
                    ),
                    opp.potential_value,
                    domain,
                )
                .with_details("Low reward patterns may be outdated or poorly matched to problems."),

                OpportunityType::Stale => Recommendation::new(
                    RecommendationType::Review,
                    opp.pattern_ids.clone(),
                    format!(
                        "Review {} stale patterns for relevance. {}",
                        opp.pattern_ids.len(),
                        opp.description
                    ),
                    opp.potential_value,
                    domain,
                )
                .with_details("Patterns without recent activity may need updating or archiving."),

                OpportunityType::OverlyComplex => Recommendation::new(
                    RecommendationType::Split,
                    opp.pattern_ids.clone(),
                    format!(
                        "Consider splitting {} complex patterns. {}",
                        opp.pattern_ids.len(),
                        opp.description
                    ),
                    opp.potential_value,
                    domain,
                )
                .with_details(
                    "Complex patterns may be more effective when broken into focused sub-patterns.",
                ),

                OpportunityType::HighPerformer => Recommendation::new(
                    RecommendationType::Promote,
                    opp.pattern_ids.clone(),
                    format!(
                        "Promote {} high-performing patterns. {}",
                        opp.pattern_ids.len(),
                        opp.description
                    ),
                    opp.potential_value,
                    domain,
                )
                .with_details(
                    "High performers can serve as templates for similar problems in other domains.",
                ),

                OpportunityType::SimilarPatterns => Recommendation::new(
                    RecommendationType::Consolidate,
                    opp.pattern_ids.clone(),
                    format!(
                        "Consolidate {} similar patterns. {}",
                        opp.pattern_ids.len(),
                        opp.description
                    ),
                    opp.potential_value,
                    domain,
                )
                .with_details("Merging similar patterns reduces redundancy and improves retrieval."),

                OpportunityType::InconsistentMetrics => Recommendation::new(
                    RecommendationType::Improve,
                    opp.pattern_ids.clone(),
                    format!(
                        "Investigate {} patterns with inconsistent metrics. {}",
                        opp.pattern_ids.len(),
                        opp.description
                    ),
                    opp.potential_value,
                    domain,
                )
                .with_details(
                    "Large gaps between confidence and reward may indicate calibration issues.",
                ),

                OpportunityType::CoverageGap => Recommendation::new(
                    RecommendationType::Review,
                    opp.pattern_ids.clone(),
                    format!("Coverage gap identified. {}", opp.description),
                    opp.potential_value,
                    domain,
                )
                .with_details("Consider adding patterns to cover this gap."),
            };

            recommendations.push(rec);

            if recommendations.len() >= self.config.max_recommendations {
                break;
            }
        }

        // Sort by priority descending
        recommendations.sort_by(|a, b| b.priority.cmp(&a.priority));

        recommendations
    }

    /// Calculate summary statistics for the improvement plan.
    fn calculate_summary(
        &self,
        patterns: &[&Pattern],
        recommendations: &[Recommendation],
    ) -> ImprovementSummary {
        let total_patterns = patterns.len();

        // Count patterns by recommendation type
        let mut patterns_by_action: HashMap<String, usize> = HashMap::new();
        for rec in recommendations {
            *patterns_by_action
                .entry(rec.recommendation_type.to_string())
                .or_insert(0) += rec.target_patterns.len();
        }

        // Calculate total expected impact
        let total_expected_impact: f32 = recommendations
            .iter()
            .map(|r| r.expected_impact)
            .sum::<f32>()
            / recommendations.len().max(1) as f32;

        // Find highest priority action
        let highest_priority_action = recommendations
            .first()
            .map(|r| r.recommendation_type.to_string());

        // Calculate average quality
        let average_quality = if total_patterns > 0 {
            patterns.iter().map(|p| p.quality_score()).sum::<f32>() / total_patterns as f32
        } else {
            0.0
        };

        // Count high and low performers
        let high_performers = patterns
            .iter()
            .filter(|p| p.reward() >= self.config.high_reward_threshold)
            .count();

        let low_performers = patterns
            .iter()
            .filter(|p| p.reward() < self.config.low_reward_threshold)
            .count();

        ImprovementSummary {
            total_patterns,
            patterns_by_action,
            total_expected_impact,
            highest_priority_action,
            average_quality,
            high_performers,
            low_performers,
        }
    }

    /// Check if consolidation should be triggered.
    pub fn check_consolidation_trigger(&self, pattern_count: usize) -> Option<ConsolidationTrigger> {
        self.consolidation_config.check_trigger(pattern_count)
    }

    /// Get a reference to the consolidation config.
    pub fn consolidation_config(&self) -> &ConsolidationConfig {
        &self.consolidation_config
    }

    /// Get a mutable reference to the consolidation config.
    pub fn consolidation_config_mut(&mut self) -> &mut ConsolidationConfig {
        &mut self.consolidation_config
    }

    /// Get the improvement config.
    pub fn config(&self) -> &ImprovementConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reasoning_bank::pattern::PatternCategory;

    fn create_test_pattern(
        problem: &str,
        solution: &str,
        category: PatternCategory,
        reward: f32,
        reuse_count: u32,
    ) -> Pattern {
        Pattern::builder()
            .problem(problem)
            .solution(solution)
            .category(category)
            .reward(reward)
            .reuse_count(reuse_count)
            .effectiveness(reward)
            .confidence(reward)
            .build()
    }

    #[test]
    fn test_improvement_config_default() {
        let config = ImprovementConfig::default();
        assert_eq!(config.high_reward_threshold, 0.8);
        assert_eq!(config.low_reward_threshold, 0.4);
        assert_eq!(config.consolidation_similarity_threshold, 0.85);
    }

    #[test]
    fn test_improvement_config_builder() {
        let config = ImprovementConfig::new()
            .with_high_reward_threshold(0.9)
            .with_low_reward_threshold(0.3)
            .with_max_recommendations(10);

        assert_eq!(config.high_reward_threshold, 0.9);
        assert_eq!(config.low_reward_threshold, 0.3);
        assert_eq!(config.max_recommendations, 10);
    }

    #[test]
    fn test_recommendation_type_display() {
        assert_eq!(RecommendationType::Consolidate.to_string(), "consolidate");
        assert_eq!(RecommendationType::Archive.to_string(), "archive");
        assert_eq!(RecommendationType::Improve.to_string(), "improve");
    }

    #[test]
    fn test_recommendation_priority() {
        let high_impact_promote = Recommendation::new(
            RecommendationType::Promote,
            vec![],
            "test",
            0.9,
            "test",
        );
        let low_impact_archive = Recommendation::new(
            RecommendationType::Archive,
            vec![],
            "test",
            0.2,
            "test",
        );

        assert!(high_impact_promote.priority > low_impact_archive.priority);
    }

    #[test]
    fn test_consolidation_trigger_display() {
        assert_eq!(ConsolidationTrigger::time_based().to_string(), "time_based(24h)");
        assert_eq!(ConsolidationTrigger::count_based().to_string(), "count_based(100)");
        assert_eq!(ConsolidationTrigger::manual().to_string(), "manual");
    }

    #[test]
    fn test_consolidation_config_check_trigger_count_based() {
        let mut config = ConsolidationConfig::default();
        config.triggers = vec![ConsolidationTrigger::CountBased(10)];
        config.last_pattern_count = 50;

        // Not triggered yet
        assert!(config.check_trigger(55).is_none());

        // Triggered
        assert!(config.check_trigger(60).is_some());
    }

    #[test]
    fn test_consolidation_config_mark_run() {
        let mut config = ConsolidationConfig::default();
        config.mark_run(100);

        assert!(config.last_run.is_some());
        assert_eq!(config.last_pattern_count, 100);
    }

    #[test]
    fn test_self_improver_find_low_performers() {
        let config = ImprovementConfig::default();
        let improver = SelfImprover::new(config);

        let patterns = vec![
            create_test_pattern("P1", "S1", PatternCategory::Testing, 0.9, 5),
            create_test_pattern("P2", "S2", PatternCategory::Testing, 0.3, 5),
            create_test_pattern("P3", "S3", PatternCategory::Testing, 0.2, 10),
        ];

        let pattern_refs: Vec<&Pattern> = patterns.iter().collect();
        let opportunities = improver.find_low_performers(&pattern_refs);

        assert_eq!(opportunities.len(), 1);
        assert_eq!(opportunities[0].opportunity_type, OpportunityType::LowPerformance);
        assert_eq!(opportunities[0].pattern_ids.len(), 2);
    }

    #[test]
    fn test_self_improver_find_high_performers() {
        let config = ImprovementConfig::default();
        let improver = SelfImprover::new(config);

        let patterns = vec![
            create_test_pattern("P1", "S1", PatternCategory::Testing, 0.9, 5),
            create_test_pattern("P2", "S2", PatternCategory::Testing, 0.85, 5),
            create_test_pattern("P3", "S3", PatternCategory::Testing, 0.5, 10),
        ];

        let pattern_refs: Vec<&Pattern> = patterns.iter().collect();
        let opportunities = improver.find_high_performers(&pattern_refs);

        assert_eq!(opportunities.len(), 1);
        assert_eq!(opportunities[0].opportunity_type, OpportunityType::HighPerformer);
        assert_eq!(opportunities[0].pattern_ids.len(), 2);
    }

    #[test]
    fn test_self_improver_find_complex_patterns() {
        let config = ImprovementConfig::new();
        let improver = SelfImprover {
            config: ImprovementConfig {
                complexity_threshold: 50,
                ..config
            },
            consolidation_config: ConsolidationConfig::default(),
        };

        let long_solution = "x".repeat(100);
        let patterns = vec![
            create_test_pattern("P1", "Short solution", PatternCategory::Testing, 0.5, 1),
            create_test_pattern("P2", &long_solution, PatternCategory::Testing, 0.5, 1),
        ];

        let pattern_refs: Vec<&Pattern> = patterns.iter().collect();
        let opportunities = improver.find_complex_patterns(&pattern_refs);

        assert_eq!(opportunities.len(), 1);
        assert_eq!(opportunities[0].opportunity_type, OpportunityType::OverlyComplex);
        assert_eq!(opportunities[0].pattern_ids.len(), 1);
    }

    #[test]
    fn test_self_improve_generates_plan() {
        let config = ImprovementConfig::default();
        let improver = SelfImprover::new(config);

        let patterns = vec![
            create_test_pattern("P1", "S1", PatternCategory::Testing, 0.9, 5),
            create_test_pattern("P2", "S2", PatternCategory::Testing, 0.3, 5),
            create_test_pattern("P3", "S3", PatternCategory::Performance, 0.5, 1),
        ];

        let plan = improver.self_improve(&patterns, None);

        assert_eq!(plan.domain, "all");
        assert_eq!(plan.summary.total_patterns, 3);
        assert!(!plan.opportunities.is_empty());
        assert!(!plan.recommendations.is_empty());
    }

    #[test]
    fn test_self_improve_filters_by_domain() {
        let config = ImprovementConfig::default();
        let improver = SelfImprover::new(config);

        let patterns = vec![
            create_test_pattern("P1", "S1", PatternCategory::Testing, 0.9, 5),
            create_test_pattern("P2", "S2", PatternCategory::Testing, 0.8, 5),
            create_test_pattern("P3", "S3", PatternCategory::Performance, 0.5, 1),
        ];

        let plan = improver.self_improve(&patterns, Some("testing"));

        assert_eq!(plan.domain, "testing");
        assert_eq!(plan.summary.total_patterns, 2);
    }

    #[test]
    fn test_improvement_summary_calculation() {
        let config = ImprovementConfig::default();
        let improver = SelfImprover::new(config);

        let patterns = vec![
            create_test_pattern("P1", "S1", PatternCategory::Testing, 0.9, 5),
            create_test_pattern("P2", "S2", PatternCategory::Testing, 0.3, 5),
            create_test_pattern("P3", "S3", PatternCategory::Testing, 0.5, 1),
        ];

        let plan = improver.self_improve(&patterns, None);

        assert_eq!(plan.summary.total_patterns, 3);
        assert_eq!(plan.summary.high_performers, 1);
        assert_eq!(plan.summary.low_performers, 1);
        assert!(plan.summary.average_quality > 0.0);
    }

    #[test]
    fn test_consolidation_result() {
        let result = ConsolidationResult::new(ConsolidationTrigger::Manual);

        assert!(!result.had_effect());
        assert_eq!(result.patterns_analyzed, 0);
    }
}
