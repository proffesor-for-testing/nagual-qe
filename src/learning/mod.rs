//! Learning infrastructure for self-optimizing SONA agents.
//!
//! This module provides:
//! - SONA learning loop for outcome recording and reward calculation
//! - Pattern consolidation for merging similar patterns
//! - Low-reward and stale pattern detection for maintenance
//! - Self-improvement cycles for pattern optimization
//! - Domain insights aggregation with trend analysis
//! - A/B testing infrastructure for comparing optimizations
//!
//! # SONA Learning Loop
//!
//! ```text
//! Pattern Application
//!        |
//!        v
//!   record_outcome()
//!        |
//!        v
//!   calculate_reward()
//!        |
//!        v
//!   Update Pattern
//!        |
//!        v
//! [Periodic Maintenance]
//!        |
//!        +---> consolidate_patterns()
//!        +---> find_low_reward_patterns()
//!        +---> find_stale_patterns()
//! ```
//!
//! # Self-Improvement Cycles
//!
//! The self-improvement system analyzes patterns and generates recommendations:
//!
//! ```text
//! Pattern Storage
//!        |
//!        v
//!   self_improve()
//!        |
//!        +---> analyze_patterns_by_domain()
//!        +---> identify_improvement_opportunities()
//!        +---> generate_recommendations()
//!        |
//!        v
//!   ImprovementPlan
//! ```
//!
//! # Example
//!
//! ```ignore
//! use nagual::learning::{SonaLearner, Outcome, SelfImprover, ImprovementConfig};
//!
//! // Record outcomes
//! let learner = SonaLearner::new(storage);
//! learner.record_outcome(&pattern_id, Outcome::Success, Some("Works great".to_string())).await?;
//!
//! // Run improvement cycle
//! let improver = SelfImprover::new(ImprovementConfig::default());
//! let plan = improver.self_improve(&patterns, Some("rust.async"));
//!
//! // A/B testing
//! let manager = AbTestManager::new(AbTestConfig::default());
//! let variant = manager.assign_variant("session-123");
//! ```

// Core SONA learning modules
pub mod ab_testing;
pub mod consolidation;
pub mod improvement;
pub mod insights;
pub mod meta;
pub mod scenario;
mod sona;
pub mod trajectory;

// KOS P4: Cross-Domain Transfer with Thompson Sampling
pub mod transfer;

// KOS P9: Elastic Weight Consolidation (anti-forgetting)
pub mod ewc;

// Meta-cognitive self-critique via strange-loop
pub mod strange_loop;

// Cross-domain transfer learning via Meta Thompson Sampling (ruvector)
pub mod domain_expansion;

// Auto-promotion engine (tier-based pattern graduation)
pub mod auto_promotion;

// Re-export auto-promotion types
pub use auto_promotion::{run_auto_promotion, PromotionRecord, PromotionResult};

// Re-export consolidation types
pub use consolidation::{
    archive_pattern, consolidate_patterns, find_low_reward_patterns, find_stale_patterns,
    mark_for_review, ConsolidatedGroup, ConsolidationConfig as PatternConsolidationConfig,
    ConsolidationResult as PatternConsolidationResult, LowRewardPatternReport,
    LowRewardRecommendation, PatternReviewStatus, RewardMergeStrategy, StalePatternReport,
};

// Re-export improvement types
pub use improvement::{
    ConsolidationConfig, ConsolidationResult, ConsolidationTrigger, ImprovementConfig,
    ImprovementConfigSnapshot, ImprovementOpportunity, ImprovementPlan, ImprovementSummary,
    OpportunityType, Recommendation, RecommendationType, SelfImprover,
};

// Re-export insights types
pub use insights::{
    aggregate_insights, ChildDomainSummary, DomainInsights, InsightsConfig, PatternTrend,
    TimeWindow, TopPatternInfo, Trend, TrendAnalysis,
};

// Re-export SONA types
pub use sona::{
    calculate_reward, get_domain_drift, get_drift_reports, get_meta_cognitive_stats,
    get_meta_cognitive_status, Outcome, OutcomeLog, OutcomeRecord, RewardModifiers, SonaConfig,
    SonaLearner, SonaStats,
};

// Re-export A/B testing types
pub use ab_testing::{
    AbTestConfig, AbTestManager, AbTestMetrics, AbTestResult, BaselineMetrics,
    ImprovementReport, ImprovementTarget, ImprovementTracker, MetricAggregation,
    MetricType, QuarterlyProgress, RegressionAlert, RegressionConfig,
    RegressionDetector, Severity, Variant, VariantStats,
};

// Re-export trajectory types
pub use trajectory::{
    CompactTrajectory, StepType, Trajectory, TrajectoryBuilder, TrajectoryFilter, TrajectoryId,
    TrajectoryOrderBy, TrajectoryStats, TrajectoryStep, TrajectoryStorage, TrajectoryStorageConfig,
    SQLITE_TRAJECTORIES_TABLE, SQLITE_TRAJECTORY_PATTERN_LINKS_TABLE, SQLITE_TRAJECTORY_STEPS_TABLE,
    // Week 2 Workstream B: Trajectory Analysis Engine
    ChainAnalysis, PatternChain, PatternTransition, TrajectoryAnalysisConfig, TrajectoryAnalyzer,
};

// Re-export scenario types (Week 3 Workstream B: Scenario Holdout System)
pub use scenario::{
    Difficulty, PatternScenarioStats, Scenario, ScenarioBuilder, ScenarioEvaluation,
    ScenarioEvaluationConfig, ScenarioEvaluator, ScenarioId, ScenarioStats, ScenarioStorage,
    SQLITE_SCENARIOS_TABLE, SQLITE_SCENARIO_EVALUATIONS_TABLE,
};

// Re-export strange-loop meta-cognitive types
pub use strange_loop::{
    evaluate_quality, load_latest as load_meta_latest, load_stats as load_meta_stats,
    persist_report as persist_meta_report, MetaCognitiveReport, MetaCognitiveTracker,
};

// Re-export meta-learning types (ADR-035: Meta-Learning)
pub use meta::{
    DomainTransfer, EwcConfig, EwcEngine, LearningRateConfig, MetaLearningConfig,
    MetaLearningEngine, MetaLearningStats, OptimizationResult, PatternImportance,
    PatternMapping, PatternTemplate, TransferEngine,
};

// Re-export domain expansion functions
pub use domain_expansion::{
    check_domain_plateau, get_expansion_domains, get_expansion_health, initiate_transfer,
    record_domain_outcome,
};
