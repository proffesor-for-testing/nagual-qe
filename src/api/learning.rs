//! Learning namespace API for recording outcomes and triggering improvement.
//!
//! The learning API provides methods for the SONA learning loop, including
//! outcome recording, pattern consolidation, self-improvement, and insight
//! aggregation.
//!
//! # Example
//!
//! ```rust,ignore
//! use nagual::learning::Outcome;
//!
//! // Record an outcome
//! let reward = nagual.learning.record_outcome(
//!     &pattern_id,
//!     Outcome::Success,
//!     Some("Pattern worked well")
//! ).await?;
//!
//! // Get insights for a domain
//! let insights = nagual.learning.insights("rust.async").await?;
//!
//! // Run self-improvement cycle
//! let result = nagual.learning.improve(Some("rust")).await?;
//!
//! // Consolidate similar patterns
//! let result = nagual.learning.consolidate().await?;
//! ```

use std::sync::Arc;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tracing::{debug, info, instrument};

use super::NagualState;
use crate::error::{NagualError, Result};
use crate::learning::{
    aggregate_insights, consolidate_patterns, DomainInsights, ImprovementPlan,
    InsightsConfig, Outcome, PatternConsolidationConfig,
    RewardModifiers, SelfImprover, SonaConfig, SonaLearner, SonaStats,
};
use crate::reasoning_bank::pattern::PatternId;

/// Result of a self-improvement cycle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImprovementResult {
    /// The improvement plan generated
    pub plan: ImprovementPlan,

    /// Number of patterns analyzed
    pub patterns_analyzed: usize,

    /// Number of recommendations generated
    pub recommendations_count: usize,

    /// Domain that was improved (if scoped)
    pub domain: Option<String>,

    /// When the improvement was run
    pub timestamp: DateTime<Utc>,
}

/// Result of pattern consolidation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsolidationResult {
    /// Number of pattern groups consolidated
    pub groups_consolidated: usize,

    /// Number of patterns merged
    pub patterns_merged: usize,

    /// Number of patterns archived (low reward)
    pub patterns_archived: usize,

    /// Number of patterns marked for review
    pub patterns_for_review: usize,

    /// When the consolidation was run
    pub timestamp: DateTime<Utc>,
}

/// Result of insights aggregation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InsightsResult {
    /// Domain these insights are for
    pub domain: String,

    /// Overall success rate
    pub success_rate: f32,

    /// Average reward
    pub avg_reward: f32,

    /// Average effectiveness
    pub avg_effectiveness: f32,

    /// Total patterns in domain
    pub total_patterns: usize,

    /// Overall trend direction
    pub trend: String,

    /// Top pattern IDs
    pub top_patterns: Vec<String>,

    /// When insights were generated
    pub timestamp: DateTime<Utc>,
}

impl From<DomainInsights> for InsightsResult {
    fn from(insights: DomainInsights) -> Self {
        Self {
            domain: insights.domain,
            success_rate: insights.success_rate,
            avg_reward: insights.avg_reward,
            avg_effectiveness: insights.avg_effectiveness,
            total_patterns: insights.total_patterns,
            trend: insights.trend.to_string(),
            top_patterns: insights
                .top_patterns
                .iter()
                .take(5)
                .map(|p| p.id.to_string())
                .collect(),
            timestamp: insights.generated_at,
        }
    }
}

/// Options for recording outcomes.
#[derive(Debug, Clone, Default)]
pub struct RecordOutcomeOptions {
    /// Confidence in the outcome assessment
    pub confidence: Option<f32>,

    /// Context relevance (how well the pattern matched)
    pub context_relevance: Option<f32>,

    /// Speed factor (1.0 = faster than expected)
    pub speed_factor: Option<f32>,

    /// User satisfaction rating
    pub user_satisfaction: Option<f32>,

    /// Whether the outcome was verified
    pub verified: bool,

    /// Session ID for tracking
    pub session_id: Option<String>,

    /// Agent ID that recorded this outcome
    pub agent_id: Option<String>,
}

impl RecordOutcomeOptions {
    /// Create new options with defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set confidence level.
    pub fn confidence(mut self, confidence: f32) -> Self {
        self.confidence = Some(confidence.clamp(0.0, 1.0));
        self
    }

    /// Set context relevance.
    pub fn context_relevance(mut self, relevance: f32) -> Self {
        self.context_relevance = Some(relevance.clamp(0.0, 1.0));
        self
    }

    /// Set speed factor.
    pub fn speed_factor(mut self, speed: f32) -> Self {
        self.speed_factor = Some(speed.clamp(0.0, 1.0));
        self
    }

    /// Set user satisfaction.
    pub fn user_satisfaction(mut self, satisfaction: f32) -> Self {
        self.user_satisfaction = Some(satisfaction.clamp(0.0, 1.0));
        self
    }

    /// Mark as verified.
    pub fn verified(mut self) -> Self {
        self.verified = true;
        self
    }

    /// Set session ID.
    pub fn session_id(mut self, session_id: impl Into<String>) -> Self {
        self.session_id = Some(session_id.into());
        self
    }

    /// Set agent ID.
    pub fn agent_id(mut self, agent_id: impl Into<String>) -> Self {
        self.agent_id = Some(agent_id.into());
        self
    }

    /// Convert to RewardModifiers.
    fn to_modifiers(&self) -> Option<RewardModifiers> {
        if self.confidence.is_none()
            && self.context_relevance.is_none()
            && self.speed_factor.is_none()
            && self.user_satisfaction.is_none()
            && !self.verified
        {
            return None;
        }

        let mut modifiers = RewardModifiers::new();

        if let Some(c) = self.confidence {
            modifiers = modifiers.with_confidence(c);
        }

        if let Some(cr) = self.context_relevance {
            modifiers = modifiers.with_context_relevance(cr);
        }

        if let Some(sf) = self.speed_factor {
            modifiers = modifiers.with_speed_factor(sf);
        }

        if let Some(us) = self.user_satisfaction {
            modifiers = modifiers.with_user_satisfaction(us);
        }

        if self.verified {
            modifiers = modifiers.verified();
        }

        Some(modifiers)
    }
}

/// Learning and improvement API.
///
/// This API provides methods for the SONA learning loop, including outcome
/// recording, pattern consolidation, self-improvement, and insight aggregation.
#[derive(Clone)]
pub struct LearningApi {
    state: NagualState,
    learner: Arc<SonaLearner>,
}

impl LearningApi {
    /// Create a new LearningApi instance.
    pub(crate) fn new(state: NagualState) -> Self {
        let learner = Arc::new(SonaLearner::with_config(
            state.pattern_storage.clone(),
            state.config.sona_config.clone(),
        ));

        Self { state, learner }
    }

    /// Record the outcome of using a pattern.
    ///
    /// This is the primary method for the SONA learning loop. It records
    /// the outcome of applying a pattern and updates the pattern's reward
    /// and effectiveness metrics.
    ///
    /// # Arguments
    ///
    /// * `pattern_id` - The ID of the pattern that was used
    /// * `outcome` - The result of the pattern application
    /// * `feedback` - Optional feedback or notes about the outcome
    ///
    /// # Returns
    ///
    /// The calculated reward value.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let reward = nagual.learning.record_outcome(
    ///     &pattern_id,
    ///     Outcome::Success,
    ///     Some("Pattern worked perfectly for this use case")
    /// ).await?;
    /// println!("Reward: {}", reward);
    /// ```
    #[instrument(skip(self, feedback), fields(pattern_id = %pattern_id, outcome = %outcome))]
    pub async fn record_outcome(
        &self,
        pattern_id: &str,
        outcome: Outcome,
        feedback: Option<String>,
    ) -> Result<f32> {
        let pattern_id = PatternId::from_string(pattern_id);
        self.learner.record_outcome(&pattern_id, outcome, feedback).await
    }

    /// Record an outcome with additional options.
    ///
    /// # Arguments
    ///
    /// * `pattern_id` - The ID of the pattern that was used
    /// * `outcome` - The result of the pattern application
    /// * `feedback` - Optional feedback or notes about the outcome
    /// * `options` - Additional options for recording
    ///
    /// # Returns
    ///
    /// The calculated reward value.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let reward = nagual.learning.record_outcome_with_options(
    ///     &pattern_id,
    ///     Outcome::Success,
    ///     Some("Pattern worked well"),
    ///     RecordOutcomeOptions::new()
    ///         .confidence(0.95)
    ///         .context_relevance(0.9)
    ///         .verified()
    /// ).await?;
    /// ```
    #[instrument(skip(self, feedback, options), fields(pattern_id = %pattern_id, outcome = %outcome))]
    pub async fn record_outcome_with_options(
        &self,
        pattern_id: &str,
        outcome: Outcome,
        feedback: Option<String>,
        options: RecordOutcomeOptions,
    ) -> Result<f32> {
        let pattern_id = PatternId::from_string(pattern_id);
        let modifiers = options.to_modifiers();

        self.learner
            .record_outcome_with_modifiers(&pattern_id, outcome, feedback, modifiers)
            .await
    }

    /// Run a self-improvement cycle.
    ///
    /// This analyzes patterns and generates recommendations for improvement,
    /// including pattern consolidation, low-reward pattern handling, and
    /// domain-specific optimizations.
    ///
    /// # Arguments
    ///
    /// * `domain` - Optional domain to scope the improvement (e.g., "rust.async")
    ///
    /// # Returns
    ///
    /// An `ImprovementResult` containing the improvement plan.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let result = nagual.learning.improve(Some("rust")).await?;
    /// println!("Generated {} recommendations", result.recommendations_count);
    /// ```
    #[instrument(skip(self))]
    pub async fn improve(&self, domain: Option<&str>) -> Result<ImprovementResult> {
        // Get all patterns (or filtered by domain)
        let patterns = if let Some(d) = domain {
            let category = crate::reasoning_bank::pattern::PatternCategory::from(d);
            self.state
                .pattern_storage
                .get_by_category(&category, 1000)
                .await?
        } else {
            self.state.pattern_storage.get_recent(1000).await?
        };

        let patterns_analyzed = patterns.len();

        // Run self-improvement
        let improver = SelfImprover::new(crate::learning::ImprovementConfig::default());
        let plan = improver.self_improve(&patterns, domain);

        let recommendations_count = plan.recommendations.len();

        info!(
            domain = ?domain,
            patterns_analyzed = patterns_analyzed,
            recommendations = recommendations_count,
            "Self-improvement cycle completed"
        );

        Ok(ImprovementResult {
            plan,
            patterns_analyzed,
            recommendations_count,
            domain: domain.map(String::from),
            timestamp: Utc::now(),
        })
    }

    /// Get insights for a domain.
    ///
    /// Aggregates pattern performance data for the specified domain,
    /// including success rates, trends, and top patterns.
    ///
    /// # Arguments
    ///
    /// * `domain` - Domain to get insights for (use empty string for all domains)
    ///
    /// # Returns
    ///
    /// An `InsightsResult` containing aggregated insights.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let insights = nagual.learning.insights("rust.async").await?;
    /// println!("Success rate: {:.1}%", insights.success_rate * 100.0);
    /// println!("Trend: {}", insights.trend);
    /// ```
    #[instrument(skip(self))]
    pub async fn insights(&self, domain: &str) -> Result<InsightsResult> {
        // Get patterns for the domain
        let patterns = if domain.is_empty() {
            self.state.pattern_storage.get_recent(1000).await?
        } else {
            let category = crate::reasoning_bank::pattern::PatternCategory::from(domain);
            self.state
                .pattern_storage
                .get_by_category(&category, 1000)
                .await?
        };

        let config = InsightsConfig::default();
        let insights = aggregate_insights(&patterns, domain, &config);

        debug!(
            domain = %domain,
            patterns = patterns.len(),
            success_rate = insights.success_rate,
            "Insights generated"
        );

        Ok(InsightsResult::from(insights))
    }

    /// Consolidate similar patterns.
    ///
    /// This merges similar patterns, archives low-reward patterns, and
    /// marks stale patterns for review.
    ///
    /// # Returns
    ///
    /// A `ConsolidationResult` containing consolidation statistics.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let result = nagual.learning.consolidate().await?;
    /// println!("Merged {} patterns", result.patterns_merged);
    /// ```
    #[instrument(skip(self))]
    pub async fn consolidate(&self) -> Result<ConsolidationResult> {
        let config = PatternConsolidationConfig::default();
        let result = consolidate_patterns(&*self.state.pattern_storage, &config).await?;

        let groups_consolidated = result.groups_formed;
        let patterns_merged = result.patterns_consolidated;

        info!(
            groups = groups_consolidated,
            merged = patterns_merged,
            "Consolidation completed"
        );

        Ok(ConsolidationResult {
            groups_consolidated,
            patterns_merged,
            patterns_archived: 0, // Would come from archive operation
            patterns_for_review: 0, // Would come from review operation
            timestamp: Utc::now(),
        })
    }

    /// Get learning statistics.
    ///
    /// # Returns
    ///
    /// Current SONA learning statistics.
    pub fn stats(&self) -> SonaStats {
        self.learner.stats()
    }

    /// Reset learning statistics.
    pub fn reset_stats(&self) {
        self.learner.reset_stats();
    }

    /// Get the SONA configuration.
    pub fn config(&self) -> &SonaConfig {
        self.learner.config()
    }

    /// Record multiple outcomes in a batch.
    ///
    /// # Arguments
    ///
    /// * `outcomes` - Vector of (pattern_id, outcome, feedback) tuples
    ///
    /// # Returns
    ///
    /// Vector of calculated rewards.
    pub async fn record_outcomes_batch(
        &self,
        outcomes: Vec<(String, Outcome, Option<String>)>,
    ) -> Result<Vec<f32>> {
        let converted: Vec<(PatternId, Outcome, Option<String>)> = outcomes
            .into_iter()
            .map(|(id, o, f)| (PatternId::from_string(&id), o, f))
            .collect();

        self.learner.record_outcomes_batch(converted).await
    }

    /// Check if a pattern is performing well.
    ///
    /// # Arguments
    ///
    /// * `pattern_id` - The pattern ID to check
    /// * `min_reward` - Minimum reward threshold (default: 0.6)
    ///
    /// # Returns
    ///
    /// `true` if the pattern's reward is above the threshold.
    pub async fn is_pattern_effective(
        &self,
        pattern_id: &str,
        min_reward: Option<f32>,
    ) -> Result<bool> {
        let pattern_id = PatternId::from_string(pattern_id);
        let pattern = self
            .state
            .pattern_storage
            .get_pattern(&pattern_id)
            .await?
            .ok_or_else(|| NagualError::internal(format!("Pattern not found: {}", pattern_id)))?;

        let threshold = min_reward.unwrap_or(0.6);
        Ok(pattern.reward() >= threshold)
    }

    /// Get patterns that need improvement (low reward).
    ///
    /// # Arguments
    ///
    /// * `max_reward` - Maximum reward threshold (default: 0.4)
    /// * `limit` - Maximum number of patterns to return
    ///
    /// # Returns
    ///
    /// Vector of pattern IDs that need improvement.
    pub async fn patterns_needing_improvement(
        &self,
        max_reward: Option<f32>,
        limit: usize,
    ) -> Result<Vec<String>> {
        let patterns = self.state.pattern_storage.get_recent(limit * 2).await?;
        let threshold = max_reward.unwrap_or(0.4);

        let low_reward: Vec<String> = patterns
            .into_iter()
            .filter(|p| p.reward() < threshold)
            .take(limit)
            .map(|p| p.id().to_string())
            .collect();

        Ok(low_reward)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_record_outcome_options_builder() {
        let options = RecordOutcomeOptions::new()
            .confidence(0.9)
            .context_relevance(0.85)
            .speed_factor(1.0)
            .user_satisfaction(0.95)
            .verified()
            .session_id("session-123")
            .agent_id("agent-456");

        assert_eq!(options.confidence, Some(0.9));
        assert_eq!(options.context_relevance, Some(0.85));
        assert_eq!(options.speed_factor, Some(1.0));
        assert_eq!(options.user_satisfaction, Some(0.95));
        assert!(options.verified);
        assert_eq!(options.session_id, Some("session-123".to_string()));
        assert_eq!(options.agent_id, Some("agent-456".to_string()));
    }

    #[test]
    fn test_to_modifiers_none() {
        let options = RecordOutcomeOptions::default();
        assert!(options.to_modifiers().is_none());
    }

    #[test]
    fn test_to_modifiers_some() {
        let options = RecordOutcomeOptions::new().confidence(0.9);
        assert!(options.to_modifiers().is_some());
    }

    #[test]
    fn test_value_clamping() {
        let options = RecordOutcomeOptions::new()
            .confidence(1.5) // Should clamp to 1.0
            .context_relevance(-0.5); // Should clamp to 0.0

        assert_eq!(options.confidence, Some(1.0));
        assert_eq!(options.context_relevance, Some(0.0));
    }
}
