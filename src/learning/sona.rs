//! SONA Learner - Self-Optimizing Neural Architecture for reward-based learning.
//!
//! Implements the core learning loop with outcome recording, reward calculation,
//! and pattern updates based on feedback signals.

use std::sync::{Arc, OnceLock};

use chrono::{DateTime, Utc};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use tracing::{debug, info, instrument, trace};

use crate::drift::DriftMonitor;
use crate::learning::strange_loop::{MetaCognitiveReport, MetaCognitiveTracker};
use crate::reasoning_bank::pattern::{Pattern, PatternId};
use crate::reasoning_bank::storage::PatternStorage;
use crate::error::{NagualError, Result};

/// Resolve the SQLite database path for persisting learning data.
///
/// Checks (in order):
/// 1. `NAGUAL_DB_PATH` environment variable
/// 2. `sqlite_path` in `~/.nagual/config.toml`
/// 3. Falls back to `nagual.db` (current directory)
fn resolve_db_path() -> String {
    // 1. Environment variable
    if let Ok(path) = std::env::var("NAGUAL_DB_PATH") {
        if !path.is_empty() {
            return path;
        }
    }
    // 2. Config file
    if let Ok(home) = std::env::var("HOME") {
        let config_path = std::path::Path::new(&home)
            .join(".nagual")
            .join("config.toml");
        if let Ok(content) = std::fs::read_to_string(config_path) {
            for line in content.lines() {
                let trimmed = line.trim();
                if trimmed.starts_with("sqlite_path") {
                    if let Some(value) = trimmed.split('=').nth(1) {
                        let path = value.trim().trim_matches('"').trim_matches('\'');
                        if !path.is_empty() {
                            return path.to_string();
                        }
                    }
                }
            }
        }
    }
    // 3. Default
    "nagual.db".to_string()
}

/// Access the process-wide drift monitor.
///
/// The monitor is lazily initialised on first call and shared across all
/// `SonaLearner` instances. It tracks per-domain embedding centroids and
/// computes drift statistics (coefficient of variation of consecutive
/// centroid distances).
fn global_drift_monitor() -> &'static Mutex<DriftMonitor> {
    static INSTANCE: OnceLock<Mutex<DriftMonitor>> = OnceLock::new();
    INSTANCE.get_or_init(|| Mutex::new(DriftMonitor::new()))
}

/// Get drift reports from the global drift monitor for all tracked domains.
pub fn get_drift_reports() -> Vec<crate::drift::DriftReport> {
    global_drift_monitor().lock().all_reports()
}

/// Get the drift report for a specific domain.
///
/// Returns `None` if the domain has never been recorded or does not exist
/// in the monitor.
pub fn get_domain_drift(domain: &str) -> Option<crate::drift::DriftReport> {
    global_drift_monitor().lock().compute_drift(domain)
}

/// Access the process-wide meta-cognitive tracker.
///
/// The tracker accumulates [`MetaCognitiveReport`]s produced by the
/// strange-loop evaluation that runs on every SONA outcome recording.
fn global_meta_tracker() -> &'static Mutex<MetaCognitiveTracker> {
    static INSTANCE: OnceLock<Mutex<MetaCognitiveTracker>> = OnceLock::new();
    INSTANCE.get_or_init(|| Mutex::new(MetaCognitiveTracker::new()))
}

/// Get the latest meta-cognitive evaluation report.
pub fn get_meta_cognitive_status() -> Option<MetaCognitiveReport> {
    global_meta_tracker().lock().latest().cloned()
}

/// Get aggregate meta-cognitive statistics.
///
/// Returns `(avg_quality, health_rate, evaluation_count)`.
pub fn get_meta_cognitive_stats() -> (f64, f64, usize) {
    let tracker = global_meta_tracker().lock();
    (tracker.avg_quality(), tracker.health_rate(), tracker.count())
}

/// Outcome of a pattern application.
///
/// Represents the result of applying a pattern to a task, ranging from
/// complete success to failure, with intermediate states.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Outcome {
    /// Pattern application was fully successful
    Success,
    /// Pattern was partially successful (some goals achieved)
    PartialSuccess,
    /// Pattern application had neutral results (no clear success or failure)
    Neutral,
    /// Pattern application failed
    Failure,
}

impl Outcome {
    /// Get the base reward value for this outcome.
    pub fn base_reward(&self) -> f32 {
        match self {
            Outcome::Success => 0.9,
            Outcome::PartialSuccess => 0.7,
            Outcome::Neutral => 0.5,
            Outcome::Failure => 0.2,
        }
    }

    /// Check if this outcome indicates success (Success or PartialSuccess).
    pub fn is_successful(&self) -> bool {
        matches!(self, Outcome::Success | Outcome::PartialSuccess)
    }

    /// Convert from a boolean success flag.
    pub fn from_success(success: bool) -> Self {
        if success {
            Outcome::Success
        } else {
            Outcome::Failure
        }
    }
}

impl std::fmt::Display for Outcome {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Outcome::Success => write!(f, "success"),
            Outcome::PartialSuccess => write!(f, "partial_success"),
            Outcome::Neutral => write!(f, "neutral"),
            Outcome::Failure => write!(f, "failure"),
        }
    }
}

impl From<&str> for Outcome {
    fn from(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "success" => Outcome::Success,
            "partial_success" | "partial" => Outcome::PartialSuccess,
            "neutral" => Outcome::Neutral,
            "failure" | "failed" => Outcome::Failure,
            _ => Outcome::Neutral,
        }
    }
}

/// Modifiers that affect the final reward calculation.
///
/// These allow fine-tuning the reward based on context and confidence.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RewardModifiers {
    /// Confidence in the outcome assessment (0.0 - 1.0).
    /// Higher confidence amplifies reward deviation from neutral.
    #[serde(default = "default_confidence")]
    pub confidence: f32,

    /// Context relevance (0.0 - 1.0).
    /// How well the pattern matched the context where it was applied.
    #[serde(default = "default_context_relevance")]
    pub context_relevance: f32,

    /// Speed of execution (0.0 - 1.0).
    /// 1.0 = completed faster than expected, 0.5 = as expected, 0.0 = much slower.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speed_factor: Option<f32>,

    /// User satisfaction rating if available (0.0 - 1.0).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user_satisfaction: Option<f32>,

    /// Whether the outcome was verified (increases confidence in reward).
    #[serde(default)]
    pub verified: bool,
}

fn default_confidence() -> f32 {
    0.8
}

fn default_context_relevance() -> f32 {
    1.0
}

impl RewardModifiers {
    /// Create new modifiers with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the confidence level.
    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = confidence.clamp(0.0, 1.0);
        self
    }

    /// Set the context relevance.
    pub fn with_context_relevance(mut self, relevance: f32) -> Self {
        self.context_relevance = relevance.clamp(0.0, 1.0);
        self
    }

    /// Set the speed factor.
    pub fn with_speed_factor(mut self, speed: f32) -> Self {
        self.speed_factor = Some(speed.clamp(0.0, 1.0));
        self
    }

    /// Set user satisfaction.
    pub fn with_user_satisfaction(mut self, satisfaction: f32) -> Self {
        self.user_satisfaction = Some(satisfaction.clamp(0.0, 1.0));
        self
    }

    /// Mark as verified.
    pub fn verified(mut self) -> Self {
        self.verified = true;
        self
    }

    /// Calculate the combined modifier effect.
    ///
    /// Returns a multiplier in the range [0.5, 1.5] that adjusts the base reward.
    pub fn combined_modifier(&self) -> f32 {
        let mut modifier = 1.0;

        // Confidence affects how much we trust this outcome
        // High confidence (> 0.8) slightly boosts, low confidence reduces
        modifier *= 0.7 + (self.confidence * 0.4);

        // Context relevance affects how applicable this outcome is
        modifier *= 0.8 + (self.context_relevance * 0.2);

        // Speed factor gives a small bonus/penalty
        if let Some(speed) = self.speed_factor {
            modifier *= 0.95 + (speed * 0.1);
        }

        // User satisfaction is highly weighted if provided
        if let Some(satisfaction) = self.user_satisfaction {
            modifier *= 0.8 + (satisfaction * 0.4);
        }

        // Verification bonus
        if self.verified {
            modifier *= 1.05;
        }

        // Clamp to reasonable range
        modifier.clamp(0.5, 1.5)
    }
}

/// Calculate the final reward value based on outcome and modifiers.
///
/// # Formula
///
/// ```text
/// final_reward = clamp(base_reward * modifier, 0.0, 1.0)
/// ```
///
/// Where:
/// - `base_reward` is determined by the Outcome type
/// - `modifier` is the combined effect of all RewardModifiers
///
/// # Example
///
/// ```ignore
/// let reward = calculate_reward(
///     Outcome::Success,
///     Some(RewardModifiers::new()
///         .with_confidence(0.95)
///         .with_context_relevance(0.9)
///         .verified())
/// );
/// assert!(reward > 0.9); // High confidence success gives bonus
/// ```
pub fn calculate_reward(outcome: Outcome, modifiers: Option<RewardModifiers>) -> f32 {
    let base = outcome.base_reward();

    let modifier = modifiers
        .map(|m| m.combined_modifier())
        .unwrap_or(1.0);

    // Apply modifier but keep reward in valid range
    (base * modifier).clamp(0.0, 1.0)
}

/// Record of a single outcome event for logging and analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutcomeRecord {
    /// The pattern that was applied
    pub pattern_id: PatternId,

    /// The outcome of the application
    pub outcome: Outcome,

    /// Calculated reward value
    pub reward: f32,

    /// Optional feedback or notes
    pub feedback: Option<String>,

    /// When this outcome was recorded
    pub recorded_at: DateTime<Utc>,

    /// Modifiers that were applied (if any)
    pub modifiers: Option<RewardModifiers>,

    /// Session or context ID
    pub session_id: Option<String>,

    /// Agent that recorded the outcome
    pub agent_id: Option<String>,
}

impl OutcomeRecord {
    /// Create a new outcome record.
    pub fn new(pattern_id: PatternId, outcome: Outcome, reward: f32) -> Self {
        Self {
            pattern_id,
            outcome,
            reward,
            feedback: None,
            recorded_at: Utc::now(),
            modifiers: None,
            session_id: None,
            agent_id: None,
        }
    }

    /// Set the feedback.
    pub fn with_feedback(mut self, feedback: impl Into<String>) -> Self {
        self.feedback = Some(feedback.into());
        self
    }

    /// Set the modifiers.
    pub fn with_modifiers(mut self, modifiers: RewardModifiers) -> Self {
        self.modifiers = Some(modifiers);
        self
    }

    /// Set the session ID.
    pub fn with_session_id(mut self, session_id: impl Into<String>) -> Self {
        self.session_id = Some(session_id.into());
        self
    }

    /// Set the agent ID.
    pub fn with_agent_id(mut self, agent_id: impl Into<String>) -> Self {
        self.agent_id = Some(agent_id.into());
        self
    }
}

/// Log of outcome records for a pattern.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OutcomeLog {
    /// All recorded outcomes
    records: Vec<OutcomeRecord>,

    /// Running statistics
    pub total_count: u32,
    pub success_count: u32,
    pub failure_count: u32,
    pub average_reward: f32,
}

impl OutcomeLog {
    /// Create a new empty outcome log.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add an outcome record.
    pub fn add(&mut self, record: OutcomeRecord) {
        self.total_count += 1;

        if record.outcome.is_successful() {
            self.success_count += 1;
        } else if record.outcome == Outcome::Failure {
            self.failure_count += 1;
        }

        // Update running average
        let n = self.total_count as f32;
        self.average_reward = ((n - 1.0) * self.average_reward + record.reward) / n;

        self.records.push(record);
    }

    /// Get all records.
    pub fn records(&self) -> &[OutcomeRecord] {
        &self.records
    }

    /// Get recent records (last N).
    pub fn recent(&self, n: usize) -> &[OutcomeRecord] {
        let start = self.records.len().saturating_sub(n);
        &self.records[start..]
    }

    /// Get the success rate.
    pub fn success_rate(&self) -> f32 {
        if self.total_count == 0 {
            0.0
        } else {
            self.success_count as f32 / self.total_count as f32
        }
    }

    /// Calculate a weighted recent reward (more recent outcomes weighted higher).
    pub fn weighted_recent_reward(&self, window: usize) -> f32 {
        let recent = self.recent(window);
        if recent.is_empty() {
            return self.average_reward;
        }

        let mut total_weight = 0.0;
        let mut weighted_sum = 0.0;

        for (i, record) in recent.iter().enumerate() {
            // More recent records get higher weight
            let weight = (i + 1) as f32;
            weighted_sum += record.reward * weight;
            total_weight += weight;
        }

        if total_weight > 0.0 {
            weighted_sum / total_weight
        } else {
            self.average_reward
        }
    }
}

/// Configuration for the SONA learner.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SonaConfig {
    /// Learning rate for reward updates (0.0 - 1.0).
    /// Higher values make pattern rewards change faster.
    #[serde(default = "default_learning_rate")]
    pub learning_rate: f32,

    /// Minimum samples before updating pattern reward.
    #[serde(default = "default_min_samples")]
    pub min_samples_for_update: u32,

    /// Whether to use exponential moving average for rewards.
    #[serde(default = "default_use_ema")]
    pub use_ema: bool,

    /// EMA decay factor (0.0 - 1.0, higher = slower decay).
    #[serde(default = "default_ema_decay")]
    pub ema_decay: f32,

    /// Whether to log all outcomes for analysis.
    #[serde(default)]
    pub log_outcomes: bool,

    /// Maximum outcome log entries per pattern.
    #[serde(default = "default_max_log_entries")]
    pub max_log_entries: usize,
}

fn default_learning_rate() -> f32 {
    0.1
}

fn default_min_samples() -> u32 {
    1
}

fn default_use_ema() -> bool {
    true
}

fn default_ema_decay() -> f32 {
    0.9
}

fn default_max_log_entries() -> usize {
    100
}

impl Default for SonaConfig {
    fn default() -> Self {
        Self {
            learning_rate: default_learning_rate(),
            min_samples_for_update: default_min_samples(),
            use_ema: default_use_ema(),
            ema_decay: default_ema_decay(),
            log_outcomes: false,
            max_log_entries: default_max_log_entries(),
        }
    }
}

/// Statistics from SONA learning operations.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SonaStats {
    /// Total outcomes recorded
    pub outcomes_recorded: u64,

    /// Patterns updated
    pub patterns_updated: u64,

    /// Average reward across all outcomes
    pub average_reward: f32,

    /// Success rate across all outcomes
    pub success_rate: f32,

    /// Outcomes by type
    pub success_count: u64,
    pub partial_success_count: u64,
    pub neutral_count: u64,
    pub failure_count: u64,
}

impl SonaStats {
    /// Record a new outcome in stats.
    pub fn record_outcome(&mut self, outcome: Outcome, reward: f32) {
        self.outcomes_recorded += 1;

        match outcome {
            Outcome::Success => self.success_count += 1,
            Outcome::PartialSuccess => self.partial_success_count += 1,
            Outcome::Neutral => self.neutral_count += 1,
            Outcome::Failure => self.failure_count += 1,
        }

        // Update running averages
        let n = self.outcomes_recorded as f32;
        self.average_reward = ((n - 1.0) * self.average_reward + reward) / n;

        let successful = (self.success_count + self.partial_success_count) as f32;
        self.success_rate = successful / n;
    }
}

/// SONA Learner - manages the learning loop for pattern optimization.
///
/// The learner tracks outcomes from pattern applications and updates
/// pattern quality metrics to enable self-improvement over time.
pub struct SonaLearner {
    /// Pattern storage reference
    storage: Arc<PatternStorage>,

    /// Configuration
    config: SonaConfig,

    /// Outcome logs per pattern (in-memory cache)
    outcome_logs: parking_lot::RwLock<hashbrown::HashMap<String, OutcomeLog>>,

    /// Running statistics
    stats: parking_lot::RwLock<SonaStats>,
}

impl SonaLearner {
    /// Create a new SONA learner with default configuration.
    pub fn new(storage: Arc<PatternStorage>) -> Self {
        Self::with_config(storage, SonaConfig::default())
    }

    /// Create a new SONA learner with custom configuration.
    pub fn with_config(storage: Arc<PatternStorage>, config: SonaConfig) -> Self {
        Self {
            storage,
            config,
            outcome_logs: parking_lot::RwLock::new(hashbrown::HashMap::new()),
            stats: parking_lot::RwLock::new(SonaStats::default()),
        }
    }

    /// Get the configuration.
    pub fn config(&self) -> &SonaConfig {
        &self.config
    }

    /// Get current statistics.
    pub fn stats(&self) -> SonaStats {
        self.stats.read().clone()
    }

    /// Record an outcome for a pattern application.
    ///
    /// This is the primary method for the learning loop. It:
    /// 1. Calculates the reward based on outcome and modifiers
    /// 2. Updates the pattern's reward and success fields
    /// 3. Logs the outcome for analysis
    ///
    /// # Arguments
    ///
    /// * `pattern_id` - The pattern that was applied
    /// * `outcome` - The result of the application
    /// * `feedback` - Optional feedback or notes about the outcome
    ///
    /// # Returns
    ///
    /// The calculated reward value
    ///
    /// # Example
    ///
    /// ```ignore
    /// let reward = learner.record_outcome(
    ///     &pattern_id,
    ///     Outcome::Success,
    ///     Some("Pattern worked well for caching scenario".to_string()),
    /// ).await?;
    /// ```
    #[instrument(skip(self, feedback), fields(pattern_id = %pattern_id, outcome = %outcome))]
    pub async fn record_outcome(
        &self,
        pattern_id: &PatternId,
        outcome: Outcome,
        feedback: Option<String>,
    ) -> Result<f32> {
        self.record_outcome_with_modifiers(pattern_id, outcome, feedback, None)
            .await
    }

    /// Record an outcome with custom reward modifiers.
    ///
    /// Use this method when you have additional context about the outcome
    /// that should affect the reward calculation.
    #[instrument(skip(self, feedback, modifiers), fields(pattern_id = %pattern_id, outcome = %outcome))]
    pub async fn record_outcome_with_modifiers(
        &self,
        pattern_id: &PatternId,
        outcome: Outcome,
        feedback: Option<String>,
        modifiers: Option<RewardModifiers>,
    ) -> Result<f32> {
        // Calculate reward
        let reward = calculate_reward(outcome, modifiers.clone());

        debug!(
            pattern_id = %pattern_id,
            outcome = %outcome,
            reward = reward,
            "Recording outcome"
        );

        // Get the pattern
        let pattern = self
            .storage
            .get_pattern(pattern_id)
            .await?
            .ok_or_else(|| NagualError::Internal {
                message: format!("Pattern not found: {}", pattern_id),
            })?;

        // Record embedding in the drift monitor so we can detect domain drift.
        if let Some(embedding) = pattern.embedding() {
            let domain = pattern.category().to_string();
            trace!(
                pattern_id = %pattern_id,
                domain = %domain,
                "Recording embedding to drift monitor"
            );
            global_drift_monitor()
                .lock()
                .record(&domain, embedding.to_vec());
        }

        // Update pattern with new reward
        let mut updated_pattern = pattern.clone();
        self.update_pattern_reward(&mut updated_pattern, outcome, reward);

        // Update Bayesian quality score (Beta distribution)
        match outcome {
            Outcome::Success | Outcome::PartialSuccess => {
                updated_pattern.bayesian_score_mut().upvote();
            }
            Outcome::Failure => {
                updated_pattern.bayesian_score_mut().downvote();
            }
            Outcome::Neutral => {
                // Neutral outcomes do not update the Bayesian score
            }
        }

        // Meta-cognitive evaluation via strange-loop
        let outcome_quality = match outcome {
            Outcome::Success => 1.0,
            Outcome::PartialSuccess => 0.7,
            Outcome::Neutral => 0.5,
            Outcome::Failure => 0.0,
        };
        let meta_report = crate::learning::strange_loop::evaluate_quality(
            pattern.reward() as f64,
            outcome_quality,
        );
        debug!(
            quality = meta_report.quality_score,
            healthy = meta_report.is_healthy,
            bonus = meta_report.bonus,
            "Strange loop meta-cognitive evaluation"
        );
        global_meta_tracker().lock().record(meta_report.clone());

        // Persist meta-cognitive report and drift data to SQLite so CLI
        // commands (`nagual learn strange-loop`, `nagual learn drift`) can
        // show historical data even after the process exits.
        let db_path = resolve_db_path();
        if let Err(e) = crate::learning::strange_loop::persist_report(&db_path, &meta_report) {
            debug!(error = %e, "Failed to persist meta-cognitive report to SQLite");
        }
        // Persist drift report for the pattern's domain if we recorded an embedding
        if pattern.embedding().is_some() {
            let domain = pattern.category().to_string();
            let monitor = global_drift_monitor().lock();
            if let Some(drift_report) = monitor.compute_drift(&domain) {
                if let Err(e) = crate::drift::persist_drift_report(&db_path, &drift_report) {
                    debug!(error = %e, "Failed to persist drift report to SQLite");
                }
            }
        }

        // Domain expansion tracking (Meta Thompson Sampling).
        // Maps the pattern category to a domain and records the outcome
        // so the expansion engine can learn per-difficulty-tier strategies.
        //
        // HIGH-4 fix: difficulty proxy was inverted. Previously used
        // pattern.effectiveness() directly, which means high-effectiveness
        // patterns were classified as "hard". Difficulty should be the
        // inverse: high-reward patterns are "easy" to apply successfully.
        {
            let domain = pattern.category().to_string();
            let difficulty = 1.0 - pattern.reward();
            crate::learning::domain_expansion::record_domain_outcome(
                &domain,
                &pattern.id().to_string(),
                reward,
                difficulty,
            );
        }

        // Update success field
        updated_pattern.set_success(outcome.is_successful());

        // Update critique with feedback if provided
        if let Some(ref fb) = feedback {
            let existing_critique = updated_pattern.critique();
            let new_critique = if existing_critique.is_empty() {
                fb.clone()
            } else {
                format!("{}\n---\n{}", existing_critique, fb)
            };
            updated_pattern.set_critique(new_critique);
        }

        // Increment reuse count
        updated_pattern.increment_reuse_count();

        // Save updated pattern
        self.storage.update_pattern(&updated_pattern).await?;

        // Create outcome record
        let mut record = OutcomeRecord::new(pattern_id.clone(), outcome, reward);
        if let Some(fb) = feedback {
            record = record.with_feedback(fb);
        }
        if let Some(mods) = modifiers {
            record = record.with_modifiers(mods);
        }

        // Log the outcome
        self.log_outcome(pattern_id, record);

        // Update global stats
        self.stats.write().record_outcome(outcome, reward);
        self.stats.write().patterns_updated += 1;

        info!(
            pattern_id = %pattern_id,
            outcome = %outcome,
            new_reward = updated_pattern.reward(),
            reuse_count = updated_pattern.reuse_count(),
            "Pattern updated with outcome"
        );

        Ok(reward)
    }

    /// Update a pattern's reward based on the new outcome.
    fn update_pattern_reward(&self, pattern: &mut Pattern, outcome: Outcome, new_reward: f32) {
        let current_reward = pattern.reward();

        let updated_reward = if self.config.use_ema {
            // Exponential moving average
            let decay = self.config.ema_decay;
            decay * current_reward + (1.0 - decay) * new_reward
        } else {
            // Simple learning rate update
            let lr = self.config.learning_rate;
            current_reward + lr * (new_reward - current_reward)
        };

        pattern.set_reward(updated_reward);

        // Also update effectiveness based on outcome
        let current_effectiveness = pattern.effectiveness();
        let outcome_effectiveness = match outcome {
            Outcome::Success => 1.0,
            Outcome::PartialSuccess => 0.75,
            Outcome::Neutral => 0.5,
            Outcome::Failure => 0.25,
        };

        let updated_effectiveness = if self.config.use_ema {
            let decay = self.config.ema_decay;
            decay * current_effectiveness + (1.0 - decay) * outcome_effectiveness
        } else {
            let lr = self.config.learning_rate;
            current_effectiveness + lr * (outcome_effectiveness - current_effectiveness)
        };

        pattern.set_effectiveness(updated_effectiveness);
    }

    /// Log an outcome record.
    fn log_outcome(&self, pattern_id: &PatternId, record: OutcomeRecord) {
        if !self.config.log_outcomes {
            return;
        }

        let mut logs = self.outcome_logs.write();
        let log = logs
            .entry(pattern_id.to_string())
            .or_insert_with(OutcomeLog::new);

        log.add(record);

        // Trim log if too large
        while log.records.len() > self.config.max_log_entries {
            log.records.remove(0);
        }
    }

    /// Get the outcome log for a pattern.
    pub fn get_outcome_log(&self, pattern_id: &PatternId) -> Option<OutcomeLog> {
        self.outcome_logs
            .read()
            .get(&pattern_id.to_string())
            .cloned()
    }

    /// Get all outcome logs.
    pub fn get_all_outcome_logs(&self) -> hashbrown::HashMap<String, OutcomeLog> {
        self.outcome_logs.read().clone()
    }

    /// Batch record multiple outcomes.
    ///
    /// More efficient than recording outcomes one by one.
    pub async fn record_outcomes_batch(
        &self,
        outcomes: Vec<(PatternId, Outcome, Option<String>)>,
    ) -> Result<Vec<f32>> {
        let mut rewards = Vec::with_capacity(outcomes.len());

        for (pattern_id, outcome, feedback) in outcomes {
            let reward = self.record_outcome(&pattern_id, outcome, feedback).await?;
            rewards.push(reward);
        }

        Ok(rewards)
    }

    /// Clear outcome logs (for memory management).
    pub fn clear_outcome_logs(&self) {
        self.outcome_logs.write().clear();
    }

    /// Reset statistics.
    pub fn reset_stats(&self) {
        *self.stats.write() = SonaStats::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_outcome_base_reward() {
        assert_eq!(Outcome::Success.base_reward(), 0.9);
        assert_eq!(Outcome::PartialSuccess.base_reward(), 0.7);
        assert_eq!(Outcome::Neutral.base_reward(), 0.5);
        assert_eq!(Outcome::Failure.base_reward(), 0.2);
    }

    #[test]
    fn test_outcome_is_successful() {
        assert!(Outcome::Success.is_successful());
        assert!(Outcome::PartialSuccess.is_successful());
        assert!(!Outcome::Neutral.is_successful());
        assert!(!Outcome::Failure.is_successful());
    }

    #[test]
    fn test_outcome_from_str() {
        assert_eq!(Outcome::from("success"), Outcome::Success);
        assert_eq!(Outcome::from("partial_success"), Outcome::PartialSuccess);
        assert_eq!(Outcome::from("partial"), Outcome::PartialSuccess);
        assert_eq!(Outcome::from("neutral"), Outcome::Neutral);
        assert_eq!(Outcome::from("failure"), Outcome::Failure);
        assert_eq!(Outcome::from("unknown"), Outcome::Neutral);
    }

    #[test]
    fn test_calculate_reward_basic() {
        let reward = calculate_reward(Outcome::Success, None);
        assert_eq!(reward, 0.9);

        let reward = calculate_reward(Outcome::Failure, None);
        assert_eq!(reward, 0.2);
    }

    #[test]
    fn test_calculate_reward_with_modifiers() {
        // High confidence success should give bonus
        let modifiers = RewardModifiers::new()
            .with_confidence(0.95)
            .with_context_relevance(1.0)
            .verified();

        let reward = calculate_reward(Outcome::Success, Some(modifiers));
        assert!(reward > 0.9);
        assert!(reward <= 1.0);

        // Low confidence failure should reduce reward less
        let modifiers = RewardModifiers::new()
            .with_confidence(0.3)
            .with_context_relevance(0.5);

        let reward = calculate_reward(Outcome::Failure, Some(modifiers));
        assert!(reward < 0.2);
    }

    #[test]
    fn test_reward_modifiers_combined() {
        let modifiers = RewardModifiers::new()
            .with_confidence(1.0)
            .with_context_relevance(1.0)
            .with_speed_factor(1.0)
            .with_user_satisfaction(1.0)
            .verified();

        let modifier = modifiers.combined_modifier();
        assert!(modifier > 1.0);
        assert!(modifier <= 1.5);

        let low_modifiers = RewardModifiers::new()
            .with_confidence(0.1)
            .with_context_relevance(0.1);

        let low_modifier = low_modifiers.combined_modifier();
        assert!(low_modifier >= 0.5);
        assert!(low_modifier < 1.0);
    }

    #[test]
    fn test_outcome_log() {
        let mut log = OutcomeLog::new();

        assert_eq!(log.total_count, 0);
        assert_eq!(log.success_rate(), 0.0);

        // Add some outcomes
        log.add(OutcomeRecord::new(PatternId::new(), Outcome::Success, 0.9));
        log.add(OutcomeRecord::new(PatternId::new(), Outcome::Success, 0.85));
        log.add(OutcomeRecord::new(PatternId::new(), Outcome::Failure, 0.2));

        assert_eq!(log.total_count, 3);
        assert_eq!(log.success_count, 2);
        assert_eq!(log.failure_count, 1);
        assert!((log.success_rate() - 0.666).abs() < 0.01);
        assert!((log.average_reward - 0.65).abs() < 0.01);
    }

    #[test]
    fn test_outcome_log_weighted_recent() {
        let mut log = OutcomeLog::new();

        // Old outcomes (will be weighted less)
        log.add(OutcomeRecord::new(PatternId::new(), Outcome::Failure, 0.2));
        log.add(OutcomeRecord::new(PatternId::new(), Outcome::Failure, 0.2));

        // Recent outcomes (will be weighted more)
        log.add(OutcomeRecord::new(PatternId::new(), Outcome::Success, 0.9));
        log.add(OutcomeRecord::new(PatternId::new(), Outcome::Success, 0.9));

        let weighted = log.weighted_recent_reward(4);

        // Should be higher than simple average because recent successes are weighted more
        assert!(weighted > log.average_reward);
    }

    #[test]
    fn test_sona_config_default() {
        let config = SonaConfig::default();
        assert_eq!(config.learning_rate, 0.1);
        assert_eq!(config.min_samples_for_update, 1);
        assert!(config.use_ema);
        assert_eq!(config.ema_decay, 0.9);
        assert!(!config.log_outcomes);
    }

    #[test]
    fn test_sona_stats() {
        let mut stats = SonaStats::default();

        stats.record_outcome(Outcome::Success, 0.9);
        stats.record_outcome(Outcome::Success, 0.85);
        stats.record_outcome(Outcome::Failure, 0.2);

        assert_eq!(stats.outcomes_recorded, 3);
        assert_eq!(stats.success_count, 2);
        assert_eq!(stats.failure_count, 1);
        assert!((stats.success_rate - 0.666).abs() < 0.01);
    }

    #[test]
    fn test_outcome_record_builder() {
        let record = OutcomeRecord::new(PatternId::from_string("test-123"), Outcome::Success, 0.9)
            .with_feedback("Great performance")
            .with_session_id("session-456")
            .with_agent_id("agent-789")
            .with_modifiers(RewardModifiers::new().with_confidence(0.95));

        assert_eq!(record.pattern_id.as_str(), "test-123");
        assert_eq!(record.outcome, Outcome::Success);
        assert_eq!(record.feedback, Some("Great performance".to_string()));
        assert_eq!(record.session_id, Some("session-456".to_string()));
        assert_eq!(record.agent_id, Some("agent-789".to_string()));
        assert!(record.modifiers.is_some());
    }

    #[test]
    fn test_global_drift_monitor_accessible() {
        // Verify the lazy global drift monitor can be created and queried.
        let reports = get_drift_reports();
        // No domains recorded yet in the global monitor (or from other tests).
        // Just verify it doesn't panic.
        let _ = reports;
    }

    #[test]
    fn test_get_domain_drift_none_for_unknown() {
        assert!(get_domain_drift("nonexistent_domain_sona_test").is_none());
    }

    #[test]
    fn test_drift_monitor_records() {
        // Manually record into the global monitor and verify retrieval
        {
            let mut monitor = super::global_drift_monitor().lock();
            monitor.record("sona_test_domain", vec![1.0, 0.0, 0.0]);
            monitor.record("sona_test_domain", vec![1.0, 0.1, 0.0]);
            monitor.record("sona_test_domain", vec![1.0, 0.2, 0.0]);
        }

        let report = get_domain_drift("sona_test_domain");
        assert!(report.is_some(), "Should have drift report after recording");
        let report = report.unwrap();
        assert_eq!(report.domain, "sona_test_domain");
        assert!(report.window_size >= 3);
    }
}
