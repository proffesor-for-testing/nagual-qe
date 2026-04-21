//! Scenario Holdout System for pattern validation.
//!
//! This module implements the StrongDM "holdout set" concept - validation scenarios
//! stored outside the pattern training data to prevent overfitting and enable
//! probabilistic pattern validation.
//!
//! # Overview
//!
//! Scenarios are test cases that patterns are evaluated against:
//! - Each scenario describes a problem context and expected behavior
//! - Patterns are scored based on how well they address scenarios
//! - Holdout scenarios are specifically reserved for validation (patterns haven't "seen" them)
//!
//! # Example
//!
//! ```ignore
//! use nagual::learning::scenario::{Scenario, ScenarioEvaluator, Difficulty};
//!
//! // Create a scenario
//! let scenario = Scenario::new("rust")
//!     .with_description("Handle async timeout gracefully")
//!     .with_input_context("Long-running async operation that may exceed timeout")
//!     .with_expected_behavior("Should return error with timeout context, not panic")
//!     .with_difficulty(Difficulty::Medium)
//!     .as_holdout(true)
//!     .build();
//!
//! // Evaluate a pattern against the scenario
//! let evaluator = ScenarioEvaluator::new(storage);
//! let eval = evaluator.evaluate(&pattern, &scenario).await?;
//! println!("Score: {}, Passed: {}", eval.score, eval.passed);
//! ```

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::Result;
use crate::reasoning_bank::pattern::Pattern;

/// SQL table name for scenarios.
pub const SQLITE_SCENARIOS_TABLE: &str = "scenarios";

/// SQL table name for scenario evaluations.
pub const SQLITE_SCENARIO_EVALUATIONS_TABLE: &str = "scenario_evaluations";

/// Unique identifier for a scenario.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ScenarioId(pub String);

impl ScenarioId {
    /// Create a new random scenario ID.
    pub fn new() -> Self {
        Self(format!("scen_{}", Uuid::new_v4()))
    }

    /// Create a scenario ID from an existing string.
    pub fn from_string(s: impl Into<String>) -> Self {
        Self(s.into())
    }

    /// Get the underlying string.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Default for ScenarioId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for ScenarioId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<String> for ScenarioId {
    fn from(s: String) -> Self {
        Self(s)
    }
}

impl From<&str> for ScenarioId {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

/// Difficulty level for a scenario.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum Difficulty {
    /// Easy scenario - basic functionality
    Easy,
    /// Medium scenario - standard complexity
    #[default]
    Medium,
    /// Hard scenario - edge cases and complex interactions
    Hard,
}

impl Difficulty {
    /// Get string representation for database storage.
    pub fn as_str(&self) -> &'static str {
        match self {
            Difficulty::Easy => "easy",
            Difficulty::Medium => "medium",
            Difficulty::Hard => "hard",
        }
    }

    /// Parse from string.
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "easy" | "simple" | "basic" => Difficulty::Easy,
            "medium" | "normal" | "standard" => Difficulty::Medium,
            "hard" | "complex" | "advanced" => Difficulty::Hard,
            _ => Difficulty::Medium,
        }
    }

    /// Get a numeric weight for this difficulty level.
    pub fn weight(&self) -> f32 {
        match self {
            Difficulty::Easy => 1.0,
            Difficulty::Medium => 1.5,
            Difficulty::Hard => 2.0,
        }
    }
}

impl std::fmt::Display for Difficulty {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// A validation scenario - a test case for pattern evaluation.
///
/// Scenarios represent specific situations that patterns should be able to address.
/// They serve as a holdout set for validating pattern quality without overfitting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Scenario {
    /// Unique identifier for this scenario
    pub id: ScenarioId,

    /// Domain this scenario belongs to (e.g., "rust.async", "database.postgres")
    pub domain: String,

    /// Short description of the scenario
    pub description: String,

    /// The problem/situation context - what is the user facing?
    pub input_context: String,

    /// What a good solution should do/achieve
    pub expected_behavior: String,

    /// Difficulty level of this scenario
    pub difficulty: Difficulty,

    /// When the scenario was created
    pub created_at: DateTime<Utc>,

    /// Last time the scenario was used for evaluation
    pub last_evaluated: Option<DateTime<Utc>>,

    /// Number of times patterns have passed this scenario
    pub pass_count: u32,

    /// Number of times patterns have failed this scenario
    pub fail_count: u32,

    /// If true, this scenario is reserved for holdout validation
    /// Patterns haven't "seen" this during training/consolidation
    pub is_holdout: bool,

    /// Tags for filtering and grouping scenarios
    #[serde(default)]
    pub tags: Vec<String>,
}

impl Scenario {
    /// Create a new scenario builder for a domain.
    pub fn new(domain: impl Into<String>) -> ScenarioBuilder {
        ScenarioBuilder::new(domain)
    }

    /// Get the pass rate for this scenario.
    pub fn pass_rate(&self) -> f32 {
        let total = self.pass_count + self.fail_count;
        if total == 0 {
            0.5 // No evaluations yet, neutral
        } else {
            self.pass_count as f32 / total as f32
        }
    }

    /// Get the total number of evaluations for this scenario.
    pub fn total_evaluations(&self) -> u32 {
        self.pass_count + self.fail_count
    }

    /// Calculate a difficulty-weighted pass rate.
    pub fn weighted_pass_rate(&self) -> f32 {
        self.pass_rate() * self.difficulty.weight()
    }
}

/// Builder for creating Scenario instances.
#[derive(Debug, Default)]
pub struct ScenarioBuilder {
    id: Option<ScenarioId>,
    domain: String,
    description: Option<String>,
    input_context: Option<String>,
    expected_behavior: Option<String>,
    difficulty: Option<Difficulty>,
    is_holdout: Option<bool>,
    tags: Option<Vec<String>>,
}

impl ScenarioBuilder {
    /// Create a new scenario builder for a domain.
    pub fn new(domain: impl Into<String>) -> Self {
        Self {
            domain: domain.into(),
            ..Default::default()
        }
    }

    /// Set a custom ID.
    pub fn id(mut self, id: impl Into<ScenarioId>) -> Self {
        self.id = Some(id.into());
        self
    }

    /// Set the description.
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Set the input context (the problem/situation).
    pub fn with_input_context(mut self, context: impl Into<String>) -> Self {
        self.input_context = Some(context.into());
        self
    }

    /// Set the expected behavior.
    pub fn with_expected_behavior(mut self, expected: impl Into<String>) -> Self {
        self.expected_behavior = Some(expected.into());
        self
    }

    /// Set the difficulty level.
    pub fn with_difficulty(mut self, difficulty: Difficulty) -> Self {
        self.difficulty = Some(difficulty);
        self
    }

    /// Set whether this is a holdout scenario.
    pub fn as_holdout(mut self, is_holdout: bool) -> Self {
        self.is_holdout = Some(is_holdout);
        self
    }

    /// Set the tags.
    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.tags = Some(tags);
        self
    }

    /// Add a single tag.
    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.get_or_insert_with(Vec::new).push(tag.into());
        self
    }

    /// Build the scenario.
    pub fn build(self) -> Scenario {
        Scenario {
            id: self.id.unwrap_or_else(ScenarioId::new),
            domain: self.domain,
            description: self.description.unwrap_or_else(|| "Unnamed scenario".to_string()),
            input_context: self.input_context.unwrap_or_default(),
            expected_behavior: self.expected_behavior.unwrap_or_default(),
            difficulty: self.difficulty.unwrap_or_default(),
            created_at: Utc::now(),
            last_evaluated: None,
            pass_count: 0,
            fail_count: 0,
            is_holdout: self.is_holdout.unwrap_or(true),
            tags: self.tags.unwrap_or_default(),
        }
    }
}

/// Result of evaluating a pattern against a scenario.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScenarioEvaluation {
    /// Unique identifier for this evaluation
    pub id: String,

    /// ID of the scenario being evaluated
    pub scenario_id: ScenarioId,

    /// ID of the pattern being evaluated
    pub pattern_id: String,

    /// Score from 0.0 to 1.0 indicating how well the pattern satisfies the scenario
    pub score: f32,

    /// Whether the pattern passed this scenario (score >= threshold)
    pub passed: bool,

    /// Optional feedback or notes about the evaluation
    pub feedback: Option<String>,

    /// When this evaluation was performed
    pub evaluated_at: DateTime<Utc>,

    /// Duration of the evaluation in milliseconds
    pub duration_ms: Option<u64>,
}

impl ScenarioEvaluation {
    /// Create a new evaluation.
    pub fn new(scenario_id: ScenarioId, pattern_id: impl Into<String>, score: f32) -> Self {
        Self {
            id: format!("eval_{}", Uuid::new_v4()),
            scenario_id,
            pattern_id: pattern_id.into(),
            score: score.clamp(0.0, 1.0),
            passed: score >= 0.7, // Default threshold
            feedback: None,
            evaluated_at: Utc::now(),
            duration_ms: None,
        }
    }

    /// Create an evaluation with a custom pass threshold.
    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.passed = self.score >= threshold;
        self
    }

    /// Set feedback.
    pub fn with_feedback(mut self, feedback: impl Into<String>) -> Self {
        self.feedback = Some(feedback.into());
        self
    }

    /// Set duration.
    pub fn with_duration_ms(mut self, duration_ms: u64) -> Self {
        self.duration_ms = Some(duration_ms);
        self
    }
}

/// Aggregate evaluation results for a pattern across all scenarios.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternScenarioStats {
    /// ID of the pattern
    pub pattern_id: String,

    /// Total number of scenarios evaluated
    pub scenarios_evaluated: u32,

    /// Number of scenarios passed
    pub scenarios_passed: u32,

    /// Average score across all scenarios
    pub avg_score: f32,

    /// Pass rate specifically for holdout scenarios (true generalization measure)
    pub holdout_pass_rate: f32,

    /// Number of holdout scenarios evaluated
    pub holdout_count: u32,

    /// Difficulty-weighted average score
    pub weighted_avg_score: f32,
}

impl PatternScenarioStats {
    /// Create empty stats for a pattern.
    pub fn new(pattern_id: impl Into<String>) -> Self {
        Self {
            pattern_id: pattern_id.into(),
            scenarios_evaluated: 0,
            scenarios_passed: 0,
            avg_score: 0.0,
            holdout_pass_rate: 0.0,
            holdout_count: 0,
            weighted_avg_score: 0.0,
        }
    }

    /// Get the overall pass rate.
    pub fn pass_rate(&self) -> f32 {
        if self.scenarios_evaluated == 0 {
            0.0
        } else {
            self.scenarios_passed as f32 / self.scenarios_evaluated as f32
        }
    }
}

/// Configuration for scenario evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScenarioEvaluationConfig {
    /// Minimum score to pass a scenario (0.0 - 1.0)
    pub pass_threshold: f32,

    /// Weight for keyword matching in scoring
    pub keyword_weight: f32,

    /// Weight for solution length/completeness in scoring
    pub completeness_weight: f32,

    /// Weight for domain match in scoring
    pub domain_weight: f32,

    /// Minimum solution length for full completeness score
    pub min_solution_length: usize,
}

impl Default for ScenarioEvaluationConfig {
    fn default() -> Self {
        Self {
            pass_threshold: 0.7,
            keyword_weight: 0.4,
            completeness_weight: 0.3,
            domain_weight: 0.3,
            min_solution_length: 100,
        }
    }
}

/// Evaluates patterns against scenarios.
pub struct ScenarioEvaluator {
    config: ScenarioEvaluationConfig,
}

impl ScenarioEvaluator {
    /// Create a new evaluator with default configuration.
    pub fn new() -> Self {
        Self {
            config: ScenarioEvaluationConfig::default(),
        }
    }

    /// Create a new evaluator with custom configuration.
    pub fn with_config(config: ScenarioEvaluationConfig) -> Self {
        Self { config }
    }

    /// Evaluate a pattern against a scenario.
    ///
    /// Returns a score from 0.0 to 1.0 based on:
    /// - Keyword overlap between solution and expected behavior
    /// - Solution completeness (length)
    /// - Domain match
    pub fn evaluate(&self, pattern: &Pattern, scenario: &Scenario) -> ScenarioEvaluation {
        let start = std::time::Instant::now();

        // Calculate individual scores
        let keyword_score = self.calculate_keyword_score(pattern, scenario);
        let completeness_score = self.calculate_completeness_score(pattern);
        let domain_score = self.calculate_domain_score(pattern, scenario);

        // Weighted average
        let score = (keyword_score * self.config.keyword_weight)
            + (completeness_score * self.config.completeness_weight)
            + (domain_score * self.config.domain_weight);

        let passed = score >= self.config.pass_threshold;
        let duration_ms = start.elapsed().as_millis() as u64;

        ScenarioEvaluation {
            id: format!("eval_{}", Uuid::new_v4()),
            scenario_id: scenario.id.clone(),
            pattern_id: pattern.id().to_string(),
            score,
            passed,
            feedback: Some(format!(
                "keyword={:.2}, completeness={:.2}, domain={:.2}",
                keyword_score, completeness_score, domain_score
            )),
            evaluated_at: Utc::now(),
            duration_ms: Some(duration_ms),
        }
    }

    /// Batch evaluate a pattern against multiple scenarios.
    pub fn evaluate_batch(
        &self,
        pattern: &Pattern,
        scenarios: &[Scenario],
    ) -> (Vec<ScenarioEvaluation>, PatternScenarioStats) {
        let mut evaluations = Vec::with_capacity(scenarios.len());
        let mut total_score = 0.0;
        let mut total_weighted_score = 0.0;
        let mut passed_count = 0u32;
        let mut holdout_passed = 0u32;
        let mut holdout_count = 0u32;

        for scenario in scenarios {
            let eval = self.evaluate(pattern, scenario);
            if eval.passed {
                passed_count += 1;
                if scenario.is_holdout {
                    holdout_passed += 1;
                }
            }
            if scenario.is_holdout {
                holdout_count += 1;
            }
            total_score += eval.score;
            total_weighted_score += eval.score * scenario.difficulty.weight();
            evaluations.push(eval);
        }

        let count = scenarios.len() as f32;
        let stats = PatternScenarioStats {
            pattern_id: pattern.id().to_string(),
            scenarios_evaluated: scenarios.len() as u32,
            scenarios_passed: passed_count,
            avg_score: if count > 0.0 { total_score / count } else { 0.0 },
            holdout_pass_rate: if holdout_count > 0 {
                holdout_passed as f32 / holdout_count as f32
            } else {
                0.0
            },
            holdout_count,
            weighted_avg_score: if count > 0.0 {
                total_weighted_score / count
            } else {
                0.0
            },
        };

        (evaluations, stats)
    }

    /// Calculate keyword overlap score between pattern solution and expected behavior.
    fn calculate_keyword_score(&self, pattern: &Pattern, scenario: &Scenario) -> f32 {
        let solution_lower = pattern.solution().to_lowercase();
        let expected_lower = scenario.expected_behavior.to_lowercase();
        let context_lower = scenario.input_context.to_lowercase();

        // Extract words from expected behavior (filtering short words)
        let expected_words: Vec<&str> = expected_lower
            .split(|c: char| !c.is_alphanumeric())
            .filter(|w| w.len() >= 3)
            .collect();

        if expected_words.is_empty() {
            return 0.5; // No keywords to match
        }

        // Count how many expected words appear in the solution
        let matches = expected_words
            .iter()
            .filter(|word| solution_lower.contains(*word))
            .count();

        // Also check if solution addresses the input context
        let context_words: Vec<&str> = context_lower
            .split(|c: char| !c.is_alphanumeric())
            .filter(|w| w.len() >= 3)
            .collect();

        let context_matches = context_words
            .iter()
            .filter(|word| solution_lower.contains(*word))
            .count();

        // Combine expected and context matching (weighted toward expected)
        let expected_ratio = matches as f32 / expected_words.len() as f32;
        let context_ratio = if context_words.is_empty() {
            0.5
        } else {
            context_matches as f32 / context_words.len() as f32
        };

        expected_ratio * 0.7 + context_ratio * 0.3
    }

    /// Calculate completeness score based on solution length.
    fn calculate_completeness_score(&self, pattern: &Pattern) -> f32 {
        let len = pattern.solution().len();
        if len >= self.config.min_solution_length {
            1.0
        } else if len == 0 {
            0.0
        } else {
            len as f32 / self.config.min_solution_length as f32
        }
    }

    /// Calculate domain match score.
    fn calculate_domain_score(&self, pattern: &Pattern, scenario: &Scenario) -> f32 {
        let pattern_domain = pattern.category().to_string().to_lowercase();
        let scenario_domain = scenario.domain.to_lowercase();

        if pattern_domain == scenario_domain {
            1.0
        } else if scenario_domain.starts_with(&pattern_domain)
            || pattern_domain.starts_with(&scenario_domain)
        {
            0.7 // Partial domain match (parent/child)
        } else {
            0.2 // Different domains
        }
    }
}

impl Default for ScenarioEvaluator {
    fn default() -> Self {
        Self::new()
    }
}

/// Storage operations for scenarios.
pub struct ScenarioStorage<'a> {
    adapter: &'a crate::db::DualWriteAdapter,
}

impl<'a> ScenarioStorage<'a> {
    /// Create a new scenario storage with a database adapter.
    pub fn new(adapter: &'a crate::db::DualWriteAdapter) -> Self {
        Self { adapter }
    }

    /// Initialize the scenarios schema.
    pub async fn init_schema(&self) -> Result<()> {
        let sql = r#"
            CREATE TABLE IF NOT EXISTS scenarios (
                id TEXT PRIMARY KEY,
                domain TEXT NOT NULL,
                description TEXT NOT NULL,
                input_context TEXT NOT NULL,
                expected_behavior TEXT,
                difficulty TEXT DEFAULT 'medium',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                last_evaluated TEXT,
                pass_count INTEGER DEFAULT 0,
                fail_count INTEGER DEFAULT 0,
                is_holdout INTEGER DEFAULT 1,
                tags TEXT DEFAULT '[]'
            );
            CREATE INDEX IF NOT EXISTS idx_scenarios_domain ON scenarios(domain);
            CREATE INDEX IF NOT EXISTS idx_scenarios_holdout ON scenarios(is_holdout);
            CREATE INDEX IF NOT EXISTS idx_scenarios_difficulty ON scenarios(difficulty);

            CREATE TABLE IF NOT EXISTS scenario_evaluations (
                id TEXT PRIMARY KEY,
                scenario_id TEXT NOT NULL,
                pattern_id TEXT NOT NULL,
                score REAL NOT NULL,
                passed INTEGER NOT NULL,
                feedback TEXT,
                evaluated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                duration_ms INTEGER,
                FOREIGN KEY (scenario_id) REFERENCES scenarios(id)
            );
            CREATE INDEX IF NOT EXISTS idx_evals_scenario ON scenario_evaluations(scenario_id);
            CREATE INDEX IF NOT EXISTS idx_evals_pattern ON scenario_evaluations(pattern_id);
            CREATE INDEX IF NOT EXISTS idx_evals_passed ON scenario_evaluations(passed);
        "#;

        self.adapter.sqlite().execute_batch(sql).await
    }

    /// Store a new scenario.
    pub async fn create_scenario(&self, scenario: &Scenario) -> Result<()> {
        let sql = r#"
            INSERT INTO scenarios (
                id, domain, description, input_context, expected_behavior,
                difficulty, created_at, last_evaluated, pass_count, fail_count,
                is_holdout, tags
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        "#;

        let tags_json = serde_json::to_string(&scenario.tags)?;
        let last_eval = scenario.last_evaluated.map(|dt| dt.to_rfc3339());

        self.adapter
            .sqlite()
            .execute(
                sql,
                &[
                    &scenario.id.as_str(),
                    &scenario.domain,
                    &scenario.description,
                    &scenario.input_context,
                    &scenario.expected_behavior,
                    &scenario.difficulty.as_str(),
                    &scenario.created_at.to_rfc3339(),
                    &last_eval,
                    &(scenario.pass_count as i64),
                    &(scenario.fail_count as i64),
                    &(scenario.is_holdout as i64),
                    &tags_json,
                ],
            )
            .await?;

        Ok(())
    }

    /// Get a scenario by ID.
    pub async fn get_scenario(&self, id: &ScenarioId) -> Result<Option<Scenario>> {
        let sql = r#"
            SELECT id, domain, description, input_context, expected_behavior,
                   difficulty, created_at, last_evaluated, pass_count, fail_count,
                   is_holdout, tags
            FROM scenarios
            WHERE id = ?
        "#;

        let results: Vec<Scenario> = self
            .adapter
            .sqlite()
            .query(sql, &[&id.as_str()], Self::row_to_scenario)
            .await?;

        Ok(results.into_iter().next())
    }

    /// Get all scenarios for a domain.
    pub async fn get_scenarios_for_domain(&self, domain: &str) -> Result<Vec<Scenario>> {
        let sql = r#"
            SELECT id, domain, description, input_context, expected_behavior,
                   difficulty, created_at, last_evaluated, pass_count, fail_count,
                   is_holdout, tags
            FROM scenarios
            WHERE domain = ? OR domain LIKE ?
            ORDER BY created_at DESC
        "#;

        let domain_prefix = format!("{}%", domain);

        self.adapter
            .sqlite()
            .query(sql, &[&domain, &domain_prefix], Self::row_to_scenario)
            .await
    }

    /// Get holdout scenarios for a domain.
    pub async fn get_holdout_scenarios(&self, domain: &str) -> Result<Vec<Scenario>> {
        let sql = r#"
            SELECT id, domain, description, input_context, expected_behavior,
                   difficulty, created_at, last_evaluated, pass_count, fail_count,
                   is_holdout, tags
            FROM scenarios
            WHERE is_holdout = 1 AND (domain = ? OR domain LIKE ?)
            ORDER BY difficulty DESC, created_at DESC
        "#;

        let domain_prefix = format!("{}%", domain);

        self.adapter
            .sqlite()
            .query(sql, &[&domain, &domain_prefix], Self::row_to_scenario)
            .await
    }

    /// List all scenarios with optional limit.
    pub async fn list_scenarios(&self, limit: usize) -> Result<Vec<Scenario>> {
        let sql = r#"
            SELECT id, domain, description, input_context, expected_behavior,
                   difficulty, created_at, last_evaluated, pass_count, fail_count,
                   is_holdout, tags
            FROM scenarios
            ORDER BY created_at DESC
            LIMIT ?
        "#;

        self.adapter
            .sqlite()
            .query(sql, &[&(limit as i64)], Self::row_to_scenario)
            .await
    }

    /// Record an evaluation result.
    pub async fn record_evaluation(&self, eval: &ScenarioEvaluation) -> Result<()> {
        let sql = r#"
            INSERT INTO scenario_evaluations (
                id, scenario_id, pattern_id, score, passed, feedback, evaluated_at, duration_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        "#;

        self.adapter
            .sqlite()
            .execute(
                sql,
                &[
                    &eval.id,
                    &eval.scenario_id.as_str(),
                    &eval.pattern_id,
                    &eval.score,
                    &(eval.passed as i64),
                    &eval.feedback,
                    &eval.evaluated_at.to_rfc3339(),
                    &eval.duration_ms.map(|d| d as i64),
                ],
            )
            .await?;

        // Update scenario pass/fail counts
        let update_sql = if eval.passed {
            "UPDATE scenarios SET pass_count = pass_count + 1, last_evaluated = ? WHERE id = ?"
        } else {
            "UPDATE scenarios SET fail_count = fail_count + 1, last_evaluated = ? WHERE id = ?"
        };

        self.adapter
            .sqlite()
            .execute(
                update_sql,
                &[&eval.evaluated_at.to_rfc3339(), &eval.scenario_id.as_str()],
            )
            .await?;

        Ok(())
    }

    /// Get statistics for a pattern's scenario evaluations.
    pub async fn get_pattern_stats(&self, pattern_id: &str) -> Result<PatternScenarioStats> {
        let sql = r#"
            SELECT
                COUNT(*) as total,
                COALESCE(SUM(CASE WHEN passed = 1 THEN 1 ELSE 0 END), 0) as passed,
                COALESCE(AVG(score), 0.0) as avg_score,
                COALESCE(SUM(CASE WHEN s.is_holdout = 1 AND e.passed = 1 THEN 1 ELSE 0 END), 0) as holdout_passed,
                COALESCE(SUM(CASE WHEN s.is_holdout = 1 THEN 1 ELSE 0 END), 0) as holdout_total
            FROM scenario_evaluations e
            JOIN scenarios s ON e.scenario_id = s.id
            WHERE e.pattern_id = ?
        "#;

        let results: Vec<(i64, i64, f64, i64, i64)> = self
            .adapter
            .sqlite()
            .query(sql, &[&pattern_id], |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, i64>(1)?,
                    row.get::<_, f64>(2)?,
                    row.get::<_, i64>(3)?,
                    row.get::<_, i64>(4)?,
                ))
            })
            .await?;

        if let Some((total, passed, avg_score, holdout_passed, holdout_total)) =
            results.into_iter().next()
        {
            Ok(PatternScenarioStats {
                pattern_id: pattern_id.to_string(),
                scenarios_evaluated: total as u32,
                scenarios_passed: passed as u32,
                avg_score: avg_score as f32,
                holdout_pass_rate: if holdout_total > 0 {
                    holdout_passed as f32 / holdout_total as f32
                } else {
                    0.0
                },
                holdout_count: holdout_total as u32,
                weighted_avg_score: avg_score as f32, // TODO: implement weighted
            })
        } else {
            Ok(PatternScenarioStats::new(pattern_id))
        }
    }

    /// Delete a scenario by ID.
    pub async fn delete_scenario(&self, id: &ScenarioId) -> Result<bool> {
        // Delete evaluations first
        self.adapter
            .sqlite()
            .execute(
                "DELETE FROM scenario_evaluations WHERE scenario_id = ?",
                &[&id.as_str()],
            )
            .await?;

        // Delete scenario
        let deleted = self
            .adapter
            .sqlite()
            .execute("DELETE FROM scenarios WHERE id = ?", &[&id.as_str()])
            .await?;

        Ok(deleted > 0)
    }

    /// Get overall scenario statistics.
    pub async fn get_stats(&self) -> Result<ScenarioStats> {
        let sql = r#"
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN is_holdout = 1 THEN 1 ELSE 0 END) as holdout_count,
                SUM(pass_count) as total_passes,
                SUM(fail_count) as total_fails,
                COUNT(DISTINCT domain) as domain_count
            FROM scenarios
        "#;

        let results: Vec<(i64, i64, i64, i64, i64)> = self
            .adapter
            .sqlite()
            .query(sql, &[], |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, i64>(1)?,
                    row.get::<_, i64>(2)?,
                    row.get::<_, i64>(3)?,
                    row.get::<_, i64>(4)?,
                ))
            })
            .await?;

        if let Some((total, holdout, passes, fails, domains)) = results.into_iter().next() {
            let total_evals = passes + fails;
            Ok(ScenarioStats {
                total_scenarios: total as u32,
                holdout_scenarios: holdout as u32,
                total_evaluations: total_evals as u32,
                pass_rate: if total_evals > 0 {
                    passes as f32 / total_evals as f32
                } else {
                    0.0
                },
                domain_count: domains as u32,
            })
        } else {
            Ok(ScenarioStats::default())
        }
    }

    /// Convert a database row to a Scenario.
    fn row_to_scenario(row: &rusqlite::Row<'_>) -> rusqlite::Result<Scenario> {
        let id: String = row.get(0)?;
        let domain: String = row.get(1)?;
        let description: String = row.get(2)?;
        let input_context: String = row.get(3)?;
        let expected_behavior: Option<String> = row.get(4)?;
        let difficulty: String = row.get(5)?;
        let created_at: String = row.get(6)?;
        let last_evaluated: Option<String> = row.get(7)?;
        let pass_count: i64 = row.get(8)?;
        let fail_count: i64 = row.get(9)?;
        let is_holdout: i64 = row.get(10)?;
        let tags_json: String = row.get(11)?;

        let tags: Vec<String> = serde_json::from_str(&tags_json).unwrap_or_default();

        Ok(Scenario {
            id: ScenarioId(id),
            domain,
            description,
            input_context,
            expected_behavior: expected_behavior.unwrap_or_default(),
            difficulty: Difficulty::from_str(&difficulty),
            created_at: DateTime::parse_from_rfc3339(&created_at)
                .map(|dt| dt.with_timezone(&Utc))
                .unwrap_or_else(|_| Utc::now()),
            last_evaluated: last_evaluated.and_then(|s| {
                DateTime::parse_from_rfc3339(&s)
                    .map(|dt| dt.with_timezone(&Utc))
                    .ok()
            }),
            pass_count: pass_count as u32,
            fail_count: fail_count as u32,
            is_holdout: is_holdout != 0,
            tags,
        })
    }
}

/// Overall scenario statistics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ScenarioStats {
    /// Total number of scenarios
    pub total_scenarios: u32,

    /// Number of holdout scenarios
    pub holdout_scenarios: u32,

    /// Total number of evaluations performed
    pub total_evaluations: u32,

    /// Overall pass rate across all evaluations
    pub pass_rate: f32,

    /// Number of unique domains
    pub domain_count: u32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reasoning_bank::pattern::PatternCategory;

    #[test]
    fn test_scenario_id() {
        let id1 = ScenarioId::new();
        let id2 = ScenarioId::new();
        assert_ne!(id1, id2);
        assert!(id1.as_str().starts_with("scen_"));
    }

    #[test]
    fn test_difficulty_from_str() {
        assert_eq!(Difficulty::from_str("easy"), Difficulty::Easy);
        assert_eq!(Difficulty::from_str("MEDIUM"), Difficulty::Medium);
        assert_eq!(Difficulty::from_str("hard"), Difficulty::Hard);
        assert_eq!(Difficulty::from_str("unknown"), Difficulty::Medium);
    }

    #[test]
    fn test_difficulty_weight() {
        assert!((Difficulty::Easy.weight() - 1.0).abs() < 0.001);
        assert!((Difficulty::Medium.weight() - 1.5).abs() < 0.001);
        assert!((Difficulty::Hard.weight() - 2.0).abs() < 0.001);
    }

    #[test]
    fn test_scenario_builder() {
        let scenario = Scenario::new("rust.async")
            .with_description("Test timeout handling")
            .with_input_context("Long-running async operation")
            .with_expected_behavior("Should return error, not panic")
            .with_difficulty(Difficulty::Hard)
            .as_holdout(true)
            .with_tag("async")
            .with_tag("error-handling")
            .build();

        assert_eq!(scenario.domain, "rust.async");
        assert_eq!(scenario.description, "Test timeout handling");
        assert_eq!(scenario.difficulty, Difficulty::Hard);
        assert!(scenario.is_holdout);
        assert_eq!(scenario.tags.len(), 2);
    }

    #[test]
    fn test_scenario_pass_rate() {
        let mut scenario = Scenario::new("test")
            .with_description("Test scenario")
            .build();

        // No evaluations
        assert!((scenario.pass_rate() - 0.5).abs() < 0.001);

        // Some evaluations
        scenario.pass_count = 3;
        scenario.fail_count = 1;
        assert!((scenario.pass_rate() - 0.75).abs() < 0.001);
    }

    #[test]
    fn test_scenario_evaluation() {
        let eval = ScenarioEvaluation::new(
            ScenarioId::from_string("scen_test"),
            "pat_123",
            0.85,
        );

        assert!(eval.passed);
        assert!((eval.score - 0.85).abs() < 0.001);

        // Test custom threshold
        let eval_low = ScenarioEvaluation::new(
            ScenarioId::from_string("scen_test"),
            "pat_123",
            0.65,
        );
        assert!(!eval_low.passed);

        let eval_low_custom = eval_low.with_threshold(0.5);
        assert!(eval_low_custom.passed);
    }

    #[test]
    fn test_evaluator_keyword_scoring() {
        use crate::reasoning_bank::pattern::Pattern;

        let evaluator = ScenarioEvaluator::new();

        let scenario = Scenario::new("testing")
            .with_description("Error handling test")
            .with_input_context("Application crashes on invalid input")
            .with_expected_behavior("Should validate input and return descriptive error message")
            .build();

        // Pattern with good keyword match
        let good_pattern = Pattern::builder()
            .problem("How to handle invalid input")
            .solution("Validate input at the boundary, return descriptive error message to user")
            .category(PatternCategory::Testing)
            .build();

        let eval = evaluator.evaluate(&good_pattern, &scenario);
        assert!(eval.score > 0.5);

        // Pattern with poor keyword match
        let bad_pattern = Pattern::builder()
            .problem("How to optimize performance")
            .solution("Use caching and parallel processing")
            .category(PatternCategory::Performance)
            .build();

        let bad_eval = evaluator.evaluate(&bad_pattern, &scenario);
        assert!(bad_eval.score < eval.score);
    }

    #[test]
    fn test_evaluator_batch() {
        use crate::reasoning_bank::pattern::Pattern;

        let evaluator = ScenarioEvaluator::new();

        let scenarios = vec![
            Scenario::new("testing")
                .with_description("Test scenario 1")
                .with_expected_behavior("Handle errors gracefully")
                .as_holdout(true)
                .build(),
            Scenario::new("testing")
                .with_description("Test scenario 2")
                .with_expected_behavior("Log all operations")
                .as_holdout(false)
                .build(),
        ];

        let pattern = Pattern::builder()
            .problem("How to handle errors")
            .solution("Handle errors gracefully with logging and proper error messages")
            .category(PatternCategory::Testing)
            .build();

        let (evals, stats) = evaluator.evaluate_batch(&pattern, &scenarios);

        assert_eq!(evals.len(), 2);
        assert_eq!(stats.scenarios_evaluated, 2);
        assert_eq!(stats.holdout_count, 1);
    }

    #[test]
    fn test_pattern_scenario_stats() {
        let mut stats = PatternScenarioStats::new("pat_123");
        assert_eq!(stats.pass_rate(), 0.0);

        stats.scenarios_evaluated = 10;
        stats.scenarios_passed = 7;
        assert!((stats.pass_rate() - 0.7).abs() < 0.001);
    }
}
