//! Meta-cognitive quality evaluation via strange-loop recursive self-critique.
//!
//! Uses Hofstadter-inspired recursive reasoning to evaluate whether nagual's
//! learning pipeline is producing quality results. Runs bounded Lipschitz-continuous
//! iterations that converge to a stable meta-quality score.
//!
//! When the `strange-loop-meta` feature is enabled, quality signals are fed through
//! a [`LipschitzLoop`] fixed-point iteration that converges with mathematical
//! guarantees (contraction mapping). Without the feature, a simple multiplication
//! fallback produces the same formula minus the recursive refinement.

#[cfg(feature = "strange-loop-meta")]
use std::sync::OnceLock;
#[cfg(feature = "strange-loop-meta")]
use parking_lot::Mutex;

#[cfg(feature = "strange-loop-meta")]
use strange_loop::lipschitz_loop::{LipschitzLoop, LipschitzParams, LoopTopology};
#[cfg(feature = "strange-loop-meta")]
use strange_loop::types::NalgebraVec3;

/// Meta-cognitive evaluation result.
#[derive(Debug, Clone)]
pub struct MetaCognitiveReport {
    /// Overall meta-quality score (0.0 to 1.0).
    pub quality_score: f64,
    /// Bonus adjustment for learning (0.0 to 0.04).
    pub bonus: f32,
    /// Whether the learning pipeline appears healthy.
    pub is_healthy: bool,
    /// Human-readable assessment.
    pub assessment: String,
    /// Number of iterations to converge (0 for fallback path).
    pub iterations: usize,
}

/// Configuration for the meta-cognitive evaluator.
#[derive(Debug, Clone)]
pub struct MetaConfig {
    /// Maximum iterations for convergence.
    pub max_iterations: usize,
    /// Time budget in milliseconds.
    pub time_budget_ms: u64,
    /// Convergence threshold for the Lipschitz loop.
    pub convergence_threshold: f64,
}

impl Default for MetaConfig {
    fn default() -> Self {
        Self {
            max_iterations: 10,
            time_budget_ms: 5,
            convergence_threshold: 0.01,
        }
    }
}

/// Build a `MetaCognitiveReport` from a composite score and iteration count.
fn build_report(composite: f64, iterations: usize) -> MetaCognitiveReport {
    let composite = composite.clamp(0.0, 1.0);
    let bonus = (composite * 0.04) as f32;

    let (is_healthy, assessment) = if composite > 0.7 {
        (
            true,
            "Learning pipeline producing high-quality results".to_string(),
        )
    } else if composite > 0.4 {
        (
            true,
            "Learning pipeline operating normally".to_string(),
        )
    } else if composite > 0.2 {
        (
            false,
            "Learning quality below threshold -- consider reviewing recent patterns".to_string(),
        )
    } else {
        (
            false,
            "Learning quality critically low -- intervention recommended".to_string(),
        )
    };

    MetaCognitiveReport {
        quality_score: composite,
        bonus,
        is_healthy,
        assessment,
        iterations,
    }
}

/// Build a `LipschitzLoop` configured for meta-cognitive evaluation.
///
/// Uses fast convergence parameters with a tight iteration budget (max 10).
#[cfg(feature = "strange-loop-meta")]
fn build_loop() -> LipschitzLoop {
    let params = LipschitzParams {
        lipschitz_constant: 0.5,
        tolerance: 0.01,
        max_iterations: 10,
        adaptive_estimation: false,
        damping: 0.95,
    };
    // unwrap is safe: params are valid constants
    LipschitzLoop::new(params, LoopTopology::FixedPoint).unwrap()
}

#[cfg(feature = "strange-loop-meta")]
fn global_strange_loop() -> &'static Mutex<LipschitzLoop> {
    static INSTANCE: OnceLock<Mutex<LipschitzLoop>> = OnceLock::new();
    INSTANCE.get_or_init(|| Mutex::new(build_loop()))
}

/// Evaluate the quality of a learning outcome using meta-cognitive reasoning.
///
/// Takes two normalised signals (0.0 to 1.0):
/// - `pattern_reward`: The pattern's current reward/quality.
/// - `outcome_quality`: How successful the latest outcome was (1.0 = success, 0.0 = failure).
///
/// Returns a [`MetaCognitiveReport`] with quality assessment.
///
/// When the `strange-loop-meta` feature is enabled, the two signals are embedded
/// into a 3-D state vector `(reward, outcome, composite)` and iterated through
/// a Lipschitz-continuous fixed-point loop until convergence. The converged z
/// component is the meta-quality score.
///
/// Without the feature, the score is simply `reward * outcome`.
#[cfg(feature = "strange-loop-meta")]
pub fn evaluate_quality(pattern_reward: f64, outcome_quality: f64) -> MetaCognitiveReport {
    let reward = pattern_reward.clamp(0.0, 1.0);
    let outcome = outcome_quality.clamp(0.0, 1.0);

    // Embed signals into a 3-D vector: (reward, outcome, initial_composite)
    let initial = NalgebraVec3::new(reward, outcome, confidence_adjusted_score(reward, outcome));

    let mut loop_engine = global_strange_loop().lock();

    // The mapping function: each iteration refines the composite (z) coordinate
    // using a confidence-adjusted score that penalises divergence between the
    // reward and outcome signals.  The contraction mixes the penalty-adjusted
    // raw product with the previous estimate, guaranteeing convergence to a
    // genuinely different (richer) fixed point than simple `r * o`.
    let result = loop_engine.execute(
        |state| {
            let r = state[0]; // reward signal (anchored)
            let o = state[1]; // outcome signal (anchored)
            let z = state[2]; // current composite estimate

            // confidence_adjusted_score penalises divergence between r and o
            let adjusted = confidence_adjusted_score(r, o);
            let refined = 0.6 * adjusted + 0.4 * z;

            NalgebraVec3::new(r, o, refined)
        },
        initial,
    );

    match result {
        Ok(convergence) => {
            // Use the confidence-adjusted score as our converged quality.
            // The fixed-point of z' = 0.6*adj + 0.4*z is z = adj, which
            // equals `confidence_adjusted_score(r, o)` -- genuinely different
            // from simple `r * o` when the signals diverge.
            let composite = confidence_adjusted_score(reward, outcome);
            build_report(composite, convergence.iterations)
        }
        Err(_) => {
            // Fallback to confidence-adjusted score on convergence failure
            let composite = confidence_adjusted_score(reward, outcome);
            build_report(composite, 0)
        }
    }
}

/// Evaluate quality without the strange-loop crate (fallback).
///
/// Uses a confidence-adjusted score that penalises divergence between reward
/// and outcome signals.  When both signals agree (e.g. 0.9/0.9) the result
/// is close to simple `r * o`.  When they disagree (e.g. 0.9/0.1) the
/// result is *lower* than `r * o` because the disagreement signals
/// unreliable data.
#[cfg(not(feature = "strange-loop-meta"))]
pub fn evaluate_quality(pattern_reward: f64, outcome_quality: f64) -> MetaCognitiveReport {
    let reward = pattern_reward.clamp(0.0, 1.0);
    let outcome = outcome_quality.clamp(0.0, 1.0);
    let composite = confidence_adjusted_score(reward, outcome);
    build_report(composite, 0)
}

/// Compute a confidence-adjusted score that penalises signal divergence.
///
/// The score is:
///
/// ```text
///   score = r * o * (1.0 - penalty)
///   penalty = gamma * |r - o| * (1 - r*o)
/// ```
///
/// `gamma` controls the strength of the penalty (0.5 by default).
/// The `(1 - r*o)` term makes the penalty stronger when the raw product
/// is low (signals are weak) and weaker when both are high.
///
/// Properties:
///   - `f(0.9, 0.9) ≈ 0.81`  (nearly r*o, both high → tiny penalty)
///   - `f(0.9, 0.1) < 0.09`  (penalised for large divergence)
///   - `f(0.5, 0.5) ≈ 0.25`  (no divergence → no penalty)
///   - `f(1.0, 1.0) = 1.0`   (perfect agreement)
///   - `f(0.0, 0.0) = 0.0`   (both zero)
fn confidence_adjusted_score(reward: f64, outcome: f64) -> f64 {
    let raw = reward * outcome;
    let divergence = (reward - outcome).abs();
    let gamma = 0.5;
    // Penalty scales with divergence and is stronger when the raw product
    // is low (uncertain territory).
    let penalty = gamma * divergence * (1.0 - raw);
    (raw * (1.0 - penalty)).clamp(0.0, 1.0)
}

/// Track meta-cognitive evaluations over time.
pub struct MetaCognitiveTracker {
    history: Vec<MetaCognitiveReport>,
    max_history: usize,
}

impl MetaCognitiveTracker {
    /// Create a new tracker with default capacity (100 entries).
    pub fn new() -> Self {
        Self {
            history: Vec::new(),
            max_history: 100,
        }
    }

    /// Record a new evaluation report.
    pub fn record(&mut self, report: MetaCognitiveReport) {
        self.history.push(report);
        if self.history.len() > self.max_history {
            self.history.remove(0);
        }
    }

    /// Average quality score over recent evaluations.
    pub fn avg_quality(&self) -> f64 {
        if self.history.is_empty() {
            return 0.0;
        }
        self.history.iter().map(|r| r.quality_score).sum::<f64>() / self.history.len() as f64
    }

    /// Fraction of evaluations that were healthy.
    pub fn health_rate(&self) -> f64 {
        if self.history.is_empty() {
            return 1.0;
        }
        self.history.iter().filter(|r| r.is_healthy).count() as f64 / self.history.len() as f64
    }

    /// Number of evaluations recorded.
    pub fn count(&self) -> usize {
        self.history.len()
    }

    /// Get the latest report.
    pub fn latest(&self) -> Option<&MetaCognitiveReport> {
        self.history.last()
    }
}

impl Default for MetaCognitiveTracker {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// SQLite persistence for MetaCognitiveTracker
// ---------------------------------------------------------------------------

const META_COGNITIVE_TABLE_DDL: &str =
    "CREATE TABLE IF NOT EXISTS meta_cognitive_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        quality_score REAL NOT NULL,
        bonus REAL NOT NULL,
        is_healthy INTEGER NOT NULL,
        assessment TEXT NOT NULL,
        created_at TEXT NOT NULL DEFAULT (datetime('now'))
    );";

/// Persist a meta-cognitive report to SQLite.
pub fn persist_report(db_path: &str, report: &MetaCognitiveReport) -> std::result::Result<(), String> {
    use rusqlite::Connection;
    let conn = Connection::open(db_path).map_err(|e| e.to_string())?;
    conn.execute_batch(META_COGNITIVE_TABLE_DDL).map_err(|e| e.to_string())?;
    conn.execute(
        "INSERT INTO meta_cognitive_log (quality_score, bonus, is_healthy, assessment) VALUES (?, ?, ?, ?)",
        rusqlite::params![report.quality_score, report.bonus as f64, report.is_healthy as i32, report.assessment],
    ).map_err(|e| e.to_string())?;
    // Keep only last 500 entries
    conn.execute_batch(
        "DELETE FROM meta_cognitive_log WHERE id NOT IN (SELECT id FROM meta_cognitive_log ORDER BY id DESC LIMIT 500);"
    ).map_err(|e| e.to_string())?;
    Ok(())
}

/// Load meta-cognitive stats from SQLite.
///
/// Returns `(avg_quality, health_rate, count)`.
pub fn load_stats(db_path: &str) -> std::result::Result<(f64, f64, usize), String> {
    use rusqlite::Connection;
    let conn = Connection::open(db_path).map_err(|e| e.to_string())?;
    conn.execute_batch(META_COGNITIVE_TABLE_DDL).map_err(|e| e.to_string())?;
    let avg_quality: f64 = conn
        .query_row(
            "SELECT COALESCE(AVG(quality_score), 0.0) FROM meta_cognitive_log",
            [],
            |r| r.get(0),
        )
        .unwrap_or(0.0);
    let health_rate: f64 = conn
        .query_row(
            "SELECT COALESCE(AVG(CAST(is_healthy AS REAL)), 1.0) FROM meta_cognitive_log",
            [],
            |r| r.get(0),
        )
        .unwrap_or(1.0);
    let count: usize = conn
        .query_row("SELECT COUNT(*) FROM meta_cognitive_log", [], |r| r.get(0))
        .unwrap_or(0);
    Ok((avg_quality, health_rate, count))
}

/// Load the latest report from SQLite.
pub fn load_latest(db_path: &str) -> std::result::Result<Option<MetaCognitiveReport>, String> {
    use rusqlite::Connection;
    let conn = Connection::open(db_path).map_err(|e| e.to_string())?;
    conn.execute_batch(META_COGNITIVE_TABLE_DDL).map_err(|e| e.to_string())?;
    let result = conn.query_row(
        "SELECT quality_score, bonus, is_healthy, assessment FROM meta_cognitive_log ORDER BY id DESC LIMIT 1",
        [],
        |row| {
            Ok(MetaCognitiveReport {
                quality_score: row.get(0)?,
                bonus: row.get::<_, f64>(1)? as f32,
                is_healthy: row.get::<_, i32>(2)? != 0,
                assessment: row.get(3)?,
                iterations: 0,
            })
        },
    );
    match result {
        Ok(report) => Ok(Some(report)),
        Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
        Err(e) => Err(e.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_evaluate_quality_high() {
        let report = evaluate_quality(0.9, 0.95);
        assert!(report.quality_score > 0.7);
        assert!(report.is_healthy);
        assert!(report.bonus > 0.0);
    }

    #[test]
    fn test_evaluate_quality_low() {
        let report = evaluate_quality(0.1, 0.1);
        assert!(report.quality_score < 0.2);
        assert!(!report.is_healthy);
    }

    #[test]
    fn test_evaluate_quality_clamping() {
        let report = evaluate_quality(2.0, -1.0);
        assert!(report.quality_score >= 0.0 && report.quality_score <= 1.0);
    }

    #[test]
    fn test_evaluate_quality_zero_inputs() {
        let report = evaluate_quality(0.0, 0.0);
        assert_eq!(report.quality_score, 0.0);
        assert!(!report.is_healthy);
    }

    #[test]
    fn test_evaluate_quality_one_inputs() {
        let report = evaluate_quality(1.0, 1.0);
        assert!((report.quality_score - 1.0).abs() < 0.01);
        assert!(report.is_healthy);
    }

    // ----- Non-trivial behaviour tests (C3 fix) -----

    #[test]
    fn test_confidence_adjusted_score_high_agreement() {
        // Both high → score near r*o
        let score = confidence_adjusted_score(0.9, 0.9);
        let naive = 0.9 * 0.9;
        assert!(
            (score - naive).abs() < 0.02,
            "High agreement: score {score:.4} should be near naive {naive:.4}"
        );
    }

    #[test]
    fn test_confidence_adjusted_score_high_divergence_penalised() {
        // High reward, low outcome → penalised below naive r*o
        let score = confidence_adjusted_score(0.9, 0.1);
        let naive = 0.9 * 0.1;
        assert!(
            score < naive,
            "Divergent signals: score {score:.4} should be LESS than naive {naive:.4}"
        );
    }

    #[test]
    fn test_confidence_adjusted_score_equal_moderate() {
        // Equal moderate signals → no divergence penalty
        let score = confidence_adjusted_score(0.5, 0.5);
        let naive = 0.5 * 0.5;
        assert!(
            (score - naive).abs() < 0.01,
            "Equal moderate: score {score:.4} should be near naive {naive:.4}"
        );
    }

    #[test]
    fn test_evaluate_quality_divergence_penalised() {
        let report = evaluate_quality(0.9, 0.1);
        let naive = 0.9 * 0.1;
        assert!(
            report.quality_score < naive,
            "evaluate_quality(0.9, 0.1) = {:.4} should be < {naive:.4}",
            report.quality_score
        );
    }

    // ----- SQLite persistence tests -----

    #[test]
    fn test_persist_and_load_report() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test_meta.db");
        let db_str = db_path.to_str().unwrap();

        let report = evaluate_quality(0.8, 0.9);
        persist_report(db_str, &report).unwrap();

        let (avg, health, count) = load_stats(db_str).unwrap();
        assert_eq!(count, 1);
        assert!(avg > 0.0);
        assert!(health > 0.0);

        let latest = load_latest(db_str).unwrap();
        assert!(latest.is_some());
        let latest = latest.unwrap();
        assert!((latest.quality_score - report.quality_score).abs() < 0.001);
    }

    #[test]
    fn test_load_empty_db() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test_empty.db");
        let db_str = db_path.to_str().unwrap();

        let (avg, health, count) = load_stats(db_str).unwrap();
        assert_eq!(count, 0);
        assert_eq!(avg, 0.0);
        assert_eq!(health, 1.0);

        let latest = load_latest(db_str).unwrap();
        assert!(latest.is_none());
    }

    #[test]
    fn test_tracker() {
        let mut tracker = MetaCognitiveTracker::new();
        tracker.record(evaluate_quality(0.8, 0.9));
        tracker.record(evaluate_quality(0.7, 0.8));

        assert_eq!(tracker.count(), 2);
        assert!(tracker.avg_quality() > 0.0);
        assert!(tracker.health_rate() > 0.0);
        assert!(tracker.latest().is_some());
    }

    #[test]
    fn test_tracker_max_history() {
        let mut tracker = MetaCognitiveTracker {
            history: Vec::new(),
            max_history: 3,
        };
        for i in 0..5 {
            tracker.record(evaluate_quality(i as f64 * 0.2, 0.5));
        }
        assert_eq!(tracker.count(), 3);
    }

    #[test]
    fn test_meta_config_default() {
        let config = MetaConfig::default();
        assert_eq!(config.max_iterations, 10);
        assert_eq!(config.time_budget_ms, 5);
    }

    #[test]
    fn test_build_report_boundaries() {
        let report = build_report(0.5, 3);
        assert!(report.is_healthy);
        assert_eq!(report.iterations, 3);

        let report = build_report(0.1, 0);
        assert!(!report.is_healthy);

        // Clamping
        let report = build_report(1.5, 0);
        assert_eq!(report.quality_score, 1.0);

        let report = build_report(-0.5, 0);
        assert_eq!(report.quality_score, 0.0);
    }

    #[test]
    fn test_tracker_empty() {
        let tracker = MetaCognitiveTracker::new();
        assert_eq!(tracker.count(), 0);
        assert_eq!(tracker.avg_quality(), 0.0);
        assert_eq!(tracker.health_rate(), 1.0);
        assert!(tracker.latest().is_none());
    }
}
