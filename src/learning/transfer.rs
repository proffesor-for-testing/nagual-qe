//! Cross-Domain Transfer with Thompson Sampling (KOS P4).
//!
//! Enables knowledge transfer between domains using Beta-distributed priors
//! and Thompson Sampling to balance exploration vs exploitation when deciding
//! which patterns to transfer across domain boundaries.
//!
//! # Algorithm
//!
//! For each (source_domain, target_domain) pair, maintain a Beta(alpha, beta) prior:
//! - On successful transfer: alpha += 1
//! - On failed transfer: beta += 1
//! - Thompson sample from Beta to rank candidates
//!
//! Transferred patterns get a damped reward (source_reward * damping_factor) and
//! are linked via lineage with DerivationType::Transfer.

use std::collections::HashMap;
use std::sync::Arc;

use chrono::Utc;
use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::db::SqliteDb;
use crate::error::{DatabaseError, Result};
use crate::reasoning_bank::pattern::{PatternBuilder, PatternCategory, PatternId};

// ---------------------------------------------------------------------------
// Beta distribution for Thompson Sampling
// ---------------------------------------------------------------------------

/// Beta distribution parameters for Thompson Sampling.
///
/// Maintains conjugate prior counts: alpha (successes + 1) and beta (failures + 1).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BetaParams {
    /// Success count + 1 (prior pseudo-count).
    pub alpha: f64,
    /// Failure count + 1 (prior pseudo-count).
    pub beta: f64,
}

impl BetaParams {
    /// Create with explicit alpha/beta.
    pub fn new(alpha: f64, beta: f64) -> Self {
        Self {
            alpha: alpha.max(0.01),
            beta: beta.max(0.01),
        }
    }

    /// Uniform (uninformative) prior: Beta(1, 1).
    pub fn uniform() -> Self {
        Self {
            alpha: 1.0,
            beta: 1.0,
        }
    }

    /// Mean of the Beta distribution: alpha / (alpha + beta).
    pub fn mean(&self) -> f64 {
        self.alpha / (self.alpha + self.beta)
    }

    /// Variance of the Beta distribution.
    pub fn variance(&self) -> f64 {
        let ab = self.alpha + self.beta;
        (self.alpha * self.beta) / (ab * ab * (ab + 1.0))
    }

    /// Draw a Thompson sample from Beta(alpha, beta).
    ///
    /// Uses the Gamma-ratio trick: if X ~ Gamma(a,1) and Y ~ Gamma(b,1),
    /// then X/(X+Y) ~ Beta(a,b).
    pub fn sample(&self, rng: &mut impl Rng) -> f64 {
        let x = gamma_sample(self.alpha, rng);
        let y = gamma_sample(self.beta, rng);
        if x + y == 0.0 {
            0.5
        } else {
            x / (x + y)
        }
    }

    /// Record a successful outcome.
    pub fn update_success(&mut self) {
        self.alpha += 1.0;
    }

    /// Record a failed outcome.
    pub fn update_failure(&mut self) {
        self.beta += 1.0;
    }

    /// Confidence level based on total observations.
    ///
    /// Returns a value in [0, 1) where higher means more observations.
    pub fn confidence(&self) -> f64 {
        let n = (self.alpha + self.beta - 2.0).max(0.0);
        1.0 - 1.0 / (1.0 + n)
    }
}

// ---------------------------------------------------------------------------
// Gamma sampling helpers (for Beta via Gamma trick)
// ---------------------------------------------------------------------------

/// Sample from Gamma(shape, 1) using Marsaglia & Tsang's method.
fn gamma_sample(shape: f64, rng: &mut impl Rng) -> f64 {
    if shape <= 0.0 {
        return 0.0;
    }
    // For shape < 1 use Ahrens-Dieter reduction
    if shape < 1.0 {
        let u: f64 = rng.gen();
        return gamma_sample(shape + 1.0, rng) * u.powf(1.0 / shape);
    }
    let d = shape - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();
    loop {
        let x: f64 = loop {
            let n = standard_normal(rng);
            if 1.0 + c * n > 0.0 {
                break n;
            }
        };
        let v = (1.0 + c * x).powi(3);
        let u: f64 = rng.gen();
        if u < 1.0 - 0.0331 * x.powi(4) {
            return d * v;
        }
        if u.ln() < 0.5 * x * x + d * (1.0 - v + v.ln()) {
            return d * v;
        }
    }
}

/// Box-Muller transform for standard normal samples.
fn standard_normal(rng: &mut impl Rng) -> f64 {
    let u1: f64 = rng.gen::<f64>().max(f64::MIN_POSITIVE); // avoid ln(0)
    let u2: f64 = rng.gen();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

// ---------------------------------------------------------------------------
// Transfer candidate
// ---------------------------------------------------------------------------

/// A pattern identified as a candidate for cross-domain transfer.
#[derive(Debug, Clone)]
pub struct TransferCandidate {
    /// Source pattern's ID.
    pub source_pattern_id: String,
    /// Domain the pattern comes from.
    pub source_domain: String,
    /// Domain the pattern would be transferred to.
    pub target_domain: String,
    /// Relevance score (how applicable the source pattern is to the target domain).
    pub relevance_score: f64,
    /// Beta prior for this domain pair's transfer history.
    pub transfer_prior: BetaParams,
    /// Thompson-sampled expected reward.
    pub expected_reward: f64,
    /// Original pattern's reward in the source domain.
    pub source_reward: f64,
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the cross-domain transfer engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransferConfig {
    /// Minimum reward in source domain to be eligible for transfer (default: 0.7).
    pub min_source_reward: f64,
    /// Minimum relevance score for a transfer to be considered (default: 0.5).
    pub similarity_threshold: f64,
    /// Damping factor applied to transferred rewards (default: 0.3).
    pub damping_factor: f64,
    /// Maximum number of transfers per cycle (default: 10).
    pub max_transfers_per_cycle: usize,
}

impl Default for TransferConfig {
    fn default() -> Self {
        Self {
            min_source_reward: 0.7,
            similarity_threshold: 0.5,
            damping_factor: 0.3,
            max_transfers_per_cycle: 10,
        }
    }
}

// ---------------------------------------------------------------------------
// Transfer statistics
// ---------------------------------------------------------------------------

/// Aggregate statistics about cross-domain transfers.
pub struct TransferStats {
    /// Total number of transfers executed.
    pub total_transfers: usize,
    /// Number of successful transfers.
    pub successful_transfers: usize,
    /// Number of unique (source, target) domain pairs.
    pub domain_pairs: usize,
    /// Best performing domain pair: (source, target, success_rate).
    pub best_pair: Option<(String, String, f64)>,
}

// ---------------------------------------------------------------------------
// Domain Transfer Engine
// ---------------------------------------------------------------------------

/// Engine for cross-domain pattern transfer with Thompson Sampling.
///
/// Maintains Beta priors per (source, target) domain pair and uses
/// Thompson Sampling to rank transfer candidates.
pub struct DomainTransferEngine {
    db: Arc<SqliteDb>,
    transfer_history: parking_lot::Mutex<HashMap<(String, String), BetaParams>>,
    config: TransferConfig,
    lineage: Option<Arc<crate::lineage::LineageQuery>>,
}

impl DomainTransferEngine {
    /// Create a new transfer engine.
    pub fn new(db: Arc<SqliteDb>, config: TransferConfig) -> Self {
        Self {
            db,
            transfer_history: parking_lot::Mutex::new(HashMap::new()),
            config,
            lineage: None,
        }
    }

    /// Attach a lineage query for recording cross-domain transfer lineage.
    pub fn with_lineage(mut self, lineage: Arc<crate::lineage::LineageQuery>) -> Self {
        self.lineage = Some(lineage);
        self
    }

    /// Initialize the database schema for transfer history persistence.
    pub async fn init_schema(&self) -> Result<()> {
        self.db
            .execute_batch(
                "CREATE TABLE IF NOT EXISTS transfer_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_domain TEXT NOT NULL,
                    target_domain TEXT NOT NULL,
                    source_pattern_id TEXT NOT NULL,
                    target_pattern_id TEXT,
                    alpha REAL NOT NULL DEFAULT 1.0,
                    beta REAL NOT NULL DEFAULT 1.0,
                    success INTEGER,
                    timestamp TEXT NOT NULL,
                    damped_reward REAL
                );
                CREATE INDEX IF NOT EXISTS idx_transfer_domains
                    ON transfer_history(source_domain, target_domain);

                CREATE TABLE IF NOT EXISTS transfer_priors (
                    source_domain TEXT NOT NULL,
                    target_domain TEXT NOT NULL,
                    alpha REAL NOT NULL DEFAULT 1.0,
                    beta REAL NOT NULL DEFAULT 1.0,
                    total_transfers INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (source_domain, target_domain)
                );",
            )
            .await
    }

    /// Find candidate patterns from `source_domain` that could transfer to `target_domain`.
    ///
    /// Queries patterns with reward > min_source_reward, then ranks by
    /// Thompson-sampled expected reward (prior * source_reward * damping).
    pub async fn find_candidates(
        &self,
        source_domain: &str,
        target_domain: &str,
        limit: usize,
    ) -> Result<Vec<TransferCandidate>> {
        let min_reward = self.config.min_source_reward;
        let damping = self.config.damping_factor;

        // Query high-reward patterns from source domain
        let rows: Vec<(String, f64)> = self
            .db
            .query(
                "SELECT id, reward FROM reasoning_patterns
                 WHERE category = ?1 AND reward > ?2
                 ORDER BY reward DESC
                 LIMIT ?3",
                &[
                    &source_domain as &dyn rusqlite::ToSql,
                    &min_reward,
                    &(limit as i64),
                ],
                |row| Ok((row.get::<_, String>(0)?, row.get::<_, f64>(1)?)),
            )
            .await?;

        // Get or create the prior for this domain pair
        let prior = {
            let history = self.transfer_history.lock();
            history
                .get(&(source_domain.to_string(), target_domain.to_string()))
                .cloned()
                .unwrap_or_else(BetaParams::uniform)
        };

        let mut rng = rand::thread_rng();
        let effective_limit = limit.min(self.config.max_transfers_per_cycle);

        let mut candidates: Vec<TransferCandidate> = rows
            .into_iter()
            .map(|(id, reward)| {
                let thompson_sample = prior.sample(&mut rng);
                let expected_reward = thompson_sample * reward * damping;
                TransferCandidate {
                    source_pattern_id: id,
                    source_domain: source_domain.to_string(),
                    target_domain: target_domain.to_string(),
                    relevance_score: thompson_sample,
                    transfer_prior: prior.clone(),
                    expected_reward,
                    source_reward: reward,
                }
            })
            .collect();

        // Sort by expected reward descending
        candidates.sort_by(|a, b| {
            b.expected_reward
                .partial_cmp(&a.expected_reward)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        candidates.truncate(effective_limit);

        Ok(candidates)
    }

    /// Execute a transfer: create a new pattern in the target domain derived from the source.
    ///
    /// Returns the new pattern's ID.
    pub async fn execute_transfer(&self, candidate: &TransferCandidate) -> Result<String> {
        let damped_reward = candidate.source_reward * self.config.damping_factor;
        let new_id = PatternId::new();
        let new_id_str = new_id.as_str().to_string();

        // Read source pattern data
        let source_data: Option<(String, String, String)> = self
            .db
            .query_one(
                "SELECT problem, solution, context FROM reasoning_patterns WHERE id = ?1",
                &[&candidate.source_pattern_id as &dyn rusqlite::ToSql],
                |row| {
                    Ok((
                        row.get::<_, String>(0).unwrap_or_default(),
                        row.get::<_, String>(1).unwrap_or_default(),
                        row.get::<_, String>(2).unwrap_or_default(),
                    ))
                },
            )
            .await?;

        let (problem, solution, context) = source_data.unwrap_or_else(|| {
            (
                "Transferred pattern".to_string(),
                String::new(),
                String::new(),
            )
        });

        let now = Utc::now().to_rfc3339();

        // Insert the new pattern into the target domain
        self.db
            .execute(
                "INSERT INTO reasoning_patterns (id, category, problem, solution, context, reward, timestamp, parent_id, derivation_type, lineage_depth)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, 'transfer', 1)",
                &[
                    &new_id_str as &dyn rusqlite::ToSql,
                    &candidate.target_domain,
                    &problem,
                    &solution,
                    &context,
                    &damped_reward,
                    &now,
                    &candidate.source_pattern_id,
                ],
            )
            .await?;

        // Record in transfer_history
        let prior = {
            let history = self.transfer_history.lock();
            history
                .get(&(
                    candidate.source_domain.clone(),
                    candidate.target_domain.clone(),
                ))
                .cloned()
                .unwrap_or_else(BetaParams::uniform)
        };

        self.db
            .execute(
                "INSERT INTO transfer_history (source_domain, target_domain, source_pattern_id, target_pattern_id, alpha, beta, timestamp, damped_reward)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
                &[
                    &candidate.source_domain as &dyn rusqlite::ToSql,
                    &candidate.target_domain,
                    &candidate.source_pattern_id,
                    &new_id_str,
                    &prior.alpha,
                    &prior.beta,
                    &now,
                    &damped_reward,
                ],
            )
            .await?;

        Ok(new_id_str)
    }

    /// Record whether a transfer was successful, updating Beta priors.
    pub fn record_outcome(&self, source_domain: &str, target_domain: &str, success: bool) {
        let mut history = self.transfer_history.lock();
        let key = (source_domain.to_string(), target_domain.to_string());
        let prior = history.entry(key).or_insert_with(BetaParams::uniform);
        if success {
            prior.update_success();
        } else {
            prior.update_failure();
        }
    }

    /// Compute the acceleration score for a domain pair.
    ///
    /// Returns the mean of the Beta prior: > 0.5 means transfer helps,
    /// < 0.5 means it hurts. A uniform prior returns exactly 0.5.
    pub fn acceleration_score(&self, source: &str, target: &str) -> f64 {
        let history = self.transfer_history.lock();
        history
            .get(&(source.to_string(), target.to_string()))
            .map(|p| p.mean())
            .unwrap_or(0.5)
    }

    /// Get aggregate transfer statistics.
    pub fn stats(&self) -> TransferStats {
        let history = self.transfer_history.lock();

        let mut total_transfers = 0usize;
        let mut successful_transfers = 0usize;
        let mut best_pair: Option<(String, String, f64)> = None;

        for ((src, tgt), prior) in history.iter() {
            let n = ((prior.alpha - 1.0) + (prior.beta - 1.0)) as usize;
            let s = (prior.alpha - 1.0) as usize;
            total_transfers += n;
            successful_transfers += s;

            let rate = prior.mean();
            if best_pair.as_ref().map_or(true, |(_, _, r)| rate > *r) && n > 0 {
                best_pair = Some((src.clone(), tgt.clone(), rate));
            }
        }

        TransferStats {
            total_transfers,
            successful_transfers,
            domain_pairs: history.len(),
            best_pair,
        }
    }

    /// Load persisted transfer priors from the database.
    pub async fn load_history(&self) -> Result<()> {
        let rows: Vec<(String, String, f64, f64)> = self
            .db
            .query(
                "SELECT source_domain, target_domain, alpha, beta FROM transfer_priors",
                &[],
                |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, f64>(2)?,
                        row.get::<_, f64>(3)?,
                    ))
                },
            )
            .await?;

        let mut history = self.transfer_history.lock();
        for (src, tgt, alpha, beta) in rows {
            history.insert((src, tgt), BetaParams::new(alpha, beta));
        }

        Ok(())
    }

    /// Persist the current transfer priors to the database.
    pub async fn save_history(&self) -> Result<()> {
        let snapshot: Vec<((String, String), BetaParams)> = {
            let history = self.transfer_history.lock();
            history
                .iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect()
        };

        for ((src, tgt), prior) in &snapshot {
            let n = ((prior.alpha - 1.0) + (prior.beta - 1.0)) as i64;
            self.db
                .execute(
                    "INSERT INTO transfer_priors (source_domain, target_domain, alpha, beta, total_transfers)
                     VALUES (?1, ?2, ?3, ?4, ?5)
                     ON CONFLICT(source_domain, target_domain)
                     DO UPDATE SET alpha = ?3, beta = ?4, total_transfers = ?5",
                    &[
                        src as &dyn rusqlite::ToSql,
                        tgt,
                        &prior.alpha,
                        &prior.beta,
                        &n,
                    ],
                )
                .await?;
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: create an in-memory SqliteDb for testing
    fn test_db() -> Arc<SqliteDb> {
        let db = SqliteDb::open_in_memory().unwrap();
        Arc::new(db)
    }

    // Helper: create the reasoning_patterns table used by tests
    async fn create_patterns_table(db: &SqliteDb) {
        db.execute_batch(
            "CREATE TABLE IF NOT EXISTS reasoning_patterns (
                id TEXT PRIMARY KEY,
                category TEXT,
                problem TEXT DEFAULT '',
                solution TEXT DEFAULT '',
                context TEXT DEFAULT '',
                reward REAL DEFAULT 0.0,
                timestamp TEXT DEFAULT '',
                parent_id TEXT,
                derivation_type TEXT,
                lineage_depth INTEGER DEFAULT 0
            )",
        )
        .await
        .unwrap();
    }

    // Helper: insert a pattern into the test DB
    async fn insert_pattern(db: &SqliteDb, id: &str, category: &str, reward: f64) {
        let now = Utc::now().to_rfc3339();
        db.execute(
            "INSERT INTO reasoning_patterns (id, category, problem, solution, reward, timestamp)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            &[
                &id as &dyn rusqlite::ToSql,
                &category,
                &format!("Problem from {}", category),
                &format!("Solution from {}", category),
                &reward,
                &now,
            ],
        )
        .await
        .unwrap();
    }

    // -----------------------------------------------------------------------
    // BetaParams tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_beta_params_new() {
        let b = BetaParams::new(2.0, 3.0);
        assert_eq!(b.alpha, 2.0);
        assert_eq!(b.beta, 3.0);
    }

    #[test]
    fn test_beta_params_new_clamps_to_positive() {
        let b = BetaParams::new(-1.0, -5.0);
        assert!(b.alpha > 0.0);
        assert!(b.beta > 0.0);
    }

    #[test]
    fn test_beta_params_uniform() {
        let b = BetaParams::uniform();
        assert_eq!(b.alpha, 1.0);
        assert_eq!(b.beta, 1.0);
        assert!((b.mean() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_beta_params_mean() {
        let b = BetaParams::new(3.0, 7.0);
        assert!((b.mean() - 0.3).abs() < f64::EPSILON);
    }

    #[test]
    fn test_beta_params_variance() {
        let b = BetaParams::new(2.0, 2.0);
        // Var = 2*2 / (4*4*5) = 4/80 = 0.05
        assert!((b.variance() - 0.05).abs() < 1e-10);
    }

    #[test]
    fn test_beta_params_sample_in_range() {
        let b = BetaParams::new(2.0, 5.0);
        let mut rng = rand::thread_rng();
        for _ in 0..100 {
            let s = b.sample(&mut rng);
            assert!(s >= 0.0, "Sample {} should be >= 0", s);
            assert!(s <= 1.0, "Sample {} should be <= 1", s);
        }
    }

    #[test]
    fn test_beta_params_sample_mean_convergence() {
        // With many samples the empirical mean should approach the true mean
        let b = BetaParams::new(5.0, 3.0);
        let mut rng = rand::thread_rng();
        let n = 5000;
        let sum: f64 = (0..n).map(|_| b.sample(&mut rng)).sum();
        let empirical_mean = sum / n as f64;
        let true_mean = b.mean(); // 5/8 = 0.625
        assert!(
            (empirical_mean - true_mean).abs() < 0.05,
            "Empirical mean {} should be close to true mean {}",
            empirical_mean,
            true_mean,
        );
    }

    #[test]
    fn test_beta_params_update_success() {
        let mut b = BetaParams::uniform();
        b.update_success();
        assert_eq!(b.alpha, 2.0);
        assert_eq!(b.beta, 1.0);
        assert!(b.mean() > 0.5);
    }

    #[test]
    fn test_beta_params_update_failure() {
        let mut b = BetaParams::uniform();
        b.update_failure();
        assert_eq!(b.alpha, 1.0);
        assert_eq!(b.beta, 2.0);
        assert!(b.mean() < 0.5);
    }

    #[test]
    fn test_beta_params_confidence_increases_with_observations() {
        let b1 = BetaParams::uniform();
        let b2 = BetaParams::new(10.0, 10.0);
        assert!(b2.confidence() > b1.confidence());
    }

    #[test]
    fn test_beta_params_confidence_zero_for_uniform() {
        let b = BetaParams::uniform();
        assert!((b.confidence() - 0.0).abs() < f64::EPSILON);
    }

    // -----------------------------------------------------------------------
    // Gamma sampling tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_gamma_sample_positive() {
        let mut rng = rand::thread_rng();
        for _ in 0..100 {
            let g = gamma_sample(2.0, &mut rng);
            assert!(g >= 0.0, "Gamma sample should be non-negative, got {}", g);
        }
    }

    #[test]
    fn test_gamma_sample_shape_less_than_one() {
        let mut rng = rand::thread_rng();
        for _ in 0..50 {
            let g = gamma_sample(0.5, &mut rng);
            assert!(g >= 0.0, "Gamma sample should be non-negative, got {}", g);
        }
    }

    #[test]
    fn test_gamma_sample_shape_one() {
        // Gamma(1, 1) = Exponential(1), mean should be ~1.0
        let mut rng = rand::thread_rng();
        let n = 3000;
        let sum: f64 = (0..n).map(|_| gamma_sample(1.0, &mut rng)).sum();
        let mean = sum / n as f64;
        assert!(
            (mean - 1.0).abs() < 0.15,
            "Gamma(1) mean {} should be near 1.0",
            mean,
        );
    }

    #[test]
    fn test_gamma_sample_zero_shape_returns_zero() {
        let mut rng = rand::thread_rng();
        assert_eq!(gamma_sample(0.0, &mut rng), 0.0);
    }

    // -----------------------------------------------------------------------
    // Standard normal test
    // -----------------------------------------------------------------------

    #[test]
    fn test_standard_normal_mean_near_zero() {
        let mut rng = rand::thread_rng();
        let n = 5000;
        let sum: f64 = (0..n).map(|_| standard_normal(&mut rng)).sum();
        let mean = sum / n as f64;
        assert!(
            mean.abs() < 0.1,
            "Normal mean {} should be near 0",
            mean,
        );
    }

    // -----------------------------------------------------------------------
    // TransferConfig tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_transfer_config_default() {
        let cfg = TransferConfig::default();
        assert!((cfg.min_source_reward - 0.7).abs() < f64::EPSILON);
        assert!((cfg.similarity_threshold - 0.5).abs() < f64::EPSILON);
        assert!((cfg.damping_factor - 0.3).abs() < f64::EPSILON);
        assert_eq!(cfg.max_transfers_per_cycle, 10);
    }

    // -----------------------------------------------------------------------
    // DomainTransferEngine tests
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_engine_init_schema() {
        let db = test_db();
        let engine = DomainTransferEngine::new(db.clone(), TransferConfig::default());
        engine.init_schema().await.unwrap();

        // Verify tables exist
        assert!(db.table_exists("transfer_history").await.unwrap());
        assert!(db.table_exists("transfer_priors").await.unwrap());
    }

    #[tokio::test]
    async fn test_find_candidates_empty_domain() {
        let db = test_db();
        let engine = DomainTransferEngine::new(db.clone(), TransferConfig::default());
        engine.init_schema().await.unwrap();
        create_patterns_table(&db).await;

        let candidates = engine.find_candidates("rust", "python", 5).await.unwrap();
        assert!(candidates.is_empty());
    }

    #[tokio::test]
    async fn test_find_candidates_filters_low_reward() {
        let db = test_db();
        let engine = DomainTransferEngine::new(db.clone(), TransferConfig::default());
        engine.init_schema().await.unwrap();
        create_patterns_table(&db).await;

        // Insert patterns with different rewards
        insert_pattern(&db, "p1", "rust", 0.9).await;
        insert_pattern(&db, "p2", "rust", 0.5).await; // Below min_source_reward
        insert_pattern(&db, "p3", "rust", 0.8).await;

        let candidates = engine.find_candidates("rust", "python", 10).await.unwrap();
        // Only p1 and p3 should pass the 0.7 threshold
        assert_eq!(candidates.len(), 2);
        for c in &candidates {
            assert!(c.source_reward > 0.7);
        }
    }

    #[tokio::test]
    async fn test_find_candidates_respects_limit() {
        let db = test_db();
        let engine = DomainTransferEngine::new(
            db.clone(),
            TransferConfig {
                max_transfers_per_cycle: 2,
                ..Default::default()
            },
        );
        engine.init_schema().await.unwrap();
        create_patterns_table(&db).await;

        for i in 0..5 {
            insert_pattern(&db, &format!("p{}", i), "rust", 0.8 + (i as f64) * 0.01).await;
        }

        let candidates = engine.find_candidates("rust", "go", 10).await.unwrap();
        assert!(candidates.len() <= 2);
    }

    #[tokio::test]
    async fn test_execute_transfer_creates_pattern() {
        let db = test_db();
        let engine = DomainTransferEngine::new(db.clone(), TransferConfig::default());
        engine.init_schema().await.unwrap();
        create_patterns_table(&db).await;
        insert_pattern(&db, "src-1", "rust", 0.9).await;

        let candidate = TransferCandidate {
            source_pattern_id: "src-1".to_string(),
            source_domain: "rust".to_string(),
            target_domain: "python".to_string(),
            relevance_score: 0.8,
            transfer_prior: BetaParams::uniform(),
            expected_reward: 0.27,
            source_reward: 0.9,
        };

        let new_id = engine.execute_transfer(&candidate).await.unwrap();
        assert!(!new_id.is_empty());

        // Verify the pattern was created in the target domain
        let row: Option<(String, f64)> = db
            .query_one(
                "SELECT category, reward FROM reasoning_patterns WHERE id = ?1",
                &[&new_id as &dyn rusqlite::ToSql],
                |row| Ok((row.get::<_, String>(0)?, row.get::<_, f64>(1)?)),
            )
            .await
            .unwrap();

        let (category, reward) = row.unwrap();
        assert_eq!(category, "python");
        assert!((reward - 0.27).abs() < 0.01); // 0.9 * 0.3
    }

    #[tokio::test]
    async fn test_execute_transfer_sets_lineage() {
        let db = test_db();
        let engine = DomainTransferEngine::new(db.clone(), TransferConfig::default());
        engine.init_schema().await.unwrap();
        create_patterns_table(&db).await;
        insert_pattern(&db, "src-2", "testing", 0.85).await;

        let candidate = TransferCandidate {
            source_pattern_id: "src-2".to_string(),
            source_domain: "testing".to_string(),
            target_domain: "devops".to_string(),
            relevance_score: 0.7,
            transfer_prior: BetaParams::uniform(),
            expected_reward: 0.255,
            source_reward: 0.85,
        };

        let new_id = engine.execute_transfer(&candidate).await.unwrap();

        let lineage: Option<(String, String, i64)> = db
            .query_one(
                "SELECT parent_id, derivation_type, lineage_depth FROM reasoning_patterns WHERE id = ?1",
                &[&new_id as &dyn rusqlite::ToSql],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .await
            .unwrap();

        let (parent_id, derivation_type, depth) = lineage.unwrap();
        assert_eq!(parent_id, "src-2");
        assert_eq!(derivation_type, "transfer");
        assert_eq!(depth, 1);
    }

    #[tokio::test]
    async fn test_execute_transfer_records_history() {
        let db = test_db();
        let engine = DomainTransferEngine::new(db.clone(), TransferConfig::default());
        engine.init_schema().await.unwrap();
        create_patterns_table(&db).await;
        insert_pattern(&db, "src-3", "security", 0.75).await;

        let candidate = TransferCandidate {
            source_pattern_id: "src-3".to_string(),
            source_domain: "security".to_string(),
            target_domain: "devops".to_string(),
            relevance_score: 0.6,
            transfer_prior: BetaParams::uniform(),
            expected_reward: 0.225,
            source_reward: 0.75,
        };

        engine.execute_transfer(&candidate).await.unwrap();

        let count: Vec<i64> = db
            .query(
                "SELECT COUNT(*) FROM transfer_history WHERE source_domain = 'security' AND target_domain = 'devops'",
                &[],
                |row| row.get(0),
            )
            .await
            .unwrap();
        assert_eq!(count[0], 1);
    }

    #[test]
    fn test_record_outcome_updates_priors() {
        let db = test_db();
        let engine = DomainTransferEngine::new(db, TransferConfig::default());

        engine.record_outcome("rust", "python", true);
        engine.record_outcome("rust", "python", true);
        engine.record_outcome("rust", "python", false);

        let history = engine.transfer_history.lock();
        let prior = history.get(&("rust".to_string(), "python".to_string())).unwrap();
        assert_eq!(prior.alpha, 3.0); // 1 (uniform) + 2 successes
        assert_eq!(prior.beta, 2.0); // 1 (uniform) + 1 failure
    }

    #[test]
    fn test_acceleration_score_unknown_pair() {
        let db = test_db();
        let engine = DomainTransferEngine::new(db, TransferConfig::default());
        // Unknown pair returns neutral 0.5
        assert!((engine.acceleration_score("a", "b") - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_acceleration_score_positive_transfers() {
        let db = test_db();
        let engine = DomainTransferEngine::new(db, TransferConfig::default());

        for _ in 0..8 {
            engine.record_outcome("rust", "go", true);
        }
        engine.record_outcome("rust", "go", false);
        engine.record_outcome("rust", "go", false);

        let score = engine.acceleration_score("rust", "go");
        assert!(score > 0.5, "Score {} should indicate positive transfer", score);
    }

    #[test]
    fn test_acceleration_score_negative_transfers() {
        let db = test_db();
        let engine = DomainTransferEngine::new(db, TransferConfig::default());

        engine.record_outcome("java", "rust", false);
        engine.record_outcome("java", "rust", false);
        engine.record_outcome("java", "rust", false);

        let score = engine.acceleration_score("java", "rust");
        assert!(score < 0.5, "Score {} should indicate negative transfer", score);
    }

    #[test]
    fn test_stats_empty() {
        let db = test_db();
        let engine = DomainTransferEngine::new(db, TransferConfig::default());

        let stats = engine.stats();
        assert_eq!(stats.total_transfers, 0);
        assert_eq!(stats.successful_transfers, 0);
        assert_eq!(stats.domain_pairs, 0);
        assert!(stats.best_pair.is_none());
    }

    #[test]
    fn test_stats_with_data() {
        let db = test_db();
        let engine = DomainTransferEngine::new(db, TransferConfig::default());

        engine.record_outcome("rust", "go", true);
        engine.record_outcome("rust", "go", true);
        engine.record_outcome("rust", "go", false);
        engine.record_outcome("python", "js", true);

        let stats = engine.stats();
        assert_eq!(stats.total_transfers, 4);
        assert_eq!(stats.successful_transfers, 3);
        assert_eq!(stats.domain_pairs, 2);
        assert!(stats.best_pair.is_some());
    }

    // -----------------------------------------------------------------------
    // Persistence round-trip tests
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_save_and_load_history() {
        let db = test_db();
        let engine = DomainTransferEngine::new(db.clone(), TransferConfig::default());
        engine.init_schema().await.unwrap();

        // Record some outcomes
        engine.record_outcome("rust", "python", true);
        engine.record_outcome("rust", "python", true);
        engine.record_outcome("go", "rust", false);

        // Save to DB
        engine.save_history().await.unwrap();

        // Create a new engine and load
        let engine2 = DomainTransferEngine::new(db.clone(), TransferConfig::default());
        engine2.load_history().await.unwrap();

        let history = engine2.transfer_history.lock();
        let rp = history.get(&("rust".to_string(), "python".to_string())).unwrap();
        assert_eq!(rp.alpha, 3.0);
        assert_eq!(rp.beta, 1.0);

        let gr = history.get(&("go".to_string(), "rust".to_string())).unwrap();
        assert_eq!(gr.alpha, 1.0);
        assert_eq!(gr.beta, 2.0);
    }

    #[tokio::test]
    async fn test_save_history_upsert() {
        let db = test_db();
        let engine = DomainTransferEngine::new(db.clone(), TransferConfig::default());
        engine.init_schema().await.unwrap();

        engine.record_outcome("a", "b", true);
        engine.save_history().await.unwrap();

        engine.record_outcome("a", "b", true);
        engine.save_history().await.unwrap();

        // Should have exactly one row (upserted)
        let count: Vec<i64> = db
            .query(
                "SELECT COUNT(*) FROM transfer_priors",
                &[],
                |row| row.get(0),
            )
            .await
            .unwrap();
        assert_eq!(count[0], 1);
    }

    // -----------------------------------------------------------------------
    // Edge cases
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_same_source_and_target_domain() {
        let db = test_db();
        let engine = DomainTransferEngine::new(db.clone(), TransferConfig::default());
        engine.init_schema().await.unwrap();
        create_patterns_table(&db).await;
        insert_pattern(&db, "self-1", "rust", 0.9).await;

        // Same source and target is allowed (self-transfer)
        let candidates = engine.find_candidates("rust", "rust", 5).await.unwrap();
        assert_eq!(candidates.len(), 1);
    }

    #[tokio::test]
    async fn test_transfer_with_missing_source_pattern() {
        let db = test_db();
        let engine = DomainTransferEngine::new(db.clone(), TransferConfig::default());
        engine.init_schema().await.unwrap();
        create_patterns_table(&db).await;

        // Source pattern doesn't exist — execute_transfer should still succeed
        // (it will use fallback values)
        let candidate = TransferCandidate {
            source_pattern_id: "nonexistent".to_string(),
            source_domain: "rust".to_string(),
            target_domain: "go".to_string(),
            relevance_score: 0.5,
            transfer_prior: BetaParams::uniform(),
            expected_reward: 0.15,
            source_reward: 0.5,
        };

        let new_id = engine.execute_transfer(&candidate).await.unwrap();
        assert!(!new_id.is_empty());
    }

    #[test]
    fn test_record_outcome_creates_new_pair() {
        let db = test_db();
        let engine = DomainTransferEngine::new(db, TransferConfig::default());

        assert_eq!(engine.stats().domain_pairs, 0);
        engine.record_outcome("new-src", "new-tgt", true);
        assert_eq!(engine.stats().domain_pairs, 1);
    }
}
