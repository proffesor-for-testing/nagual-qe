//! Cross-domain transfer learning via Meta Thompson Sampling.
//!
//! Enables nagual to accelerate learning in new domains by transferring
//! priors from established ones. Uses ruvector-domain-expansion's
//! Thompson Sampling with Beta distributions and curiosity bonuses.
//!
//! # Architecture
//!
//! Each nagual pattern category (rust, testing, architecture, ...) maps
//! to a `DomainExpansionEngine` domain. When SONA records an outcome the
//! result is forwarded here so the Thompson Sampling engine can learn
//! which strategies work best per difficulty tier.
//!
//! Transfer is initiated explicitly via `nagual learn transfer apply <src> <tgt>`.
//! The engine extracts compact Beta priors from the source domain and seeds
//! the target with dampened copies, accelerating convergence.
//!
//! # Feature gate
//!
//! All concrete logic requires `feature = "domain-expansion"`. When the
//! feature is disabled every public function compiles to a no-op so
//! callers never need conditional compilation.

#[cfg(feature = "domain-expansion")]
use std::sync::atomic::{AtomicU64, Ordering};

#[cfg(feature = "domain-expansion")]
use std::sync::OnceLock;

#[cfg(feature = "domain-expansion")]
use parking_lot::Mutex;

#[cfg(feature = "domain-expansion")]
use ruvector_domain_expansion::{
    ArmId, ContextBucket, CostCurvePoint, Domain, DomainEmbedding, DomainExpansionEngine, DomainId,
    Evaluation, MetaLearningHealth, PlateauAction, Solution, Task, TransferPrior,
};

// ---------------------------------------------------------------------------
// NagualDomain -- a lightweight Domain impl for nagual pattern categories
// ---------------------------------------------------------------------------

/// A lightweight [`Domain`] implementation that maps nagual pattern
/// categories to the domain-expansion engine.
///
/// The `evaluate()` method extracts the real SONA reward from
/// `solution.data["reward"]` so that `evaluate_and_record` in the engine
/// feeds the actual reward to Thompson Sampling -- not a zero placeholder.
#[cfg(feature = "domain-expansion")]
pub struct NagualDomain {
    id: DomainId,
    name: String,
}

#[cfg(feature = "domain-expansion")]
impl NagualDomain {
    /// Create a new nagual domain from a category name.
    pub fn new(domain_name: &str) -> Self {
        Self {
            id: DomainId(domain_name.to_string()),
            name: domain_name.to_string(),
        }
    }
}

#[cfg(feature = "domain-expansion")]
impl Domain for NagualDomain {
    fn id(&self) -> &DomainId {
        &self.id
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn generate_tasks(&self, count: usize, difficulty: f32) -> Vec<Task> {
        (0..count)
            .map(|i| Task {
                id: format!("{}-task-{}", self.name, i),
                domain_id: self.id.clone(),
                difficulty,
                spec: serde_json::json!({"domain": self.name, "index": i}),
                constraints: vec![],
            })
            .collect()
    }

    fn evaluate(&self, _task: &Task, solution: &Solution) -> Evaluation {
        // CRITICAL-1 fix: Extract the actual SONA reward from solution.data
        // instead of returning Evaluation::zero(). The reward is set by
        // record_domain_outcome() before calling evaluate_and_record().
        let reward = solution
            .data
            .get("reward")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0) as f32;
        Evaluation::composite(reward, reward, reward)
    }

    fn embed(&self, solution: &Solution) -> DomainEmbedding {
        // Simple hash-based embedding of solution content.
        // Known limitation: not semantic -- uses hash for deterministic
        // embedding. Acceptable because Thompson Sampling's cross-domain
        // transfer relies on Beta priors, not embedding similarity.
        use sha3::{
            digest::{ExtendableOutput, Update, XofReader},
            Shake256,
        };
        let mut hasher = Shake256::default();
        hasher.update(solution.content.as_bytes());
        let mut reader = hasher.finalize_xof();
        let mut bytes = [0u8; 64]; // 16 f32s
        reader.read(&mut bytes);
        let vector: Vec<f32> = bytes
            .chunks(4)
            .map(|c| {
                let v = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
                if v.is_finite() {
                    v.clamp(-1.0, 1.0)
                } else {
                    0.0
                }
            })
            .collect();
        DomainEmbedding {
            vector,
            domain_id: self.id.clone(),
            dim: 16,
        }
    }

    fn embedding_dim(&self) -> usize {
        16
    }

    fn reference_solution(&self, _task: &Task) -> Option<Solution> {
        None
    }
}

// ---------------------------------------------------------------------------
// Reward tracker for plateau detection (HIGH-3)
// ---------------------------------------------------------------------------

/// Tracks per-domain running reward averages for plateau detection.
///
/// The crate's built-in plateau detector requires `CostCurvePoint` data
/// in the scoreboard. This tracker feeds cost curve points into the
/// engine's scoreboard as outcomes are recorded, and also provides a
/// simple moving-average fallback.
#[cfg(feature = "domain-expansion")]
struct DomainRewardTracker {
    rewards: std::collections::HashMap<String, Vec<f32>>,
    /// Outcome counter per domain for cycle numbering.
    cycles: std::collections::HashMap<String, u64>,
    window: usize,
}

#[cfg(feature = "domain-expansion")]
impl DomainRewardTracker {
    fn new() -> Self {
        Self {
            rewards: std::collections::HashMap::new(),
            cycles: std::collections::HashMap::new(),
            window: 20,
        }
    }

    fn record(&mut self, domain: &str, reward: f32) -> u64 {
        let entry = self.rewards.entry(domain.to_string()).or_default();
        entry.push(reward);
        // Keep at most 2 * window entries
        if entry.len() > self.window * 2 {
            entry.drain(..self.window);
        }
        let cycle = self.cycles.entry(domain.to_string()).or_insert(0);
        *cycle += 1;
        *cycle
    }

    fn is_plateaued(&self, domain: &str) -> bool {
        if let Some(rewards) = self.rewards.get(domain) {
            if rewards.len() < self.window * 2 {
                return false;
            }
            let n = rewards.len();
            let recent: f32 =
                rewards[n - self.window..].iter().sum::<f32>() / self.window as f32;
            let older_len = self.window.min(n - self.window);
            let older: f32 = rewards[n - self.window - older_len..n - self.window]
                .iter()
                .sum::<f32>()
                / older_len as f32;
            (recent - older).abs() < 0.01 // less than 1% change
        } else {
            false
        }
    }
}

// ---------------------------------------------------------------------------
// Global engine singleton
// ---------------------------------------------------------------------------

/// Outcome counter for auto-persistence (HIGH-1).
#[cfg(feature = "domain-expansion")]
static OUTCOME_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Auto-persist interval: persist Thompson priors every N outcomes.
#[cfg(feature = "domain-expansion")]
const AUTO_PERSIST_INTERVAL: u64 = 50;

/// Access the process-wide domain expansion engine.
///
/// Lazily initialised on first call and shared across SONA recording,
/// CLI commands, and the status display.
///
/// Note: The engine starts with 3 built-in domains (rust_synthesis,
/// structured_planning, tool_orchestration) from DomainExpansionEngine::new().
/// These do not interfere with nagual's auto-registered domains since
/// each domain has independent Thompson priors.
#[cfg(feature = "domain-expansion")]
fn global_expansion_engine() -> &'static Mutex<DomainExpansionEngine> {
    static INSTANCE: OnceLock<Mutex<DomainExpansionEngine>> = OnceLock::new();
    INSTANCE.get_or_init(|| {
        let engine = DomainExpansionEngine::new();
        Mutex::new(engine)
    })
}

/// Access the process-wide reward tracker for plateau detection.
#[cfg(feature = "domain-expansion")]
fn global_reward_tracker() -> &'static Mutex<DomainRewardTracker> {
    static INSTANCE: OnceLock<Mutex<DomainRewardTracker>> = OnceLock::new();
    INSTANCE.get_or_init(|| Mutex::new(DomainRewardTracker::new()))
}

/// Resolve the SQLite database path for auto-persistence.
///
/// Checks (in order):
/// 1. `NAGUAL_DB_PATH` environment variable
/// 2. `sqlite_path` in `~/.nagual/config.toml`
/// 3. Falls back to `nagual.db` (current directory)
#[cfg(feature = "domain-expansion")]
fn resolve_db_path() -> String {
    if let Ok(path) = std::env::var("NAGUAL_DB_PATH") {
        if !path.is_empty() {
            return path;
        }
    }
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
    "nagual.db".to_string()
}

// ---------------------------------------------------------------------------
// Public integration API
// ---------------------------------------------------------------------------

/// Record a pattern outcome in the domain expansion engine.
///
/// Maps nagual categories to domains and difficulty tiers. Called from
/// SONA's `record_outcome_with_modifiers` after each outcome recording.
///
/// ## Fixes applied
///
/// - **CRITICAL-1**: Records the actual SONA reward into the Thompson
///   engine via `evaluate_and_record()` which now calls our `evaluate()`
///   that extracts the reward from `solution.data["reward"]`.
///
/// - **HIGH-1**: Auto-persists Thompson priors every 50 outcomes.
///
/// - **HIGH-2**: Uses `select_arm_curious` for curiosity-boosted arm
///   selection instead of plain `select_arm`.
///
/// - **HIGH-3**: Records cost curve points in the scoreboard for plateau
///   detection and tracks per-domain reward averages.
#[cfg(feature = "domain-expansion")]
pub fn record_domain_outcome(domain: &str, pattern_id: &str, reward: f32, difficulty: f32) {
    let mut engine = global_expansion_engine().lock();

    // Ensure domain is registered.
    let domain_id = DomainId(domain.to_string());
    if !engine.domain_ids().contains(&domain_id) {
        engine.register_domain(Box::new(NagualDomain::new(domain)));
    }

    let bucket = ContextBucket {
        difficulty_tier: if difficulty < 0.33 {
            "easy"
        } else if difficulty < 0.67 {
            "medium"
        } else {
            "hard"
        }
        .to_string(),
        category: domain.to_string(),
    };

    // HIGH-2: Use curiosity-boosted arm selection instead of plain select_arm.
    let arm = engine
        .select_arm_curious(&domain_id, &bucket)
        .unwrap_or_else(|| ArmId("greedy".to_string()));

    // Build task and solution from the outcome.
    // CRITICAL-1: The reward is packed into solution.data["reward"] so that
    // NagualDomain::evaluate() can extract it and return a non-zero Evaluation.
    let task = Task {
        id: pattern_id.to_string(),
        domain_id: domain_id.clone(),
        difficulty,
        spec: serde_json::json!({"pattern": pattern_id}),
        constraints: vec![],
    };
    let solution = Solution {
        task_id: pattern_id.to_string(),
        content: format!("reward:{}", reward),
        data: serde_json::json!({"reward": reward}),
    };

    engine.evaluate_and_record(&domain_id, &task, &solution, bucket, arm);

    // HIGH-3: Record a cost curve point in the scoreboard for plateau detection.
    {
        let mut tracker = global_reward_tracker().lock();
        let cycle = tracker.record(domain, reward);

        // Ensure the scoreboard has a curve for this domain.
        use ruvector_domain_expansion::{ConvergenceThresholds, CostCurve};
        if !engine.scoreboard.curves.contains_key(&domain_id) {
            engine.scoreboard.add_curve(CostCurve::new(
                domain_id.clone(),
                ConvergenceThresholds::default(),
            ));
        }
        if let Some(curve) = engine.scoreboard.curves.get_mut(&domain_id) {
            curve.record(CostCurvePoint {
                cycle,
                accuracy: reward,
                cost_per_solve: 1.0 - reward,
                robustness: reward * 0.95,
                policy_violations: 0,
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_secs_f64())
                    .unwrap_or(0.0),
            });
        }
    }

    // HIGH-1: Auto-persist Thompson priors every AUTO_PERSIST_INTERVAL outcomes.
    let count = OUTCOME_COUNTER.fetch_add(1, Ordering::Relaxed);
    if count % AUTO_PERSIST_INTERVAL == AUTO_PERSIST_INTERVAL - 1 {
        drop(engine); // Release lock before doing I/O
        let db_path = resolve_db_path();
        let _ = persist_domain_state(&db_path);
    }
}

/// No-op fallback when the feature is disabled.
#[cfg(not(feature = "domain-expansion"))]
pub fn record_domain_outcome(_domain: &str, _pattern_id: &str, _reward: f32, _difficulty: f32) {}

/// Get meta-learning health status from the expansion engine.
#[cfg(feature = "domain-expansion")]
pub fn get_expansion_health() -> Option<MetaLearningHealth> {
    let engine = global_expansion_engine().lock();
    Some(engine.meta_health())
}

/// No-op fallback.
#[cfg(not(feature = "domain-expansion"))]
pub fn get_expansion_health() -> Option<DomainExpansionHealthStub> {
    None
}

/// Stub type returned when the feature is disabled so callers can
/// compile without conditional blocks at every call site.
#[cfg(not(feature = "domain-expansion"))]
#[derive(Debug)]
pub struct DomainExpansionHealthStub;

/// Get registered domain names.
pub fn get_expansion_domains() -> Vec<String> {
    #[cfg(feature = "domain-expansion")]
    {
        let engine = global_expansion_engine().lock();
        engine
            .domain_ids()
            .into_iter()
            .map(|d| d.0.clone())
            .collect()
    }
    #[cfg(not(feature = "domain-expansion"))]
    {
        vec![]
    }
}

/// Initiate transfer from source to target domain.
///
/// Extracts compact Beta priors from the source domain and seeds the
/// target domain with dampened copies.
#[cfg(feature = "domain-expansion")]
pub fn initiate_transfer(source: &str, target: &str) -> Result<String, String> {
    let mut engine = global_expansion_engine().lock();
    let source_id = DomainId(source.to_string());
    let target_id = DomainId(target.to_string());

    // Ensure target domain exists.
    if !engine.domain_ids().contains(&target_id) {
        engine.register_domain(Box::new(NagualDomain::new(target)));
    }

    engine.initiate_transfer(&source_id, &target_id);
    Ok(format!("Transfer initiated: {} -> {}", source, target))
}

/// No-op fallback.
#[cfg(not(feature = "domain-expansion"))]
pub fn initiate_transfer(_source: &str, _target: &str) -> Result<String, String> {
    Err("domain-expansion feature not enabled".to_string())
}

/// Check for learning plateaus in a domain.
///
/// Uses both the crate's built-in plateau detection (via cost curve
/// points in the scoreboard) and a local moving-average fallback.
#[cfg(feature = "domain-expansion")]
pub fn check_domain_plateau(domain: &str) -> String {
    let mut engine = global_expansion_engine().lock();
    let domain_id = DomainId(domain.to_string());

    // Try crate-level plateau detection first (uses CostCurvePoints).
    let crate_result = engine.check_plateau(&domain_id);

    match crate_result {
        PlateauAction::Continue => {
            // HIGH-3 fallback: check our own reward tracker
            let tracker = global_reward_tracker().lock();
            if tracker.is_plateaued(domain) {
                "Reward-based plateau detected -- consider cross-domain transfer".to_string()
            } else {
                "Learning is progressing normally".to_string()
            }
        }
        PlateauAction::IncreaseExploration => {
            "Plateau detected -- increasing exploration".to_string()
        }
        PlateauAction::TriggerTransfer => {
            "Extended plateau -- consider cross-domain transfer".to_string()
        }
        PlateauAction::InjectDiversity => "Severe plateau -- injecting diversity".to_string(),
        PlateauAction::Reset => "Critical plateau -- resetting exploration state".to_string(),
    }
}

/// No-op fallback.
#[cfg(not(feature = "domain-expansion"))]
pub fn check_domain_plateau(_domain: &str) -> String {
    "domain-expansion feature not enabled".to_string()
}

// ---------------------------------------------------------------------------
// SQLite persistence for Thompson priors (CRITICAL-2)
// ---------------------------------------------------------------------------

/// Persist actual Thompson Sampling priors (TransferPrior) to SQLite.
///
/// Stores the full serialised Beta parameters per bucket/arm for each
/// registered domain so that priors survive process restarts.
///
/// ## Fix (CRITICAL-2)
///
/// Previously saved a health summary JSON. Now saves the actual
/// `TransferPrior` (Beta parameters) via `thompson.extract_prior()`.
#[cfg(feature = "domain-expansion")]
pub fn persist_domain_state(db_path: &str) -> Result<(), String> {
    use rusqlite::Connection;

    let conn = Connection::open(db_path).map_err(|e| e.to_string())?;
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS domain_expansion_state (
            domain TEXT PRIMARY KEY,
            prior_json TEXT NOT NULL,
            updated_at TEXT NOT NULL DEFAULT (datetime('now'))
        );",
    )
    .map_err(|e| e.to_string())?;

    let engine = global_expansion_engine().lock();

    for domain_id in engine.domain_ids() {
        // Extract the actual TransferPrior (Beta parameters per bucket/arm).
        if let Some(prior) = engine.thompson.extract_prior(&domain_id) {
            // TransferPrior has HashMap<ContextBucket, ...> which serde_json
            // cannot serialize directly (JSON keys must be strings). Convert
            // bucket_priors to a vec of (bucket, arms) pairs for serialization.
            let bucket_priors_vec: Vec<(ContextBucket, std::collections::HashMap<ArmId, ruvector_domain_expansion::BetaParams>)> =
                prior.bucket_priors.into_iter().collect();
            let cost_ema_vec: Vec<(ContextBucket, f32)> =
                prior.cost_ema_priors.into_iter().collect();

            let json = serde_json::json!({
                "source_domain": prior.source_domain.0,
                "bucket_priors": bucket_priors_vec,
                "cost_ema_priors": cost_ema_vec,
                "training_cycles": prior.training_cycles,
                "witness_hash": prior.witness_hash,
            });
            let json_str = serde_json::to_string(&json).map_err(|e| e.to_string())?;
            conn.execute(
                "INSERT OR REPLACE INTO domain_expansion_state (domain, prior_json) VALUES (?, ?)",
                rusqlite::params![domain_id.0, json_str],
            )
            .map_err(|e| e.to_string())?;
        }
    }
    Ok(())
}

/// Load persisted domain state from SQLite.
///
/// Returns the raw domain/JSON pairs for inspection (used by CLI status).
#[cfg(feature = "domain-expansion")]
pub fn load_domain_state(db_path: &str) -> Result<Vec<(String, serde_json::Value)>, String> {
    use rusqlite::Connection;

    let conn = Connection::open(db_path).map_err(|e| e.to_string())?;
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS domain_expansion_state (
            domain TEXT PRIMARY KEY,
            prior_json TEXT NOT NULL,
            updated_at TEXT NOT NULL DEFAULT (datetime('now'))
        );",
    )
    .map_err(|e| e.to_string())?;

    let mut stmt = conn
        .prepare("SELECT domain, prior_json FROM domain_expansion_state ORDER BY updated_at DESC")
        .map_err(|e| e.to_string())?;
    let rows = stmt
        .query_map([], |row| {
            let domain: String = row.get(0)?;
            let json_str: String = row.get(1)?;
            let json: serde_json::Value = serde_json::from_str(&json_str).unwrap_or_default();
            Ok((domain, json))
        })
        .map_err(|e| e.to_string())?;
    rows.collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|e| e.to_string())
}

/// Deserialize a TransferPrior from our custom JSON format.
///
/// Because serde_json cannot serialize HashMap<ContextBucket, ...> (struct
/// keys), we serialize bucket_priors as a vec of (bucket, arms) pairs.
/// This function reverses that transformation.
#[cfg(feature = "domain-expansion")]
fn deserialize_transfer_prior(json: &serde_json::Value) -> TransferPrior {
    use ruvector_domain_expansion::BetaParams;

    let source_domain = json
        .get("source_domain")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown")
        .to_string();

    let training_cycles = json
        .get("training_cycles")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);

    let witness_hash = json
        .get("witness_hash")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    let mut bucket_priors = std::collections::HashMap::new();
    if let Some(bp_array) = json.get("bucket_priors").and_then(|v| v.as_array()) {
        for pair in bp_array {
            if let Some(pair_arr) = pair.as_array() {
                if pair_arr.len() == 2 {
                    if let Ok(bucket) =
                        serde_json::from_value::<ContextBucket>(pair_arr[0].clone())
                    {
                        if let Ok(arms) = serde_json::from_value::<
                            std::collections::HashMap<ArmId, BetaParams>,
                        >(pair_arr[1].clone())
                        {
                            bucket_priors.insert(bucket, arms);
                        }
                    }
                }
            }
        }
    }

    let mut cost_ema_priors = std::collections::HashMap::new();
    if let Some(ce_array) = json.get("cost_ema_priors").and_then(|v| v.as_array()) {
        for pair in ce_array {
            if let Some(pair_arr) = pair.as_array() {
                if pair_arr.len() == 2 {
                    if let Ok(bucket) =
                        serde_json::from_value::<ContextBucket>(pair_arr[0].clone())
                    {
                        if let Some(cost) = pair_arr[1].as_f64() {
                            cost_ema_priors.insert(bucket, cost as f32);
                        }
                    }
                }
            }
        }
    }

    TransferPrior {
        source_domain: DomainId(source_domain),
        bucket_priors,
        cost_ema_priors,
        training_cycles,
        witness_hash,
    }
}

/// Load persisted Thompson priors and restore them into the engine.
///
/// Called on startup (or first access) to restore learning state across
/// process restarts. Returns the number of domains restored.
///
/// ## Fix (CRITICAL-2)
///
/// Actually deserializes `TransferPrior` and feeds it back into the
/// Thompson engine via `init_domain_with_transfer`.
#[cfg(feature = "domain-expansion")]
pub fn load_and_restore_domain_state(db_path: &str) -> Result<usize, String> {
    use rusqlite::Connection;

    let conn = Connection::open(db_path).map_err(|e| e.to_string())?;

    // Check if the table exists; if not, nothing to restore.
    let table_exists: bool = conn
        .query_row(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='domain_expansion_state'",
            [],
            |row| row.get::<_, i64>(0),
        )
        .map(|c| c > 0)
        .unwrap_or(false);

    if !table_exists {
        return Ok(0);
    }

    let mut stmt = conn
        .prepare("SELECT domain, prior_json FROM domain_expansion_state")
        .map_err(|e| e.to_string())?;
    let rows: Vec<(String, String)> = stmt
        .query_map([], |row| {
            let domain: String = row.get(0)?;
            let json_str: String = row.get(1)?;
            Ok((domain, json_str))
        })
        .map_err(|e| e.to_string())?
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|e| e.to_string())?;

    let mut engine = global_expansion_engine().lock();
    let mut restored = 0;

    for (domain, json_str) in rows {
        // Deserialize from the custom format (vec of pairs instead of HashMap).
        if let Ok(json) = serde_json::from_str::<serde_json::Value>(&json_str) {
            let prior = deserialize_transfer_prior(&json);
            let domain_id = DomainId(domain.clone());

            // Ensure domain is registered so it has an entry in the engine.
            if !engine.domain_ids().contains(&domain_id) {
                engine.register_domain(Box::new(NagualDomain::new(&domain)));
            }

            // Restore Thompson priors from the persisted TransferPrior.
            engine
                .thompson
                .init_domain_with_transfer(domain_id, &prior);
            restored += 1;
        }
    }
    Ok(restored)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_record_domain_outcome_does_not_panic() {
        // Should not panic regardless of feature flag.
        record_domain_outcome("testing", "pat-001", 0.85, 0.5);
        record_domain_outcome("architecture", "pat-002", 0.0, 0.0);
        record_domain_outcome("unknown_domain", "pat-003", 1.0, 1.0);
    }

    #[test]
    fn test_get_expansion_domains_returns_vec() {
        let domains = get_expansion_domains();
        // With the feature enabled this will include built-in domains;
        // without the feature it returns an empty vec. Either way it should
        // return a valid Vec<String>.
        let _ = domains;
    }

    #[test]
    fn test_check_domain_plateau_returns_string() {
        let result = check_domain_plateau("nonexistent");
        assert!(!result.is_empty());
    }

    #[test]
    fn test_initiate_transfer_result() {
        let result = initiate_transfer("source_domain", "target_domain");
        // With feature: Ok, without: Err
        #[cfg(feature = "domain-expansion")]
        assert!(result.is_ok());
        #[cfg(not(feature = "domain-expansion"))]
        assert!(result.is_err());
    }

    // Note on MEDIUM-1 (global singleton test contamination):
    // Tests below that use record_domain_outcome() share the global
    // DomainExpansionEngine singleton. This is acceptable because:
    // 1. Each test uses a unique domain name to avoid cross-contamination.
    // 2. Tests that need isolated engines (MEDIUM-3 functional tests)
    //    construct local DomainExpansionEngine instances.

    #[cfg(feature = "domain-expansion")]
    #[test]
    fn test_nagual_domain_evaluate_extracts_reward() {
        // MEDIUM-3 functional test: prove evaluate() returns the actual reward
        // from solution.data, not zero.
        let domain = NagualDomain::new("test_eval_reward");

        let task = Task {
            id: "t1".to_string(),
            domain_id: DomainId("test_eval_reward".to_string()),
            difficulty: 0.5,
            spec: serde_json::json!({}),
            constraints: vec![],
        };

        // Solution with reward=0.85 packed into data.
        let solution = Solution {
            task_id: "t1".to_string(),
            content: "test".to_string(),
            data: serde_json::json!({"reward": 0.85}),
        };
        let eval = domain.evaluate(&task, &solution);

        // composite(0.85, 0.85, 0.85) = 0.6*0.85 + 0.25*0.85 + 0.15*0.85 = 0.85
        assert!(
            (eval.score - 0.85).abs() < 0.01,
            "evaluate() should return actual reward from solution.data, got {}",
            eval.score
        );

        // With no reward in data, should default to 0.0
        let solution_no_reward = Solution {
            task_id: "t2".to_string(),
            content: "test".to_string(),
            data: serde_json::Value::Null,
        };
        let eval_zero = domain.evaluate(&task, &solution_no_reward);
        assert!(
            eval_zero.score < 0.01,
            "evaluate() with no reward data should return ~0.0, got {}",
            eval_zero.score
        );
    }

    #[cfg(feature = "domain-expansion")]
    #[test]
    fn test_thompson_learns_from_real_rewards() {
        // MEDIUM-3 functional test: prove that recording multiple high-reward
        // outcomes shifts the Thompson prior so that subsequent arm selections
        // favor the arm that received high rewards.
        //
        // Uses a LOCAL engine to avoid global singleton contamination.
        let mut engine = DomainExpansionEngine::new();

        let domain_name = "thompson_learn_test";
        engine.register_domain(Box::new(NagualDomain::new(domain_name)));
        let domain_id = DomainId(domain_name.to_string());

        let bucket = ContextBucket {
            difficulty_tier: "medium".to_string(),
            category: domain_name.to_string(),
        };

        // Record many high-reward outcomes for the "greedy" arm
        // and low-reward outcomes for "exploratory".
        for _ in 0..50 {
            let task = Task {
                id: "p1".to_string(),
                domain_id: domain_id.clone(),
                difficulty: 0.5,
                spec: serde_json::json!({}),
                constraints: vec![],
            };
            let high_solution = Solution {
                task_id: "p1".to_string(),
                content: "reward:0.95".to_string(),
                data: serde_json::json!({"reward": 0.95}),
            };
            engine.evaluate_and_record(
                &domain_id,
                &task,
                &high_solution,
                bucket.clone(),
                ArmId("greedy".to_string()),
            );

            let low_solution = Solution {
                task_id: "p2".to_string(),
                content: "reward:0.1".to_string(),
                data: serde_json::json!({"reward": 0.1}),
            };
            engine.evaluate_and_record(
                &domain_id,
                &task,
                &low_solution,
                bucket.clone(),
                ArmId("exploratory".to_string()),
            );
        }

        // Extract the prior and verify that greedy has a higher mean than exploratory.
        let prior = engine
            .thompson
            .extract_prior(&domain_id)
            .expect("prior should exist after recording");

        let greedy_params = prior.get_prior(&bucket, &ArmId("greedy".to_string()));
        let exploratory_params = prior.get_prior(&bucket, &ArmId("exploratory".to_string()));

        assert!(
            greedy_params.mean() > exploratory_params.mean(),
            "Greedy arm (mean={}) should have higher mean than exploratory (mean={}) after high-reward training",
            greedy_params.mean(),
            exploratory_params.mean()
        );
        assert!(
            greedy_params.mean() > 0.5,
            "Greedy arm mean should be above 0.5 after high-reward training, got {}",
            greedy_params.mean()
        );
    }

    #[cfg(feature = "domain-expansion")]
    #[test]
    fn test_nagual_domain_trait() {
        let domain = NagualDomain::new("rust.async");
        assert_eq!(domain.id().0, "rust.async");
        assert_eq!(domain.name(), "rust.async");
        assert_eq!(domain.embedding_dim(), 16);

        let tasks = domain.generate_tasks(3, 0.5);
        assert_eq!(tasks.len(), 3);
        assert!(tasks[0].id.starts_with("rust.async-task-"));
        assert!((tasks[0].difficulty - 0.5).abs() < f32::EPSILON);

        // Evaluate extracts reward from solution data (CRITICAL-1 fix).
        let task = &tasks[0];
        let solution = Solution {
            task_id: task.id.clone(),
            content: "test solution".to_string(),
            data: serde_json::json!({"reward": 0.75}),
        };
        let eval = domain.evaluate(task, &solution);
        // composite(0.75, 0.75, 0.75) = 0.75
        assert!(
            (eval.score - 0.75).abs() < 0.01,
            "Expected ~0.75, got {}",
            eval.score
        );

        // Embed returns a 16-dim vector.
        let embedding = domain.embed(&solution);
        assert_eq!(embedding.dim, 16);
        assert_eq!(embedding.vector.len(), 16);

        // No reference solution.
        assert!(domain.reference_solution(task).is_none());
    }

    #[cfg(feature = "domain-expansion")]
    #[test]
    fn test_record_registers_domain() {
        // After recording an outcome, the domain should be registered.
        record_domain_outcome("test_auto_register", "pat-auto", 0.7, 0.4);
        let domains = get_expansion_domains();
        assert!(
            domains.contains(&"test_auto_register".to_string()),
            "Domain should be auto-registered: {:?}",
            domains
        );
    }

    #[cfg(feature = "domain-expansion")]
    #[test]
    fn test_get_expansion_health_returns_some() {
        let health = get_expansion_health();
        assert!(health.is_some());
    }

    #[cfg(feature = "domain-expansion")]
    #[test]
    fn test_persistence_round_trip() {
        let dir = tempfile::tempdir().expect("tempdir");
        let db_path = dir.path().join("test_expansion.db");
        let db_str = db_path.to_str().unwrap();

        // Record something so the engine has data.
        record_domain_outcome("persist_test", "pat-rt", 0.9, 0.3);

        // Persist.
        persist_domain_state(db_str).expect("persist should succeed");

        // Load.
        let states = load_domain_state(db_str).expect("load should succeed");
        assert!(!states.is_empty(), "Should have at least one domain state");

        // Check our domain is present.
        let found = states.iter().any(|(d, _)| d == "persist_test");
        assert!(found, "persist_test domain should be in loaded state");
    }

    #[cfg(feature = "domain-expansion")]
    #[test]
    fn test_persistence_saves_actual_priors() {
        // MEDIUM-3 functional test: prove that persistence saves actual
        // TransferPrior data (Beta parameters), not just health summaries.
        let dir = tempfile::tempdir().expect("tempdir");
        let db_path = dir.path().join("test_prior_persist.db");
        let db_str = db_path.to_str().unwrap();

        // Record outcomes to build up a prior.
        let domain = "prior_persist_test";
        for i in 0..10 {
            record_domain_outcome(domain, &format!("pat-{}", i), 0.85, 0.5);
        }

        // Persist.
        persist_domain_state(db_str).expect("persist should succeed");

        // Load the raw JSON and verify it contains bucket_priors (not health data).
        let states = load_domain_state(db_str).expect("load should succeed");
        let entry = states.iter().find(|(d, _)| d == domain);
        assert!(entry.is_some(), "Domain should be persisted");

        let (_, json) = entry.unwrap();
        // The JSON should have bucket_priors (TransferPrior structure),
        // not is_learning/is_diverse (old health structure).
        assert!(
            json.get("bucket_priors").is_some() || json.get("source_domain").is_some(),
            "Persisted JSON should contain TransferPrior fields (bucket_priors/source_domain), got: {}",
            serde_json::to_string_pretty(json).unwrap_or_default()
        );
        assert!(
            json.get("is_learning").is_none(),
            "Persisted JSON should NOT contain old health summary fields"
        );
    }

    #[cfg(feature = "domain-expansion")]
    #[test]
    fn test_load_and_restore_domain_state() {
        let dir = tempfile::tempdir().expect("tempdir");
        let db_path = dir.path().join("test_restore.db");
        let db_str = db_path.to_str().unwrap();

        // Record outcomes and persist.
        let domain = "restore_test_domain";
        for i in 0..5 {
            record_domain_outcome(domain, &format!("pat-{}", i), 0.8, 0.3);
        }
        persist_domain_state(db_str).expect("persist should succeed");

        // Restore should succeed and report domains restored.
        let restored = load_and_restore_domain_state(db_str).expect("restore should succeed");
        assert!(
            restored > 0,
            "Should restore at least one domain, got {}",
            restored
        );
    }

    #[cfg(feature = "domain-expansion")]
    #[test]
    fn test_reward_tracker_plateau_detection() {
        // Use a custom tracker with a small window for testing.
        let mut tracker = DomainRewardTracker {
            rewards: std::collections::HashMap::new(),
            cycles: std::collections::HashMap::new(),
            window: 5,
        };
        let domain = "plateau_test";

        // Record stable rewards. With window=5, need at least 10 entries.
        for _ in 0..15 {
            tracker.record(domain, 0.8);
        }
        assert!(
            tracker.is_plateaued(domain),
            "Should detect plateau with constant rewards (len={})",
            tracker.rewards.get(domain).map(|v| v.len()).unwrap_or(0)
        );

        // New domain with insufficient data should not plateau.
        let new_domain = "new_domain";
        for _ in 0..3 {
            tracker.record(new_domain, 0.5);
        }
        assert!(
            !tracker.is_plateaued(new_domain),
            "Should not detect plateau with insufficient data"
        );
    }
}
