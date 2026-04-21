//! KOS P10: Compute Routing Ladder
//!
//! 4-lane compute routing: Reflex (cached answers), Retrieval (pattern lookup),
//! Heavy (LLM inference), Human (requires human review).
//!
//! Self-learning latency tracking and automatic reflex cache promotion.
//!
//! # Lanes
//!
//! | Lane      | Target Latency | Description                        |
//! |-----------|----------------|------------------------------------|
//! | Reflex    | < 1ms          | Cached high-confidence answers     |
//! | Retrieval | < 10ms         | Pattern lookup from knowledge base |
//! | Heavy     | < 5000ms       | Requires LLM inference             |
//! | Human     | Indefinite     | Requires human-in-the-loop         |
//!
//! # Promotion
//!
//! When a retrieval result is hit `promotion_after_hits` times, it is
//! automatically promoted into the reflex cache for sub-millisecond access.

use chrono::{DateTime, Utc};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use crate::db::SqliteDb;
use crate::error::Result;

// ---------------------------------------------------------------------------
// ComputeLane
// ---------------------------------------------------------------------------

/// One of the four compute lanes a query can be routed to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ComputeLane {
    /// Sub-millisecond cached responses for high-confidence, frequently-hit queries.
    Reflex,
    /// Pattern lookup from the knowledge base (< 10 ms).
    Retrieval,
    /// Full LLM inference (< 5 000 ms).
    Heavy,
    /// Requires human-in-the-loop review (indefinite latency).
    Human,
}

impl ComputeLane {
    /// String label for the lane.
    pub fn as_str(&self) -> &'static str {
        match self {
            ComputeLane::Reflex => "reflex",
            ComputeLane::Retrieval => "retrieval",
            ComputeLane::Heavy => "heavy",
            ComputeLane::Human => "human",
        }
    }

    /// Maximum expected latency in milliseconds.
    pub fn max_latency_ms(&self) -> u64 {
        match self {
            ComputeLane::Reflex => 1,
            ComputeLane::Retrieval => 10,
            ComputeLane::Heavy => 5000,
            ComputeLane::Human => u64::MAX,
        }
    }

    /// Relative cost weight (higher = more expensive).
    pub fn cost_weight(&self) -> f64 {
        match self {
            ComputeLane::Reflex => 0.01,
            ComputeLane::Retrieval => 0.1,
            ComputeLane::Heavy => 1.0,
            ComputeLane::Human => 10.0,
        }
    }
}

impl From<&str> for ComputeLane {
    fn from(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "reflex" => ComputeLane::Reflex,
            "retrieval" => ComputeLane::Retrieval,
            "heavy" => ComputeLane::Heavy,
            "human" => ComputeLane::Human,
            _ => ComputeLane::Heavy, // safe default
        }
    }
}

impl std::fmt::Display for ComputeLane {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

// ---------------------------------------------------------------------------
// LadderConfig
// ---------------------------------------------------------------------------

/// Configuration for the routing ladder.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LadderConfig {
    /// Minimum confidence for a reflex cache hit to be used (default: 0.95).
    pub reflex_confidence_threshold: f64,
    /// Maximum number of entries in the in-memory reflex cache (default: 1000).
    pub reflex_cache_size: usize,
    /// Minimum pattern reward to qualify for the retrieval lane (default: 0.7).
    pub retrieval_min_reward: f64,
    /// Complexity below this goes to retrieval instead of heavy (default: 0.3).
    pub heavy_threshold: f64,
    /// Complexity above this requires human review (default: 0.9).
    pub human_threshold: f64,
    /// Promote to reflex after this many successful retrieval hits (default: 5).
    pub promotion_after_hits: u32,
    /// Whether to track per-lane latency (default: true).
    pub latency_tracking_enabled: bool,
    /// Expire reflex entries older than this (seconds, default: 86400 = 24h).
    pub max_reflex_age_secs: u64,
}

impl Default for LadderConfig {
    fn default() -> Self {
        Self {
            reflex_confidence_threshold: 0.95,
            reflex_cache_size: 1000,
            retrieval_min_reward: 0.7,
            heavy_threshold: 0.3,
            human_threshold: 0.9,
            promotion_after_hits: 5,
            latency_tracking_enabled: true,
            max_reflex_age_secs: 86400,
        }
    }
}

// ---------------------------------------------------------------------------
// ReflexEntry
// ---------------------------------------------------------------------------

/// A single entry in the reflex cache.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReflexEntry {
    /// Deterministic hash of the query text.
    pub query_hash: String,
    /// Cached response content.
    pub response: String,
    /// Confidence of the cached response.
    pub confidence: f64,
    /// Number of times this entry has been hit.
    pub hit_count: u64,
    /// When the entry was first created.
    pub created_at: DateTime<Utc>,
    /// Most recent hit timestamp.
    pub last_hit: DateTime<Utc>,
    /// Optional originating pattern id.
    pub source_pattern_id: Option<String>,
}

// ---------------------------------------------------------------------------
// LadderDecision
// ---------------------------------------------------------------------------

/// The result of routing a query through the ladder.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LadderDecision {
    /// Which compute lane was selected.
    pub lane: ComputeLane,
    /// Routing confidence (0.0 - 1.0).
    pub confidence: f64,
    /// Hash of the routed query.
    pub query_hash: String,
    /// Human-readable reasoning.
    pub reasoning: String,
    /// Estimated latency in milliseconds.
    pub estimated_latency_ms: u64,
    /// Whether this decision was served from the reflex cache.
    pub reflex_hit: bool,
}

// ---------------------------------------------------------------------------
// LatencyRecord
// ---------------------------------------------------------------------------

/// A single latency observation for a lane.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyRecord {
    /// The lane that was used.
    pub lane: ComputeLane,
    /// Actual measured latency.
    pub actual_latency_ms: u64,
    /// Estimated latency at routing time.
    pub estimated_latency_ms: u64,
    /// When the record was captured.
    pub recorded_at: DateTime<Utc>,
}

// ---------------------------------------------------------------------------
// LadderStats
// ---------------------------------------------------------------------------

/// Aggregate statistics for the routing ladder.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LadderStats {
    /// Number of reflex cache hits.
    pub reflex_hits: u64,
    /// Number of reflex cache misses.
    pub reflex_misses: u64,
    /// Hit rate (reflex_hits / total_requests), 0.0 when no requests.
    pub reflex_hit_rate: f64,
    /// Total queries routed to the retrieval lane.
    pub retrieval_count: u64,
    /// Total queries routed to the heavy lane.
    pub heavy_count: u64,
    /// Total queries routed to the human lane.
    pub human_count: u64,
    /// Average measured latency per lane (lane name -> avg ms).
    pub avg_latency_by_lane: HashMap<String, f64>,
    /// Total number of routing requests.
    pub total_requests: u64,
    /// Current in-memory reflex cache size.
    pub cache_size: usize,
}

// ---------------------------------------------------------------------------
// PromotionResult
// ---------------------------------------------------------------------------

/// Outcome of a reflex promotion attempt.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromotionResult {
    /// Pattern that was (or was not) promoted.
    pub pattern_id: String,
    /// Query hash associated with the promotion.
    pub query_hash: String,
    /// Whether promotion actually occurred.
    pub promoted: bool,
    /// Explanation.
    pub reason: String,
}

// ---------------------------------------------------------------------------
// query_hash helper
// ---------------------------------------------------------------------------

/// Produce a deterministic 16-hex-char hash of a query string.
/// The query is lowercased and trimmed before hashing.
fn query_hash(query: &str) -> String {
    use std::collections::hash_map::DefaultHasher;
    let mut hasher = DefaultHasher::new();
    query.to_lowercase().trim().hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

// ---------------------------------------------------------------------------
// RoutingLadder
// ---------------------------------------------------------------------------

/// The core routing ladder, backed by SQLite for persistence and an in-memory
/// reflex cache for sub-millisecond lookups.
pub struct RoutingLadder {
    db: Arc<SqliteDb>,
    config: LadderConfig,
    reflex_cache: RwLock<HashMap<String, ReflexEntry>>,
    latency_history: RwLock<Vec<LatencyRecord>>,
    tiering: Option<Arc<crate::tiering::TieringManager>>,
}

impl RoutingLadder {
    /// Create a new routing ladder, initialising tables and loading any
    /// persisted reflex entries into memory.
    pub async fn new(db: Arc<SqliteDb>, config: LadderConfig) -> Result<Self> {
        // Create tables
        db.execute_batch(
            "CREATE TABLE IF NOT EXISTS routing_reflex_cache (
                query_hash TEXT PRIMARY KEY,
                response TEXT NOT NULL,
                confidence REAL NOT NULL,
                hit_count INTEGER DEFAULT 0,
                created_at TEXT NOT NULL,
                last_hit TEXT NOT NULL,
                source_pattern_id TEXT
            );

            CREATE TABLE IF NOT EXISTS routing_latency_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                lane TEXT NOT NULL,
                actual_latency_ms INTEGER NOT NULL,
                estimated_latency_ms INTEGER NOT NULL,
                recorded_at TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS routing_retrieval_hits (
                query_hash TEXT NOT NULL,
                pattern_id TEXT NOT NULL,
                hit_count INTEGER DEFAULT 1,
                last_hit TEXT NOT NULL,
                PRIMARY KEY (query_hash, pattern_id)
            );

            CREATE INDEX IF NOT EXISTS idx_reflex_confidence
                ON routing_reflex_cache(confidence DESC);
            CREATE INDEX IF NOT EXISTS idx_latency_lane
                ON routing_latency_log(lane);",
        )
        .await?;

        // Load persisted reflex entries into memory
        let rows: Vec<ReflexEntry> = db
            .query(
                "SELECT query_hash, response, confidence, hit_count, created_at, last_hit, source_pattern_id
                 FROM routing_reflex_cache",
                &[],
                |row| {
                    let created_str: String = row.get(4)?;
                    let last_hit_str: String = row.get(5)?;
                    let created_at = chrono::DateTime::parse_from_rfc3339(&created_str)
                        .map(|dt| dt.with_timezone(&Utc))
                        .unwrap_or_else(|_| Utc::now());
                    let last_hit = chrono::DateTime::parse_from_rfc3339(&last_hit_str)
                        .map(|dt| dt.with_timezone(&Utc))
                        .unwrap_or_else(|_| Utc::now());

                    Ok(ReflexEntry {
                        query_hash: row.get(0)?,
                        response: row.get(1)?,
                        confidence: row.get(2)?,
                        hit_count: row.get::<_, i64>(3)? as u64,
                        created_at,
                        last_hit,
                        source_pattern_id: row.get(6)?,
                    })
                },
            )
            .await?;

        let mut cache = HashMap::new();
        for entry in rows {
            cache.insert(entry.query_hash.clone(), entry);
        }

        Ok(Self {
            db,
            config,
            reflex_cache: RwLock::new(cache),
            latency_history: RwLock::new(Vec::new()),
            tiering: None,
        })
    }

    /// Attach a tiering manager for cost-weighted routing decisions.
    pub fn with_tiering(mut self, tiering: Arc<crate::tiering::TieringManager>) -> Self {
        self.tiering = Some(tiering);
        self
    }

    // ----- routing --------------------------------------------------------

    /// Route a query to the appropriate compute lane.
    ///
    /// 1. Check the reflex cache first (sub-millisecond).
    /// 2. Fall through to complexity-based routing.
    pub fn route(&self, query: &str, complexity: f64) -> Result<LadderDecision> {
        let hash = query_hash(query);

        // 1. Check reflex cache
        if let Some(entry) = self.check_reflex(query) {
            if entry.confidence >= self.config.reflex_confidence_threshold {
                return Ok(LadderDecision {
                    lane: ComputeLane::Reflex,
                    confidence: entry.confidence,
                    query_hash: hash,
                    reasoning: format!(
                        "Reflex cache hit (confidence {:.2}, {} prior hits)",
                        entry.confidence, entry.hit_count
                    ),
                    estimated_latency_ms: ComputeLane::Reflex.max_latency_ms(),
                    reflex_hit: true,
                });
            }
        }

        // 2. Complexity-based routing
        if complexity >= self.config.human_threshold {
            Ok(LadderDecision {
                lane: ComputeLane::Human,
                confidence: complexity,
                query_hash: hash,
                reasoning: format!(
                    "Complexity {:.2} >= human threshold {:.2}",
                    complexity, self.config.human_threshold
                ),
                estimated_latency_ms: ComputeLane::Human.max_latency_ms(),
                reflex_hit: false,
            })
        } else if complexity >= self.config.heavy_threshold {
            Ok(LadderDecision {
                lane: ComputeLane::Heavy,
                confidence: complexity,
                query_hash: hash,
                reasoning: format!(
                    "Complexity {:.2} >= heavy threshold {:.2}",
                    complexity, self.config.heavy_threshold
                ),
                estimated_latency_ms: ComputeLane::Heavy.max_latency_ms(),
                reflex_hit: false,
            })
        } else {
            Ok(LadderDecision {
                lane: ComputeLane::Retrieval,
                confidence: 1.0 - complexity,
                query_hash: hash,
                reasoning: format!(
                    "Complexity {:.2} < heavy threshold {:.2}, using retrieval",
                    complexity, self.config.heavy_threshold
                ),
                estimated_latency_ms: ComputeLane::Retrieval.max_latency_ms(),
                reflex_hit: false,
            })
        }
    }

    // ----- reflex cache ---------------------------------------------------

    /// Look up a query in the in-memory reflex cache.
    /// Returns a clone of the entry and bumps the hit counter.
    pub fn check_reflex(&self, query: &str) -> Option<ReflexEntry> {
        let hash = query_hash(query);
        let mut cache = self.reflex_cache.write();
        if let Some(entry) = cache.get_mut(&hash) {
            entry.hit_count += 1;
            entry.last_hit = Utc::now();
            Some(entry.clone())
        } else {
            None
        }
    }

    /// Store a new reflex entry (in-memory + SQLite).
    pub async fn store_reflex(
        &self,
        query: &str,
        response: &str,
        confidence: f64,
        source_pattern_id: Option<&str>,
    ) -> Result<()> {
        let hash = query_hash(query);
        let now = Utc::now();
        let now_str = now.to_rfc3339();

        let entry = ReflexEntry {
            query_hash: hash.clone(),
            response: response.to_string(),
            confidence,
            hit_count: 0,
            created_at: now,
            last_hit: now,
            source_pattern_id: source_pattern_id.map(|s| s.to_string()),
        };

        // Persist to SQLite
        self.db
            .execute(
                "INSERT OR REPLACE INTO routing_reflex_cache
                    (query_hash, response, confidence, hit_count, created_at, last_hit, source_pattern_id)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                &[
                    &hash as &dyn rusqlite::ToSql,
                    &response as &dyn rusqlite::ToSql,
                    &confidence as &dyn rusqlite::ToSql,
                    &(entry.hit_count as i64) as &dyn rusqlite::ToSql,
                    &now_str as &dyn rusqlite::ToSql,
                    &now_str as &dyn rusqlite::ToSql,
                    &source_pattern_id as &dyn rusqlite::ToSql,
                ],
            )
            .await?;

        // Update in-memory cache (evict oldest if full)
        let mut cache = self.reflex_cache.write();
        if cache.len() >= self.config.reflex_cache_size && !cache.contains_key(&hash) {
            // Evict the entry with the oldest last_hit
            if let Some(oldest_key) = cache
                .iter()
                .min_by_key(|(_, v)| v.last_hit)
                .map(|(k, _)| k.clone())
            {
                cache.remove(&oldest_key);
            }
        }
        cache.insert(hash, entry);

        Ok(())
    }

    /// Promote a retrieval result to the reflex cache.
    pub async fn promote_to_reflex(
        &self,
        pattern_id: &str,
        query: &str,
        response: &str,
    ) -> Result<PromotionResult> {
        let hash = query_hash(query);

        // Check if already in cache
        {
            let cache = self.reflex_cache.read();
            if cache.contains_key(&hash) {
                return Ok(PromotionResult {
                    pattern_id: pattern_id.to_string(),
                    query_hash: hash,
                    promoted: false,
                    reason: "Already in reflex cache".to_string(),
                });
            }
        }

        // Store with high confidence
        self.store_reflex(query, response, 0.99, Some(pattern_id))
            .await?;

        Ok(PromotionResult {
            pattern_id: pattern_id.to_string(),
            query_hash: hash,
            promoted: true,
            reason: format!(
                "Promoted after reaching {} retrieval hits",
                self.config.promotion_after_hits
            ),
        })
    }

    // ----- retrieval hit tracking -----------------------------------------

    /// Record a retrieval hit for a (query, pattern) pair.
    /// If the hit count reaches `promotion_after_hits`, auto-promote to reflex.
    pub async fn record_retrieval_hit(
        &self,
        query: &str,
        pattern_id: &str,
    ) -> Result<Option<PromotionResult>> {
        let hash = query_hash(query);
        let now_str = Utc::now().to_rfc3339();

        // Upsert retrieval hit counter
        self.db
            .execute(
                "INSERT INTO routing_retrieval_hits (query_hash, pattern_id, hit_count, last_hit)
                 VALUES (?1, ?2, 1, ?3)
                 ON CONFLICT(query_hash, pattern_id) DO UPDATE SET
                     hit_count = hit_count + 1,
                     last_hit = ?3",
                &[
                    &hash as &dyn rusqlite::ToSql,
                    &pattern_id as &dyn rusqlite::ToSql,
                    &now_str as &dyn rusqlite::ToSql,
                ],
            )
            .await?;

        // Read current count
        let count: Option<i64> = self
            .db
            .query_one(
                "SELECT hit_count FROM routing_retrieval_hits WHERE query_hash = ?1 AND pattern_id = ?2",
                &[&hash as &dyn rusqlite::ToSql, &pattern_id as &dyn rusqlite::ToSql],
                |row| row.get(0),
            )
            .await?;

        let count = count.unwrap_or(0) as u32;

        if count >= self.config.promotion_after_hits {
            // Build a synthetic response for promotion
            let response = format!("[auto-promoted from pattern {}]", pattern_id);
            let result = self.promote_to_reflex(pattern_id, query, &response).await?;
            Ok(Some(result))
        } else {
            Ok(None)
        }
    }

    // ----- latency tracking -----------------------------------------------

    /// Record an actual latency observation.
    pub async fn record_latency(
        &self,
        lane: ComputeLane,
        actual_ms: u64,
        estimated_ms: u64,
    ) -> Result<()> {
        let now = Utc::now();
        let now_str = now.to_rfc3339();

        // Persist
        self.db
            .execute(
                "INSERT INTO routing_latency_log (lane, actual_latency_ms, estimated_latency_ms, recorded_at)
                 VALUES (?1, ?2, ?3, ?4)",
                &[
                    &lane.as_str() as &dyn rusqlite::ToSql,
                    &(actual_ms as i64) as &dyn rusqlite::ToSql,
                    &(estimated_ms as i64) as &dyn rusqlite::ToSql,
                    &now_str as &dyn rusqlite::ToSql,
                ],
            )
            .await?;

        // Also keep in memory for fast avg computation
        if self.config.latency_tracking_enabled {
            let record = LatencyRecord {
                lane,
                actual_latency_ms: actual_ms,
                estimated_latency_ms: estimated_ms,
                recorded_at: now,
            };
            self.latency_history.write().push(record);
        }

        Ok(())
    }

    /// Average measured latency for a given lane (from in-memory history).
    pub fn avg_latency(&self, lane: ComputeLane) -> f64 {
        let history = self.latency_history.read();
        let (sum, count) = history
            .iter()
            .filter(|r| r.lane == lane)
            .fold((0u64, 0u64), |(s, c), r| (s + r.actual_latency_ms, c + 1));
        if count == 0 {
            0.0
        } else {
            sum as f64 / count as f64
        }
    }

    // ----- expiration -----------------------------------------------------

    /// Remove reflex entries older than `max_reflex_age_secs`.
    /// Returns the number of entries removed.
    pub async fn expire_stale_reflexes(&self) -> Result<u64> {
        let cutoff = Utc::now()
            - chrono::Duration::seconds(self.config.max_reflex_age_secs as i64);
        let cutoff_str = cutoff.to_rfc3339();

        // Remove from SQLite
        self.db
            .execute(
                "DELETE FROM routing_reflex_cache WHERE last_hit < ?1",
                &[&cutoff_str as &dyn rusqlite::ToSql],
            )
            .await?;

        // Remove from in-memory cache
        let mut cache = self.reflex_cache.write();
        let before = cache.len();
        cache.retain(|_, entry| entry.last_hit >= cutoff);
        let removed = (before - cache.len()) as u64;

        Ok(removed)
    }

    // ----- stats ----------------------------------------------------------

    /// Aggregate statistics from the routing ladder.
    pub async fn stats(&self) -> Result<LadderStats> {
        // Lane counts from latency log
        let lane_counts: Vec<(String, i64)> = self
            .db
            .query(
                "SELECT lane, COUNT(*) FROM routing_latency_log GROUP BY lane",
                &[],
                |row| Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?)),
            )
            .await?;

        let mut retrieval_count: u64 = 0;
        let mut heavy_count: u64 = 0;
        let mut human_count: u64 = 0;
        let mut reflex_db_count: u64 = 0;

        for (lane_str, count) in &lane_counts {
            match lane_str.as_str() {
                "retrieval" => retrieval_count = *count as u64,
                "heavy" => heavy_count = *count as u64,
                "human" => human_count = *count as u64,
                "reflex" => reflex_db_count = *count as u64,
                _ => {}
            }
        }

        // Avg latency by lane from in-memory history
        let mut avg_latency_by_lane = HashMap::new();
        for lane in &[
            ComputeLane::Reflex,
            ComputeLane::Retrieval,
            ComputeLane::Heavy,
            ComputeLane::Human,
        ] {
            let avg = self.avg_latency(*lane);
            if avg > 0.0 {
                avg_latency_by_lane.insert(lane.as_str().to_string(), avg);
            }
        }

        let cache_size = self.reflex_cache_size();

        // Reflex hit / miss estimation from cache
        let total_reflex_hits: u64 = {
            let cache = self.reflex_cache.read();
            cache.values().map(|e| e.hit_count).sum()
        };

        let total_requests = reflex_db_count + retrieval_count + heavy_count + human_count;
        let reflex_misses = total_requests.saturating_sub(total_reflex_hits);

        let reflex_hit_rate = if total_requests == 0 {
            0.0
        } else {
            total_reflex_hits as f64 / total_requests as f64
        };

        Ok(LadderStats {
            reflex_hits: total_reflex_hits,
            reflex_misses,
            reflex_hit_rate,
            retrieval_count,
            heavy_count,
            human_count,
            avg_latency_by_lane,
            total_requests,
            cache_size,
        })
    }

    // ----- utilities ------------------------------------------------------

    /// Clear all entries from the reflex cache (memory + SQLite).
    pub async fn clear_reflex_cache(&self) -> Result<()> {
        self.db
            .execute_batch("DELETE FROM routing_reflex_cache")
            .await?;
        self.reflex_cache.write().clear();
        Ok(())
    }

    /// Number of entries currently in the in-memory reflex cache.
    pub fn reflex_cache_size(&self) -> usize {
        self.reflex_cache.read().len()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: create an in-memory SQLite database wrapped in Arc.
    async fn setup_test_db() -> Arc<SqliteDb> {
        let db = SqliteDb::open_in_memory().unwrap();
        Arc::new(db)
    }

    // ---- ComputeLane tests (4) -------------------------------------------

    #[test]
    fn test_compute_lane_as_str() {
        assert_eq!(ComputeLane::Reflex.as_str(), "reflex");
        assert_eq!(ComputeLane::Retrieval.as_str(), "retrieval");
        assert_eq!(ComputeLane::Heavy.as_str(), "heavy");
        assert_eq!(ComputeLane::Human.as_str(), "human");
    }

    #[test]
    fn test_compute_lane_from_str() {
        assert_eq!(ComputeLane::from("reflex"), ComputeLane::Reflex);
        assert_eq!(ComputeLane::from("RETRIEVAL"), ComputeLane::Retrieval);
        assert_eq!(ComputeLane::from("Heavy"), ComputeLane::Heavy);
        assert_eq!(ComputeLane::from("human"), ComputeLane::Human);
        // Unknown defaults to Heavy
        assert_eq!(ComputeLane::from("unknown"), ComputeLane::Heavy);
    }

    #[test]
    fn test_compute_lane_max_latency() {
        assert_eq!(ComputeLane::Reflex.max_latency_ms(), 1);
        assert_eq!(ComputeLane::Retrieval.max_latency_ms(), 10);
        assert_eq!(ComputeLane::Heavy.max_latency_ms(), 5000);
        assert_eq!(ComputeLane::Human.max_latency_ms(), u64::MAX);
    }

    #[test]
    fn test_compute_lane_cost_weight() {
        assert!(ComputeLane::Reflex.cost_weight() < ComputeLane::Retrieval.cost_weight());
        assert!(ComputeLane::Retrieval.cost_weight() < ComputeLane::Heavy.cost_weight());
        assert!(ComputeLane::Heavy.cost_weight() < ComputeLane::Human.cost_weight());
    }

    // ---- LadderConfig tests (2) ------------------------------------------

    #[test]
    fn test_ladder_config_defaults() {
        let cfg = LadderConfig::default();
        assert!((cfg.reflex_confidence_threshold - 0.95).abs() < f64::EPSILON);
        assert_eq!(cfg.reflex_cache_size, 1000);
        assert!((cfg.retrieval_min_reward - 0.7).abs() < f64::EPSILON);
        assert!((cfg.heavy_threshold - 0.3).abs() < f64::EPSILON);
        assert!((cfg.human_threshold - 0.9).abs() < f64::EPSILON);
        assert_eq!(cfg.promotion_after_hits, 5);
        assert!(cfg.latency_tracking_enabled);
        assert_eq!(cfg.max_reflex_age_secs, 86400);
    }

    #[test]
    fn test_ladder_config_custom() {
        let cfg = LadderConfig {
            reflex_confidence_threshold: 0.8,
            reflex_cache_size: 500,
            retrieval_min_reward: 0.5,
            heavy_threshold: 0.4,
            human_threshold: 0.95,
            promotion_after_hits: 10,
            latency_tracking_enabled: false,
            max_reflex_age_secs: 3600,
        };
        assert!((cfg.reflex_confidence_threshold - 0.8).abs() < f64::EPSILON);
        assert_eq!(cfg.reflex_cache_size, 500);
        assert_eq!(cfg.promotion_after_hits, 10);
        assert!(!cfg.latency_tracking_enabled);
    }

    // ---- query_hash tests (2) --------------------------------------------

    #[test]
    fn test_query_hash_deterministic() {
        let h1 = query_hash("hello world");
        let h2 = query_hash("hello world");
        assert_eq!(h1, h2);
        assert_eq!(h1.len(), 16);
    }

    #[test]
    fn test_query_hash_case_insensitive() {
        let h1 = query_hash("Hello World");
        let h2 = query_hash("hello world");
        assert_eq!(h1, h2);
    }

    // ---- route tests (5) -------------------------------------------------

    #[tokio::test]
    async fn test_route_reflex_hit() {
        let db = setup_test_db().await;
        let ladder = RoutingLadder::new(db, LadderConfig::default()).await.unwrap();

        // Seed a reflex entry with high confidence
        ladder
            .store_reflex("test query", "cached answer", 0.99, None)
            .await
            .unwrap();

        let decision = ladder.route("test query", 0.5).unwrap();
        assert_eq!(decision.lane, ComputeLane::Reflex);
        assert!(decision.reflex_hit);
        assert!(decision.reasoning.contains("Reflex cache hit"));
    }

    #[tokio::test]
    async fn test_route_retrieval_lane() {
        let db = setup_test_db().await;
        let ladder = RoutingLadder::new(db, LadderConfig::default()).await.unwrap();

        // Low complexity, no reflex entry
        let decision = ladder.route("simple query", 0.1).unwrap();
        assert_eq!(decision.lane, ComputeLane::Retrieval);
        assert!(!decision.reflex_hit);
    }

    #[tokio::test]
    async fn test_route_heavy_lane() {
        let db = setup_test_db().await;
        let ladder = RoutingLadder::new(db, LadderConfig::default()).await.unwrap();

        // Medium complexity
        let decision = ladder.route("moderate query", 0.5).unwrap();
        assert_eq!(decision.lane, ComputeLane::Heavy);
        assert!(!decision.reflex_hit);
    }

    #[tokio::test]
    async fn test_route_human_lane() {
        let db = setup_test_db().await;
        let ladder = RoutingLadder::new(db, LadderConfig::default()).await.unwrap();

        // Very high complexity
        let decision = ladder.route("extremely complex query", 0.95).unwrap();
        assert_eq!(decision.lane, ComputeLane::Human);
        assert!(!decision.reflex_hit);
    }

    #[tokio::test]
    async fn test_route_reflex_miss_falls_through() {
        let db = setup_test_db().await;
        let ladder = RoutingLadder::new(db, LadderConfig::default()).await.unwrap();

        // Store reflex entry with LOW confidence (below threshold)
        ladder
            .store_reflex("low confidence query", "maybe", 0.5, None)
            .await
            .unwrap();

        let decision = ladder.route("low confidence query", 0.2).unwrap();
        // Should NOT use reflex, falls through to retrieval
        assert_ne!(decision.lane, ComputeLane::Reflex);
        assert!(!decision.reflex_hit);
    }

    // ---- store/check_reflex tests (3) ------------------------------------

    #[tokio::test]
    async fn test_store_and_check_reflex() {
        let db = setup_test_db().await;
        let ladder = RoutingLadder::new(db, LadderConfig::default()).await.unwrap();

        ladder
            .store_reflex("greeting", "hello!", 0.95, Some("p-1"))
            .await
            .unwrap();

        let entry = ladder.check_reflex("greeting");
        assert!(entry.is_some());
        let entry = entry.unwrap();
        assert_eq!(entry.response, "hello!");
        assert!((entry.confidence - 0.95).abs() < f64::EPSILON);
        assert_eq!(entry.source_pattern_id.as_deref(), Some("p-1"));
    }

    #[tokio::test]
    async fn test_check_reflex_missing_returns_none() {
        let db = setup_test_db().await;
        let ladder = RoutingLadder::new(db, LadderConfig::default()).await.unwrap();

        assert!(ladder.check_reflex("nonexistent").is_none());
    }

    #[tokio::test]
    async fn test_check_reflex_updates_hit_count() {
        let db = setup_test_db().await;
        let ladder = RoutingLadder::new(db, LadderConfig::default()).await.unwrap();

        ladder
            .store_reflex("q", "a", 0.99, None)
            .await
            .unwrap();

        // First check bumps from 0 to 1
        let e1 = ladder.check_reflex("q").unwrap();
        assert_eq!(e1.hit_count, 1);

        // Second check bumps to 2
        let e2 = ladder.check_reflex("q").unwrap();
        assert_eq!(e2.hit_count, 2);
    }

    // ---- record_latency tests (2) ----------------------------------------

    #[tokio::test]
    async fn test_record_latency() {
        let db = setup_test_db().await;
        let ladder = RoutingLadder::new(db, LadderConfig::default()).await.unwrap();

        ladder
            .record_latency(ComputeLane::Heavy, 120, 100)
            .await
            .unwrap();

        let records = ladder.latency_history.read();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].lane, ComputeLane::Heavy);
        assert_eq!(records[0].actual_latency_ms, 120);
    }

    #[tokio::test]
    async fn test_avg_latency_computed() {
        let db = setup_test_db().await;
        let ladder = RoutingLadder::new(db, LadderConfig::default()).await.unwrap();

        ladder
            .record_latency(ComputeLane::Retrieval, 8, 10)
            .await
            .unwrap();
        ladder
            .record_latency(ComputeLane::Retrieval, 12, 10)
            .await
            .unwrap();

        let avg = ladder.avg_latency(ComputeLane::Retrieval);
        assert!((avg - 10.0).abs() < f64::EPSILON);

        // No data for a lane => 0.0
        assert!((ladder.avg_latency(ComputeLane::Human) - 0.0).abs() < f64::EPSILON);
    }

    // ---- promote_to_reflex tests (3) -------------------------------------

    #[tokio::test]
    async fn test_promote_successful() {
        let db = setup_test_db().await;
        let ladder = RoutingLadder::new(db, LadderConfig::default()).await.unwrap();

        let result = ladder
            .promote_to_reflex("pat-1", "some query", "some answer")
            .await
            .unwrap();

        assert!(result.promoted);
        assert_eq!(result.pattern_id, "pat-1");
        assert!(ladder.check_reflex("some query").is_some());
    }

    #[tokio::test]
    async fn test_promote_already_in_cache() {
        let db = setup_test_db().await;
        let ladder = RoutingLadder::new(db, LadderConfig::default()).await.unwrap();

        ladder
            .store_reflex("dup query", "existing", 0.99, None)
            .await
            .unwrap();

        let result = ladder
            .promote_to_reflex("pat-2", "dup query", "new answer")
            .await
            .unwrap();

        assert!(!result.promoted);
        assert!(result.reason.contains("Already in reflex cache"));
    }

    #[tokio::test]
    async fn test_promotion_result_fields() {
        let db = setup_test_db().await;
        let ladder = RoutingLadder::new(db, LadderConfig::default()).await.unwrap();

        let result = ladder
            .promote_to_reflex("p-99", "my query", "my answer")
            .await
            .unwrap();

        assert_eq!(result.pattern_id, "p-99");
        assert_eq!(result.query_hash, query_hash("my query"));
        assert!(result.promoted);
        assert!(!result.reason.is_empty());
    }

    // ---- record_retrieval_hit tests (3) ----------------------------------

    #[tokio::test]
    async fn test_retrieval_hit_tracking() {
        let db = setup_test_db().await;
        let ladder = RoutingLadder::new(db, LadderConfig::default()).await.unwrap();

        let result = ladder
            .record_retrieval_hit("foo query", "p-1")
            .await
            .unwrap();

        // After 1 hit, no promotion
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_retrieval_hit_auto_promotes_after_threshold() {
        let db = setup_test_db().await;
        let config = LadderConfig {
            promotion_after_hits: 3,
            ..LadderConfig::default()
        };
        let ladder = RoutingLadder::new(db, config).await.unwrap();

        // Hit 1 and 2 -> no promotion
        ladder.record_retrieval_hit("q", "p-1").await.unwrap();
        let r2 = ladder.record_retrieval_hit("q", "p-1").await.unwrap();
        assert!(r2.is_none());

        // Hit 3 -> promotion
        let r3 = ladder.record_retrieval_hit("q", "p-1").await.unwrap();
        assert!(r3.is_some());
        let promo = r3.unwrap();
        assert!(promo.promoted);
        assert_eq!(promo.pattern_id, "p-1");

        // Reflex cache should now contain the entry
        assert!(ladder.check_reflex("q").is_some());
    }

    #[tokio::test]
    async fn test_retrieval_hit_no_promotion_below_threshold() {
        let db = setup_test_db().await;
        let config = LadderConfig {
            promotion_after_hits: 10,
            ..LadderConfig::default()
        };
        let ladder = RoutingLadder::new(db, config).await.unwrap();

        for _ in 0..9 {
            let result = ladder.record_retrieval_hit("q", "p-1").await.unwrap();
            assert!(result.is_none());
        }
    }

    // ---- expire_stale_reflexes tests (2) ---------------------------------

    #[tokio::test]
    async fn test_expire_stale_removes_expired() {
        let db = setup_test_db().await;
        let config = LadderConfig {
            max_reflex_age_secs: 1, // 1 second
            ..LadderConfig::default()
        };
        let ladder = RoutingLadder::new(db, config).await.unwrap();

        // Insert entry with old timestamp
        let old_time = Utc::now() - chrono::Duration::seconds(10);
        {
            let mut cache = ladder.reflex_cache.write();
            cache.insert(
                "old_hash".to_string(),
                ReflexEntry {
                    query_hash: "old_hash".to_string(),
                    response: "old".to_string(),
                    confidence: 0.99,
                    hit_count: 0,
                    created_at: old_time,
                    last_hit: old_time,
                    source_pattern_id: None,
                },
            );
        }

        // Also persist to DB so the delete query executes
        ladder
            .db
            .execute(
                "INSERT INTO routing_reflex_cache (query_hash, response, confidence, hit_count, created_at, last_hit)
                 VALUES (?1, ?2, ?3, 0, ?4, ?4)",
                &[
                    &"old_hash" as &dyn rusqlite::ToSql,
                    &"old" as &dyn rusqlite::ToSql,
                    &0.99 as &dyn rusqlite::ToSql,
                    &old_time.to_rfc3339() as &dyn rusqlite::ToSql,
                ],
            )
            .await
            .unwrap();

        let removed = ladder.expire_stale_reflexes().await.unwrap();
        assert_eq!(removed, 1);
        assert!(ladder.check_reflex("anything_with_hash_old_hash").is_none());
        assert_eq!(ladder.reflex_cache_size(), 0);
    }

    #[tokio::test]
    async fn test_expire_stale_keeps_fresh() {
        let db = setup_test_db().await;
        let config = LadderConfig {
            max_reflex_age_secs: 86400,
            ..LadderConfig::default()
        };
        let ladder = RoutingLadder::new(db, config).await.unwrap();

        ladder
            .store_reflex("fresh query", "fresh", 0.99, None)
            .await
            .unwrap();

        let removed = ladder.expire_stale_reflexes().await.unwrap();
        assert_eq!(removed, 0);
        assert_eq!(ladder.reflex_cache_size(), 1);
    }

    // ---- stats tests (2) -------------------------------------------------

    #[tokio::test]
    async fn test_stats_empty() {
        let db = setup_test_db().await;
        let ladder = RoutingLadder::new(db, LadderConfig::default()).await.unwrap();

        let stats = ladder.stats().await.unwrap();
        assert_eq!(stats.total_requests, 0);
        assert_eq!(stats.cache_size, 0);
        assert!((stats.reflex_hit_rate - 0.0).abs() < f64::EPSILON);
    }

    #[tokio::test]
    async fn test_stats_populated() {
        let db = setup_test_db().await;
        let ladder = RoutingLadder::new(db, LadderConfig::default()).await.unwrap();

        ladder
            .record_latency(ComputeLane::Retrieval, 5, 10)
            .await
            .unwrap();
        ladder
            .record_latency(ComputeLane::Heavy, 200, 300)
            .await
            .unwrap();
        ladder
            .store_reflex("q1", "a1", 0.99, None)
            .await
            .unwrap();

        let stats = ladder.stats().await.unwrap();
        assert_eq!(stats.total_requests, 2); // 1 retrieval + 1 heavy
        assert_eq!(stats.retrieval_count, 1);
        assert_eq!(stats.heavy_count, 1);
        assert_eq!(stats.cache_size, 1);
    }

    // ---- clear_reflex_cache tests (1) ------------------------------------

    #[tokio::test]
    async fn test_clear_reflex_cache() {
        let db = setup_test_db().await;
        let ladder = RoutingLadder::new(db, LadderConfig::default()).await.unwrap();

        ladder.store_reflex("q1", "a1", 0.99, None).await.unwrap();
        ladder.store_reflex("q2", "a2", 0.98, None).await.unwrap();
        assert_eq!(ladder.reflex_cache_size(), 2);

        ladder.clear_reflex_cache().await.unwrap();
        assert_eq!(ladder.reflex_cache_size(), 0);
    }

    // ---- reflex_cache_size tests (1) -------------------------------------

    #[tokio::test]
    async fn test_reflex_cache_size_correct() {
        let db = setup_test_db().await;
        let ladder = RoutingLadder::new(db, LadderConfig::default()).await.unwrap();

        assert_eq!(ladder.reflex_cache_size(), 0);
        ladder.store_reflex("q1", "a1", 0.99, None).await.unwrap();
        assert_eq!(ladder.reflex_cache_size(), 1);
        ladder.store_reflex("q2", "a2", 0.98, None).await.unwrap();
        assert_eq!(ladder.reflex_cache_size(), 2);
    }

    // ---- LadderDecision tests (2) ----------------------------------------

    #[tokio::test]
    async fn test_ladder_decision_correct_fields() {
        let db = setup_test_db().await;
        let ladder = RoutingLadder::new(db, LadderConfig::default()).await.unwrap();

        let decision = ladder.route("test", 0.5).unwrap();
        assert_eq!(decision.lane, ComputeLane::Heavy);
        assert!(!decision.query_hash.is_empty());
        assert_eq!(decision.query_hash, query_hash("test"));
        assert!(!decision.reflex_hit);
    }

    #[tokio::test]
    async fn test_ladder_decision_reasoning_populated() {
        let db = setup_test_db().await;
        let ladder = RoutingLadder::new(db, LadderConfig::default()).await.unwrap();

        let decision = ladder.route("anything", 0.1).unwrap();
        assert!(!decision.reasoning.is_empty());
        assert!(decision.reasoning.contains("Complexity"));
    }

    // ---- edge case tests (2) ---------------------------------------------

    #[tokio::test]
    async fn test_route_empty_query() {
        let db = setup_test_db().await;
        let ladder = RoutingLadder::new(db, LadderConfig::default()).await.unwrap();

        // Empty query should still route successfully
        let decision = ladder.route("", 0.1).unwrap();
        assert_eq!(decision.lane, ComputeLane::Retrieval);
        assert!(!decision.query_hash.is_empty());
    }

    #[tokio::test]
    async fn test_route_very_high_complexity() {
        let db = setup_test_db().await;
        let ladder = RoutingLadder::new(db, LadderConfig::default()).await.unwrap();

        let decision = ladder.route("impossible task", 1.0).unwrap();
        assert_eq!(decision.lane, ComputeLane::Human);
        assert!((decision.confidence - 1.0).abs() < f64::EPSILON);
    }
}
