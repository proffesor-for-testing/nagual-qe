//! Temperature-based pattern tiering for the Knowledge Operating System (KOS P7).
//!
//! Classifies patterns into hot/warm/cold tiers based on access frequency and recency.
//! Hot patterns are kept in an LRU cache for fast retrieval. Periodic reclassification
//! scans the full tier table and promotes or demotes patterns as their access characteristics
//! change.
//!
//! # Tier Classification
//!
//! | Tier | Criteria |
//! |------|----------|
//! | Hot  | Accessed within `hot_threshold_days` AND access_count >= `min_access_count_for_hot` |
//! | Warm | Accessed within `warm_threshold_days` but not qualifying for Hot |
//! | Cold | Not accessed within `warm_threshold_days` |
//!
//! # Example
//!
//! ```ignore
//! use nagual::tiering::{TieringManager, TieringConfig};
//! use nagual::db::SqliteDb;
//! use std::sync::Arc;
//!
//! let db = Arc::new(SqliteDb::open_in_memory().unwrap());
//! let manager = TieringManager::new(db, TieringConfig::default()).await.unwrap();
//!
//! // Record an access (creates record if first access)
//! let record = manager.record_access("pattern-123").await.unwrap();
//! println!("Tier: {}", record.tier);
//!
//! // Reclassify all patterns
//! let result = manager.reclassify_all().await.unwrap();
//! println!("Promoted: {}, Demoted: {}", result.promoted.len(), result.demoted.len());
//! ```

use chrono::{DateTime, Utc};
use lru::LruCache;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::num::NonZeroUsize;
use std::sync::Arc;

use crate::db::SqliteDb;
use crate::error::Result;

// ---------------------------------------------------------------------------
// TemperatureTier
// ---------------------------------------------------------------------------

/// Temperature classification for a pattern based on access frequency and recency.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TemperatureTier {
    /// Frequently accessed within the hot threshold window.
    Hot,
    /// Moderately accessed -- not hot but not stale.
    Warm,
    /// Rarely or never accessed beyond the warm threshold window.
    Cold,
}

impl TemperatureTier {
    /// Return the tier as a lowercase string slice.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Hot => "hot",
            Self::Warm => "warm",
            Self::Cold => "cold",
        }
    }
}

impl std::fmt::Display for TemperatureTier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl From<&str> for TemperatureTier {
    fn from(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "hot" => Self::Hot,
            "warm" => Self::Warm,
            _ => Self::Cold,
        }
    }
}

// ---------------------------------------------------------------------------
// TieringConfig
// ---------------------------------------------------------------------------

/// Configuration for the temperature tiering system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TieringConfig {
    /// Number of days within which an access is considered "hot".
    pub hot_threshold_days: u32,
    /// Number of days within which an access is considered "warm".
    pub warm_threshold_days: u32,
    /// Number of days after which a pattern is considered "cold" (informational).
    pub cold_threshold_days: u32,
    /// Maximum number of hot-tier entries kept in the in-memory LRU cache.
    pub hot_cache_size: usize,
    /// How often automatic reclassification runs, in seconds.
    pub reclassification_interval_secs: u64,
    /// Minimum number of accesses required for a pattern to be classified as Hot,
    /// even if the recency criterion is met.
    pub min_access_count_for_hot: u32,
}

impl Default for TieringConfig {
    fn default() -> Self {
        Self {
            hot_threshold_days: 7,
            warm_threshold_days: 30,
            cold_threshold_days: 90,
            hot_cache_size: 500,
            reclassification_interval_secs: 3600,
            min_access_count_for_hot: 3,
        }
    }
}

// ---------------------------------------------------------------------------
// AccessRecord
// ---------------------------------------------------------------------------

/// Tracks per-pattern access metadata and tier placement.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccessRecord {
    /// Pattern identifier.
    pub pattern_id: String,
    /// Total number of accesses recorded.
    pub access_count: u64,
    /// Timestamp of the most recent access.
    pub last_accessed: DateTime<Utc>,
    /// Current temperature tier.
    pub tier: TemperatureTier,
    /// When the pattern was last promoted (moved to a hotter tier).
    pub promoted_at: Option<DateTime<Utc>>,
    /// When the pattern was last demoted (moved to a colder tier).
    pub demoted_at: Option<DateTime<Utc>>,
}

// ---------------------------------------------------------------------------
// TieringStats
// ---------------------------------------------------------------------------

/// Aggregate statistics for the tiering system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TieringStats {
    /// Number of patterns in the hot tier.
    pub hot_count: u64,
    /// Number of patterns in the warm tier.
    pub warm_count: u64,
    /// Number of patterns in the cold tier.
    pub cold_count: u64,
    /// Sum of all access_count values.
    pub total_accesses: u64,
    /// Mean access count per pattern (0.0 when no patterns exist).
    pub avg_access_frequency: f64,
    /// When the most recent reclassification cycle ran.
    pub last_reclassification: Option<DateTime<Utc>>,
}

// ---------------------------------------------------------------------------
// ReclassificationResult
// ---------------------------------------------------------------------------

/// Summary of a reclassification sweep.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReclassificationResult {
    /// Patterns that moved to a hotter tier: `(id, from_tier, to_tier)`.
    pub promoted: Vec<(String, TemperatureTier, TemperatureTier)>,
    /// Patterns that moved to a colder tier: `(id, from_tier, to_tier)`.
    pub demoted: Vec<(String, TemperatureTier, TemperatureTier)>,
    /// Number of patterns whose tier did not change.
    pub unchanged: u64,
}

// ---------------------------------------------------------------------------
// TieringManager
// ---------------------------------------------------------------------------

/// Manages temperature-based tiering for patterns.
///
/// Persists tier data in a `pattern_tiers` SQLite table and maintains an
/// in-memory LRU cache of hot-tier records for fast lookups.
pub struct TieringManager {
    db: Arc<SqliteDb>,
    config: TieringConfig,
    hot_cache: Arc<RwLock<LruCache<String, AccessRecord>>>,
}

impl TieringManager {
    /// Create a new `TieringManager`, creating the backing table and indexes
    /// if they do not already exist, then pre-loading hot patterns into the cache.
    pub async fn new(db: Arc<SqliteDb>, config: TieringConfig) -> Result<Self> {
        // Ensure the schema is present.
        db.execute_batch(
            "CREATE TABLE IF NOT EXISTS pattern_tiers (
                pattern_id TEXT PRIMARY KEY,
                access_count INTEGER DEFAULT 0,
                last_accessed TEXT NOT NULL,
                tier TEXT NOT NULL DEFAULT 'cold',
                promoted_at TEXT,
                demoted_at TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_tiers_tier ON pattern_tiers(tier);
            CREATE INDEX IF NOT EXISTS idx_tiers_last_accessed ON pattern_tiers(last_accessed);
            CREATE INDEX IF NOT EXISTS idx_tiers_access_count ON pattern_tiers(access_count DESC);",
        )
        .await?;

        let capacity = NonZeroUsize::new(config.hot_cache_size.max(1))
            .expect("hot_cache_size clamped to >= 1");
        let cache = LruCache::new(capacity);
        let manager = Self {
            db,
            config,
            hot_cache: Arc::new(RwLock::new(cache)),
        };

        // Seed the cache with existing hot patterns.
        manager.reload_hot_cache().await?;

        Ok(manager)
    }

    // -- public API ---------------------------------------------------------

    /// Record an access for `pattern_id`. If no record exists, one is created
    /// with `access_count = 1` and an initial tier classification. If the
    /// record already exists, `access_count` is incremented, `last_accessed`
    /// is updated, and the tier is reclassified.
    pub async fn record_access(&self, pattern_id: &str) -> Result<AccessRecord> {
        let now = Utc::now();
        let existing = self.get_access_record(pattern_id).await?;

        let record = match existing {
            Some(mut rec) => {
                rec.access_count += 1;
                rec.last_accessed = now;
                let new_tier = self.classify_tier(rec.access_count, rec.last_accessed);
                if new_tier != rec.tier {
                    if tier_ordinal(new_tier) > tier_ordinal(rec.tier) {
                        rec.promoted_at = Some(now);
                    } else {
                        rec.demoted_at = Some(now);
                    }
                    rec.tier = new_tier;
                }
                self.upsert_record(&rec).await?;
                rec
            }
            None => {
                let tier = self.classify_tier(1, now);
                let rec = AccessRecord {
                    pattern_id: pattern_id.to_string(),
                    access_count: 1,
                    last_accessed: now,
                    tier,
                    promoted_at: None,
                    demoted_at: None,
                };
                self.upsert_record(&rec).await?;
                rec
            }
        };

        // Keep the hot cache up to date.
        if record.tier == TemperatureTier::Hot {
            self.hot_cache
                .write()
                .put(record.pattern_id.clone(), record.clone());
        } else {
            self.hot_cache.write().pop(&record.pattern_id);
        }

        Ok(record)
    }

    /// Return the current tier for a pattern, defaulting to `Cold` if no record exists.
    pub async fn get_tier(&self, pattern_id: &str) -> Result<TemperatureTier> {
        // Fast path: check the hot cache first.
        if let Some(rec) = self.hot_cache.read().peek(pattern_id) {
            return Ok(rec.tier);
        }

        match self.get_access_record(pattern_id).await? {
            Some(rec) => Ok(rec.tier),
            None => Ok(TemperatureTier::Cold),
        }
    }

    /// Return the full access record for a pattern, if one exists.
    pub async fn get_access_record(&self, pattern_id: &str) -> Result<Option<AccessRecord>> {
        self.db
            .query_one(
                "SELECT pattern_id, access_count, last_accessed, tier, promoted_at, demoted_at
                 FROM pattern_tiers WHERE pattern_id = ?",
                &[&pattern_id],
                row_to_access_record,
            )
            .await
    }

    /// Pure function: classify a tier given access count and last-accessed timestamp.
    pub fn classify_tier(
        &self,
        access_count: u64,
        last_accessed: DateTime<Utc>,
    ) -> TemperatureTier {
        let age_days = (Utc::now() - last_accessed).num_days().max(0) as u32;

        if age_days <= self.config.hot_threshold_days
            && access_count >= self.config.min_access_count_for_hot as u64
        {
            TemperatureTier::Hot
        } else if age_days <= self.config.warm_threshold_days {
            TemperatureTier::Warm
        } else {
            TemperatureTier::Cold
        }
    }

    /// Scan every record in `pattern_tiers` and reclassify.
    /// Returns a summary of promotions, demotions, and unchanged counts.
    pub async fn reclassify_all(&self) -> Result<ReclassificationResult> {
        let all_records: Vec<AccessRecord> = self
            .db
            .query(
                "SELECT pattern_id, access_count, last_accessed, tier, promoted_at, demoted_at
                 FROM pattern_tiers",
                &[],
                row_to_access_record,
            )
            .await?;

        let now = Utc::now();
        let mut promoted = Vec::new();
        let mut demoted = Vec::new();
        let mut unchanged: u64 = 0;

        for rec in &all_records {
            let new_tier = self.classify_tier(rec.access_count, rec.last_accessed);
            if new_tier == rec.tier {
                unchanged += 1;
                continue;
            }

            let is_promotion = tier_ordinal(new_tier) > tier_ordinal(rec.tier);
            let promoted_at = if is_promotion { Some(now) } else { rec.promoted_at };
            let demoted_at = if !is_promotion { Some(now) } else { rec.demoted_at };

            self.db
                .execute(
                    "UPDATE pattern_tiers SET tier = ?, promoted_at = ?, demoted_at = ?
                     WHERE pattern_id = ?",
                    &[
                        &new_tier.as_str(),
                        &promoted_at.map(|d| d.to_rfc3339()),
                        &demoted_at.map(|d| d.to_rfc3339()),
                        &rec.pattern_id,
                    ],
                )
                .await?;

            if is_promotion {
                promoted.push((rec.pattern_id.clone(), rec.tier, new_tier));
            } else {
                demoted.push((rec.pattern_id.clone(), rec.tier, new_tier));
            }
        }

        // Rebuild the hot cache after a full reclassification.
        self.reload_hot_cache().await?;

        Ok(ReclassificationResult {
            promoted,
            demoted,
            unchanged,
        })
    }

    /// Return the hottest patterns, up to `limit`. Prefers the in-memory cache
    /// but falls back to the database.
    pub async fn get_hot_patterns(&self, limit: usize) -> Result<Vec<AccessRecord>> {
        self.patterns_by_tier(TemperatureTier::Hot, limit).await
    }

    /// Return the coldest patterns, suitable for archival, up to `limit`.
    pub async fn get_cold_patterns(&self, limit: usize) -> Result<Vec<AccessRecord>> {
        self.patterns_by_tier(TemperatureTier::Cold, limit).await
    }

    /// Aggregate statistics across all tiers.
    pub async fn stats(&self) -> Result<TieringStats> {
        let counts: Vec<(String, u64)> = self
            .db
            .query(
                "SELECT tier, COUNT(*) FROM pattern_tiers GROUP BY tier",
                &[],
                |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, u64>(1)?,
                    ))
                },
            )
            .await?;

        let mut hot_count: u64 = 0;
        let mut warm_count: u64 = 0;
        let mut cold_count: u64 = 0;
        for (tier_str, cnt) in &counts {
            match tier_str.as_str() {
                "hot" => hot_count = *cnt,
                "warm" => warm_count = *cnt,
                _ => cold_count = *cnt,
            }
        }

        let totals = self
            .db
            .query_one(
                "SELECT COALESCE(SUM(access_count), 0), COALESCE(AVG(access_count), 0.0)
                 FROM pattern_tiers",
                &[],
                |row| {
                    Ok((
                        row.get::<_, u64>(0)?,
                        row.get::<_, f64>(1)?,
                    ))
                },
            )
            .await?;

        let (total_accesses, avg_access_frequency) = totals.unwrap_or((0, 0.0));

        Ok(TieringStats {
            hot_count,
            warm_count,
            cold_count,
            total_accesses,
            avg_access_frequency,
            last_reclassification: None,
        })
    }

    /// Manually promote a pattern to `target` tier, regardless of classification rules.
    pub async fn promote(&self, pattern_id: &str, target: TemperatureTier) -> Result<()> {
        let now = Utc::now();
        let existing = self.get_access_record(pattern_id).await?;
        match existing {
            Some(mut rec) => {
                rec.tier = target;
                rec.promoted_at = Some(now);
                self.upsert_record(&rec).await?;
                self.sync_cache(&rec);
            }
            None => {
                let rec = AccessRecord {
                    pattern_id: pattern_id.to_string(),
                    access_count: 0,
                    last_accessed: now,
                    tier: target,
                    promoted_at: Some(now),
                    demoted_at: None,
                };
                self.upsert_record(&rec).await?;
                self.sync_cache(&rec);
            }
        }
        Ok(())
    }

    /// Manually demote a pattern to `target` tier, regardless of classification rules.
    pub async fn demote(&self, pattern_id: &str, target: TemperatureTier) -> Result<()> {
        let now = Utc::now();
        let existing = self.get_access_record(pattern_id).await?;
        match existing {
            Some(mut rec) => {
                rec.tier = target;
                rec.demoted_at = Some(now);
                self.upsert_record(&rec).await?;
                self.sync_cache(&rec);
            }
            None => {
                let rec = AccessRecord {
                    pattern_id: pattern_id.to_string(),
                    access_count: 0,
                    last_accessed: now,
                    tier: target,
                    promoted_at: None,
                    demoted_at: Some(now),
                };
                self.upsert_record(&rec).await?;
                self.sync_cache(&rec);
            }
        }
        Ok(())
    }

    /// Query patterns belonging to a specific tier, ordered by access_count descending,
    /// up to `limit`.
    pub async fn patterns_by_tier(
        &self,
        tier: TemperatureTier,
        limit: usize,
    ) -> Result<Vec<AccessRecord>> {
        self.db
            .query(
                "SELECT pattern_id, access_count, last_accessed, tier, promoted_at, demoted_at
                 FROM pattern_tiers WHERE tier = ? ORDER BY access_count DESC LIMIT ?",
                &[&tier.as_str(), &(limit as i64)],
                row_to_access_record,
            )
            .await
    }

    // -- private helpers ----------------------------------------------------

    /// Persist an access record via INSERT OR REPLACE.
    async fn upsert_record(&self, rec: &AccessRecord) -> Result<()> {
        self.db
            .execute(
                "INSERT OR REPLACE INTO pattern_tiers
                    (pattern_id, access_count, last_accessed, tier, promoted_at, demoted_at, created_at)
                 VALUES (?, ?, ?, ?, ?, ?, COALESCE(
                    (SELECT created_at FROM pattern_tiers WHERE pattern_id = ?),
                    datetime('now')))",
                &[
                    &rec.pattern_id,
                    &(rec.access_count as i64),
                    &rec.last_accessed.to_rfc3339(),
                    &rec.tier.as_str(),
                    &rec.promoted_at.map(|d| d.to_rfc3339()),
                    &rec.demoted_at.map(|d| d.to_rfc3339()),
                    &rec.pattern_id,
                ],
            )
            .await?;
        Ok(())
    }

    /// Reload the hot cache from the database.
    async fn reload_hot_cache(&self) -> Result<()> {
        let hots = self
            .db
            .query(
                "SELECT pattern_id, access_count, last_accessed, tier, promoted_at, demoted_at
                 FROM pattern_tiers WHERE tier = 'hot'
                 ORDER BY access_count DESC
                 LIMIT ?",
                &[&(self.config.hot_cache_size as i64)],
                row_to_access_record,
            )
            .await?;

        let mut cache = self.hot_cache.write();
        cache.clear();
        for rec in hots {
            cache.put(rec.pattern_id.clone(), rec);
        }
        Ok(())
    }

    /// Push or remove a record from the hot cache depending on its tier.
    fn sync_cache(&self, rec: &AccessRecord) {
        if rec.tier == TemperatureTier::Hot {
            self.hot_cache
                .write()
                .put(rec.pattern_id.clone(), rec.clone());
        } else {
            self.hot_cache.write().pop(&rec.pattern_id);
        }
    }
}

// ---------------------------------------------------------------------------
// Free helpers
// ---------------------------------------------------------------------------

/// Map a rusqlite row to an `AccessRecord`.
fn row_to_access_record(row: &rusqlite::Row<'_>) -> rusqlite::Result<AccessRecord> {
    let last_accessed_str: String = row.get(2)?;
    let tier_str: String = row.get(3)?;
    let promoted_str: Option<String> = row.get(4)?;
    let demoted_str: Option<String> = row.get(5)?;

    let last_accessed = DateTime::parse_from_rfc3339(&last_accessed_str)
        .map(|dt| dt.with_timezone(&Utc))
        .unwrap_or_else(|_| Utc::now());
    let promoted_at = promoted_str.and_then(|s| {
        DateTime::parse_from_rfc3339(&s)
            .map(|dt| dt.with_timezone(&Utc))
            .ok()
    });
    let demoted_at = demoted_str.and_then(|s| {
        DateTime::parse_from_rfc3339(&s)
            .map(|dt| dt.with_timezone(&Utc))
            .ok()
    });

    Ok(AccessRecord {
        pattern_id: row.get(0)?,
        access_count: row.get::<_, i64>(1)? as u64,
        last_accessed,
        tier: TemperatureTier::from(tier_str.as_str()),
        promoted_at,
        demoted_at,
    })
}

/// Ordinal ranking for tier comparison: Hot > Warm > Cold.
fn tier_ordinal(tier: TemperatureTier) -> u8 {
    match tier {
        TemperatureTier::Hot => 2,
        TemperatureTier::Warm => 1,
        TemperatureTier::Cold => 0,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Duration;

    async fn setup_test_db() -> Arc<SqliteDb> {
        let db = SqliteDb::open_in_memory().unwrap();
        Arc::new(db)
    }

    async fn setup_manager() -> TieringManager {
        let db = setup_test_db().await;
        TieringManager::new(db, TieringConfig::default()).await.unwrap()
    }

    async fn setup_manager_with_config(config: TieringConfig) -> TieringManager {
        let db = setup_test_db().await;
        TieringManager::new(db, config).await.unwrap()
    }

    // -- LruCache integration tests (using lru crate) -----------------------

    #[test]
    fn test_lru_cache_new() {
        let cache: LruCache<String, i32> = LruCache::new(NonZeroUsize::new(10).unwrap());
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn test_lru_cache_insert_and_get() {
        let mut cache: LruCache<String, i32> = LruCache::new(NonZeroUsize::new(10).unwrap());
        cache.put(String::from("a"), 1);
        cache.put(String::from("b"), 2);
        assert_eq!(*cache.get(&String::from("a")).unwrap(), 1);
        assert_eq!(*cache.get(&String::from("b")).unwrap(), 2);
    }

    #[test]
    fn test_lru_cache_get_nonexistent() {
        let mut cache: LruCache<String, i32> = LruCache::new(NonZeroUsize::new(10).unwrap());
        assert!(cache.get(&String::from("missing")).is_none());
    }

    #[test]
    fn test_lru_cache_eviction() {
        let mut cache: LruCache<String, i32> = LruCache::new(NonZeroUsize::new(2).unwrap());
        cache.put(String::from("a"), 1);
        cache.put(String::from("b"), 2);
        cache.put(String::from("c"), 3); // evicts "a"
        assert!(cache.get(&String::from("a")).is_none());
        assert_eq!(*cache.get(&String::from("b")).unwrap(), 2);
        assert_eq!(*cache.get(&String::from("c")).unwrap(), 3);
    }

    #[test]
    fn test_lru_cache_contains_key() {
        let mut cache: LruCache<String, i32> = LruCache::new(NonZeroUsize::new(10).unwrap());
        cache.put(String::from("key"), 42);
        assert!(cache.contains(&String::from("key")));
        assert!(!cache.contains(&String::from("other")));
    }

    // -- TemperatureTier tests -----------------------------------------------

    #[test]
    fn test_temperature_tier_display() {
        assert_eq!(format!("{}", TemperatureTier::Hot), "hot");
        assert_eq!(format!("{}", TemperatureTier::Warm), "warm");
        assert_eq!(format!("{}", TemperatureTier::Cold), "cold");
    }

    #[test]
    fn test_temperature_tier_from_str() {
        assert_eq!(TemperatureTier::from("hot"), TemperatureTier::Hot);
        assert_eq!(TemperatureTier::from("HOT"), TemperatureTier::Hot);
        assert_eq!(TemperatureTier::from("warm"), TemperatureTier::Warm);
        assert_eq!(TemperatureTier::from("Warm"), TemperatureTier::Warm);
        assert_eq!(TemperatureTier::from("cold"), TemperatureTier::Cold);
        assert_eq!(TemperatureTier::from("anything"), TemperatureTier::Cold);
    }

    #[test]
    fn test_temperature_tier_as_str() {
        assert_eq!(TemperatureTier::Hot.as_str(), "hot");
        assert_eq!(TemperatureTier::Warm.as_str(), "warm");
        assert_eq!(TemperatureTier::Cold.as_str(), "cold");
    }

    // -- TieringConfig tests -------------------------------------------------

    #[test]
    fn test_tiering_config_defaults() {
        let config = TieringConfig::default();
        assert_eq!(config.hot_threshold_days, 7);
        assert_eq!(config.warm_threshold_days, 30);
        assert_eq!(config.cold_threshold_days, 90);
        assert_eq!(config.hot_cache_size, 500);
        assert_eq!(config.reclassification_interval_secs, 3600);
        assert_eq!(config.min_access_count_for_hot, 3);
    }

    #[test]
    fn test_tiering_config_custom() {
        let config = TieringConfig {
            hot_threshold_days: 3,
            warm_threshold_days: 14,
            cold_threshold_days: 60,
            hot_cache_size: 100,
            reclassification_interval_secs: 1800,
            min_access_count_for_hot: 5,
        };
        assert_eq!(config.hot_threshold_days, 3);
        assert_eq!(config.min_access_count_for_hot, 5);
    }

    // -- classify_tier tests -------------------------------------------------

    #[tokio::test]
    async fn test_classify_tier_hot() {
        let config = TieringConfig {
            min_access_count_for_hot: 3,
            hot_threshold_days: 7,
            ..TieringConfig::default()
        };
        let manager = setup_manager_with_config(config).await;
        let now = Utc::now();
        assert_eq!(manager.classify_tier(5, now), TemperatureTier::Hot);
    }

    #[tokio::test]
    async fn test_classify_tier_warm() {
        let manager = setup_manager().await;
        // Recent but low access count -> warm (doesn't meet min_access_count_for_hot)
        let now = Utc::now();
        assert_eq!(manager.classify_tier(1, now), TemperatureTier::Warm);

        // Within warm window but outside hot window
        let fifteen_days_ago = Utc::now() - Duration::days(15);
        assert_eq!(
            manager.classify_tier(10, fifteen_days_ago),
            TemperatureTier::Warm
        );
    }

    #[tokio::test]
    async fn test_classify_tier_cold() {
        let manager = setup_manager().await;
        let sixty_days_ago = Utc::now() - Duration::days(60);
        assert_eq!(
            manager.classify_tier(100, sixty_days_ago),
            TemperatureTier::Cold
        );
    }

    // -- record_access tests -------------------------------------------------

    #[tokio::test]
    async fn test_record_access_creates_record() {
        let manager = setup_manager().await;
        let rec = manager.record_access("p-1").await.unwrap();
        assert_eq!(rec.pattern_id, "p-1");
        assert_eq!(rec.access_count, 1);
        // With 1 access and min_access_count_for_hot = 3, first access is Warm
        assert_eq!(rec.tier, TemperatureTier::Warm);
    }

    #[tokio::test]
    async fn test_record_access_increments() {
        let manager = setup_manager().await;
        manager.record_access("p-2").await.unwrap();
        manager.record_access("p-2").await.unwrap();
        let rec = manager.record_access("p-2").await.unwrap();
        assert_eq!(rec.access_count, 3);
    }

    #[tokio::test]
    async fn test_record_access_tier_promotion() {
        let config = TieringConfig {
            min_access_count_for_hot: 3,
            ..TieringConfig::default()
        };
        let manager = setup_manager_with_config(config).await;

        // First two accesses -> Warm
        manager.record_access("p-promo").await.unwrap();
        let rec = manager.record_access("p-promo").await.unwrap();
        assert_eq!(rec.tier, TemperatureTier::Warm);

        // Third access -> Hot (meets min_access_count_for_hot)
        let rec = manager.record_access("p-promo").await.unwrap();
        assert_eq!(rec.tier, TemperatureTier::Hot);
        assert!(rec.promoted_at.is_some());
    }

    // -- reclassify_all tests ------------------------------------------------

    #[tokio::test]
    async fn test_reclassify_promotes_cold_to_hot() {
        let config = TieringConfig {
            min_access_count_for_hot: 1,
            hot_threshold_days: 7,
            ..TieringConfig::default()
        };
        let db = setup_test_db().await;
        let manager = TieringManager::new(db.clone(), config).await.unwrap();

        // Insert a record that is marked cold but was accessed recently with enough count.
        let now = Utc::now();
        db.execute(
            "INSERT INTO pattern_tiers (pattern_id, access_count, last_accessed, tier)
             VALUES (?, ?, ?, ?)",
            &[
                &"stale-hot",
                &5_i64,
                &now.to_rfc3339(),
                &"cold",
            ],
        )
        .await
        .unwrap();

        let result = manager.reclassify_all().await.unwrap();
        assert_eq!(result.promoted.len(), 1);
        assert_eq!(result.promoted[0].0, "stale-hot");
        assert_eq!(result.promoted[0].1, TemperatureTier::Cold);
        assert_eq!(result.promoted[0].2, TemperatureTier::Hot);
    }

    #[tokio::test]
    async fn test_reclassify_demotes_hot_to_cold() {
        let db = setup_test_db().await;
        let manager = TieringManager::new(db.clone(), TieringConfig::default())
            .await
            .unwrap();

        // Insert a record marked hot but last accessed 60 days ago.
        let old = (Utc::now() - Duration::days(60)).to_rfc3339();
        db.execute(
            "INSERT INTO pattern_tiers (pattern_id, access_count, last_accessed, tier)
             VALUES (?, ?, ?, ?)",
            &[&"old-hot", &10_i64, &old, &"hot"],
        )
        .await
        .unwrap();

        let result = manager.reclassify_all().await.unwrap();
        assert_eq!(result.demoted.len(), 1);
        assert_eq!(result.demoted[0].0, "old-hot");
        assert_eq!(result.demoted[0].1, TemperatureTier::Hot);
        assert_eq!(result.demoted[0].2, TemperatureTier::Cold);
    }

    #[tokio::test]
    async fn test_reclassify_mixed() {
        let db = setup_test_db().await;
        let config = TieringConfig {
            min_access_count_for_hot: 2,
            ..TieringConfig::default()
        };
        let manager = TieringManager::new(db.clone(), config).await.unwrap();

        let now = Utc::now();
        let old = (now - Duration::days(60)).to_rfc3339();

        // Pattern A: cold but should be warm (recent access, low count)
        db.execute(
            "INSERT INTO pattern_tiers (pattern_id, access_count, last_accessed, tier)
             VALUES (?, ?, ?, ?)",
            &[&"a", &1_i64, &now.to_rfc3339(), &"cold"],
        )
        .await
        .unwrap();

        // Pattern B: hot but should be cold (old access)
        db.execute(
            "INSERT INTO pattern_tiers (pattern_id, access_count, last_accessed, tier)
             VALUES (?, ?, ?, ?)",
            &[&"b", &50_i64, &old, &"hot"],
        )
        .await
        .unwrap();

        // Pattern C: warm, stays warm (recent, low count)
        db.execute(
            "INSERT INTO pattern_tiers (pattern_id, access_count, last_accessed, tier)
             VALUES (?, ?, ?, ?)",
            &[&"c", &1_i64, &now.to_rfc3339(), &"warm"],
        )
        .await
        .unwrap();

        let result = manager.reclassify_all().await.unwrap();
        assert_eq!(result.promoted.len(), 1); // a: cold -> warm
        assert_eq!(result.demoted.len(), 1); // b: hot -> cold
        assert_eq!(result.unchanged, 1); // c: warm -> warm
    }

    // -- get_hot_patterns / get_cold_patterns tests --------------------------

    #[tokio::test]
    async fn test_get_hot_patterns() {
        let config = TieringConfig {
            min_access_count_for_hot: 1,
            ..TieringConfig::default()
        };
        let manager = setup_manager_with_config(config).await;

        // Two accesses to make it solidly hot (access_count >= 1 and recent)
        manager.record_access("h1").await.unwrap();
        manager.record_access("h2").await.unwrap();

        let hots = manager.get_hot_patterns(10).await.unwrap();
        assert_eq!(hots.len(), 2);
    }

    #[tokio::test]
    async fn test_get_cold_patterns() {
        let db = setup_test_db().await;
        let manager = TieringManager::new(db.clone(), TieringConfig::default())
            .await
            .unwrap();

        let old = (Utc::now() - Duration::days(90)).to_rfc3339();
        db.execute(
            "INSERT INTO pattern_tiers (pattern_id, access_count, last_accessed, tier)
             VALUES (?, ?, ?, ?)",
            &[&"cold-1", &1_i64, &old, &"cold"],
        )
        .await
        .unwrap();

        let colds = manager.get_cold_patterns(10).await.unwrap();
        assert_eq!(colds.len(), 1);
        assert_eq!(colds[0].pattern_id, "cold-1");
    }

    // -- promote / demote tests ----------------------------------------------

    #[tokio::test]
    async fn test_manual_promote() {
        let manager = setup_manager().await;
        manager.record_access("mp-1").await.unwrap();

        manager
            .promote("mp-1", TemperatureTier::Hot)
            .await
            .unwrap();
        let rec = manager.get_access_record("mp-1").await.unwrap().unwrap();
        assert_eq!(rec.tier, TemperatureTier::Hot);
        assert!(rec.promoted_at.is_some());
    }

    #[tokio::test]
    async fn test_manual_demote() {
        let config = TieringConfig {
            min_access_count_for_hot: 1,
            ..TieringConfig::default()
        };
        let manager = setup_manager_with_config(config).await;
        manager.record_access("md-1").await.unwrap();

        // Should now be hot (access_count >= 1, recent)
        let rec = manager.get_access_record("md-1").await.unwrap().unwrap();
        assert_eq!(rec.tier, TemperatureTier::Hot);

        manager
            .demote("md-1", TemperatureTier::Cold)
            .await
            .unwrap();
        let rec = manager.get_access_record("md-1").await.unwrap().unwrap();
        assert_eq!(rec.tier, TemperatureTier::Cold);
        assert!(rec.demoted_at.is_some());
    }

    // -- stats tests ---------------------------------------------------------

    #[tokio::test]
    async fn test_stats_empty() {
        let manager = setup_manager().await;
        let stats = manager.stats().await.unwrap();
        assert_eq!(stats.hot_count, 0);
        assert_eq!(stats.warm_count, 0);
        assert_eq!(stats.cold_count, 0);
        assert_eq!(stats.total_accesses, 0);
        assert_eq!(stats.avg_access_frequency, 0.0);
    }

    #[tokio::test]
    async fn test_stats_populated() {
        let config = TieringConfig {
            min_access_count_for_hot: 1,
            ..TieringConfig::default()
        };
        let db = setup_test_db().await;
        let manager = TieringManager::new(db.clone(), config).await.unwrap();

        // Insert records of each tier directly
        let now = Utc::now().to_rfc3339();
        let old = (Utc::now() - Duration::days(60)).to_rfc3339();

        db.execute(
            "INSERT INTO pattern_tiers (pattern_id, access_count, last_accessed, tier)
             VALUES (?, ?, ?, ?)",
            &[&"h", &5_i64, &now, &"hot"],
        )
        .await
        .unwrap();
        db.execute(
            "INSERT INTO pattern_tiers (pattern_id, access_count, last_accessed, tier)
             VALUES (?, ?, ?, ?)",
            &[&"w", &3_i64, &now, &"warm"],
        )
        .await
        .unwrap();
        db.execute(
            "INSERT INTO pattern_tiers (pattern_id, access_count, last_accessed, tier)
             VALUES (?, ?, ?, ?)",
            &[&"c", &2_i64, &old, &"cold"],
        )
        .await
        .unwrap();

        let stats = manager.stats().await.unwrap();
        assert_eq!(stats.hot_count, 1);
        assert_eq!(stats.warm_count, 1);
        assert_eq!(stats.cold_count, 1);
        assert_eq!(stats.total_accesses, 10);
        assert!((stats.avg_access_frequency - 10.0 / 3.0).abs() < 0.01);
    }

    // -- patterns_by_tier tests ----------------------------------------------

    #[tokio::test]
    async fn test_patterns_by_tier_filters() {
        let db = setup_test_db().await;
        let manager = TieringManager::new(db.clone(), TieringConfig::default())
            .await
            .unwrap();

        let now = Utc::now().to_rfc3339();
        for (id, tier) in &[("a", "hot"), ("b", "hot"), ("c", "warm"), ("d", "cold")] {
            db.execute(
                "INSERT INTO pattern_tiers (pattern_id, access_count, last_accessed, tier)
                 VALUES (?, ?, ?, ?)",
                &[id, &1_i64, &now, tier],
            )
            .await
            .unwrap();
        }

        let hots = manager
            .patterns_by_tier(TemperatureTier::Hot, 10)
            .await
            .unwrap();
        assert_eq!(hots.len(), 2);

        let warms = manager
            .patterns_by_tier(TemperatureTier::Warm, 10)
            .await
            .unwrap();
        assert_eq!(warms.len(), 1);
    }

    #[tokio::test]
    async fn test_patterns_by_tier_limit() {
        let db = setup_test_db().await;
        let manager = TieringManager::new(db.clone(), TieringConfig::default())
            .await
            .unwrap();

        let now = Utc::now().to_rfc3339();
        for i in 0..5 {
            db.execute(
                "INSERT INTO pattern_tiers (pattern_id, access_count, last_accessed, tier)
                 VALUES (?, ?, ?, ?)",
                &[&format!("w-{}", i), &1_i64, &now, &"warm"],
            )
            .await
            .unwrap();
        }

        let limited = manager
            .patterns_by_tier(TemperatureTier::Warm, 3)
            .await
            .unwrap();
        assert_eq!(limited.len(), 3);
    }

    // -- edge case tests -----------------------------------------------------

    #[tokio::test]
    async fn test_nonexistent_pattern_tier() {
        let manager = setup_manager().await;
        let tier = manager.get_tier("does-not-exist").await.unwrap();
        assert_eq!(tier, TemperatureTier::Cold);
    }

    #[tokio::test]
    async fn test_sequential_multi_access_recording() {
        let db = setup_test_db().await;
        let manager = TieringManager::new(db, TieringConfig::default())
            .await
            .unwrap();

        // Record 10 sequential accesses (SQLiteDb + &dyn ToSql is !Send,
        // so we test sequential multi-access rather than tokio::spawn).
        for _ in 0..10 {
            manager.record_access("multi-p").await.unwrap();
        }

        let final_rec = manager
            .get_access_record("multi-p")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(final_rec.access_count, 10);
    }
}
