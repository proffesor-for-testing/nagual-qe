//! Tier-based retrieval staging for fast pattern lookup.
//!
//! Implements a three-tier retrieval system inspired by rocket staging:
//! - **Reflex**: O(1) HashMap lookup for elite patterns (tier == Reflex)
//! - **Crystal**: LRU cache for high-value patterns (tier == Crystal)
//! - **Booster**: Full FTS5 + embedding search for all other patterns
//!
//! The staging system wraps the existing `retrieve_patterns()` pipeline,
//! short-circuiting when a cached hit is available in a higher tier.

use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::num::NonZeroUsize;

use ndarray::ArrayView1;
use parking_lot::Mutex;

use super::{
    retrieve_patterns, Pattern, PatternTier, ReasoningBankResult, RetrievalConfig, RetrievalResult,
    PatternQuery, ScoredPattern,
};

/// Default capacity for the Reflex index (number of elite pattern slots).
const DEFAULT_REFLEX_CAPACITY: usize = 256;

/// Default capacity for the Crystal LRU cache.
const DEFAULT_CRYSTAL_CAPACITY: usize = 500;

/// Compute a deterministic hash for a problem string.
///
/// Uses `DefaultHasher` for consistent, non-cryptographic hashing suitable
/// for in-process cache keying.
pub fn compute_problem_hash(problem: &str) -> u64 {
    let mut hasher = DefaultHasher::new();
    problem.hash(&mut hasher);
    hasher.finish()
}

/// Hit/miss statistics for each staging tier.
#[derive(Debug, Clone, Default)]
pub struct StagingStats {
    /// Number of successful Reflex lookups.
    pub reflex_hits: u64,
    /// Number of Reflex misses.
    pub reflex_misses: u64,
    /// Number of successful Crystal lookups.
    pub crystal_hits: u64,
    /// Number of Crystal misses.
    pub crystal_misses: u64,
    /// Number of times the Booster (full search) was invoked.
    pub booster_queries: u64,
    /// Total number of staged_lookup or staged_retrieve calls.
    pub total_queries: u64,
}

impl StagingStats {
    /// Reflex hit rate as a fraction (0.0 - 1.0).
    pub fn reflex_hit_rate(&self) -> f64 {
        let total = self.reflex_hits + self.reflex_misses;
        if total == 0 {
            0.0
        } else {
            self.reflex_hits as f64 / total as f64
        }
    }

    /// Crystal hit rate as a fraction (0.0 - 1.0).
    pub fn crystal_hit_rate(&self) -> f64 {
        let total = self.crystal_hits + self.crystal_misses;
        if total == 0 {
            0.0
        } else {
            self.crystal_hits as f64 / total as f64
        }
    }
}

/// Three-tier retrieval staging cache.
///
/// Patterns are served from the highest available tier:
/// 1. **Reflex** -- O(1) HashMap keyed by `(domain, problem_hash)` for elite patterns.
/// 2. **Crystal** -- LRU cache keyed by pattern ID for high-value patterns.
/// 3. **Booster** -- Falls through to the full `retrieve_patterns()` pipeline.
pub struct RetrievalStaging {
    /// O(1) lookup index for Reflex-tier patterns.
    reflex_index: HashMap<(String, u64), ScoredPattern>,

    /// LRU cache for Crystal-tier patterns.
    crystal_cache: Mutex<lru::LruCache<String, ScoredPattern>>,

    /// Hit/miss counters.
    stats: Mutex<StagingStats>,
}

impl std::fmt::Debug for RetrievalStaging {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RetrievalStaging")
            .field("reflex_count", &self.reflex_index.len())
            .field("crystal_capacity", &self.crystal_cache.lock().cap())
            .field("stats", &*self.stats.lock())
            .finish()
    }
}

impl RetrievalStaging {
    /// Create a new staging cache with default capacities.
    pub fn new() -> Self {
        Self::with_capacity(DEFAULT_REFLEX_CAPACITY, DEFAULT_CRYSTAL_CAPACITY)
    }

    /// Create a staging cache with specified capacities.
    ///
    /// * `reflex_cap` -- Pre-allocated slots for the Reflex HashMap (does not
    ///   hard-cap; it can grow beyond this).
    /// * `crystal_cap` -- Maximum entries in the Crystal LRU cache.
    pub fn with_capacity(reflex_cap: usize, crystal_cap: usize) -> Self {
        let crystal_cap = crystal_cap.max(1); // LRU requires NonZeroUsize
        Self {
            reflex_index: HashMap::with_capacity(reflex_cap),
            crystal_cache: Mutex::new(lru::LruCache::new(
                NonZeroUsize::new(crystal_cap).expect("crystal capacity must be > 0"),
            )),
            stats: Mutex::new(StagingStats::default()),
        }
    }

    /// Populate the staging caches from a slice of `Pattern`.
    ///
    /// Scans each pattern and inserts it into the appropriate tier cache
    /// based on `pattern.tier`:
    /// - `PatternTier::Reflex` -> reflex_index
    /// - `PatternTier::Crystal` -> crystal_cache
    /// - `PatternTier::Booster` -> ignored (served by full search)
    ///
    /// Patterns are wrapped in `ScoredPattern` with default similarity/scores
    /// since the staging lookup returns them without a query-specific score.
    pub fn populate_from_patterns(&mut self, patterns: &[Pattern]) {
        for pattern in patterns {
            let scored = ScoredPattern {
                pattern: pattern.clone(),
                similarity: 1.0,
                final_score: pattern.reward,
                factor_scores: super::FactorScores::default(),
            };

            match pattern.tier {
                PatternTier::Reflex => {
                    let key = (
                        pattern.domain.clone(),
                        compute_problem_hash(&pattern.problem),
                    );
                    self.reflex_index.insert(key, scored);
                }
                PatternTier::Crystal => {
                    self.crystal_cache
                        .lock()
                        .put(pattern.id.clone(), scored);
                }
                PatternTier::Booster => {
                    // Booster-tier patterns are served by the full retrieval pipeline.
                }
            }
        }
    }

    /// Perform a staged lookup by domain and problem hash.
    ///
    /// Checks Reflex first (O(1)), then Crystal (LRU lookup). Returns `None`
    /// if neither tier has a matching pattern. Stats are updated on every call.
    pub fn staged_lookup(&self, domain: &str, problem_hash: u64) -> Option<ScoredPattern> {
        let mut stats = self.stats.lock();
        stats.total_queries += 1;

        // Tier 1: Reflex -- O(1) HashMap
        let reflex_key = (domain.to_string(), problem_hash);
        if let Some(scored) = self.reflex_index.get(&reflex_key) {
            stats.reflex_hits += 1;
            return Some(scored.clone());
        }
        stats.reflex_misses += 1;

        // Tier 2: Crystal -- LRU cache (scan by domain + hash match)
        // Crystal is keyed by pattern ID, so we do a peek-based scan.
        let mut crystal = self.crystal_cache.lock();
        let hit = crystal
            .iter()
            .find(|(_, sp)| {
                sp.pattern.domain == domain
                    && compute_problem_hash(&sp.pattern.problem) == problem_hash
            })
            .map(|(key, sp)| (key.clone(), sp.clone()));

        if let Some((key, scored)) = hit {
            // Promote to recently-used by accessing with `get`
            crystal.get(&key);
            stats.crystal_hits += 1;
            return Some(scored);
        }
        stats.crystal_misses += 1;

        None
    }

    /// Promote a scored pattern to the Reflex tier.
    pub fn promote_to_reflex(&mut self, pattern: &ScoredPattern) {
        let key = (
            pattern.pattern.domain.clone(),
            compute_problem_hash(&pattern.pattern.problem),
        );
        self.reflex_index.insert(key, pattern.clone());
    }

    /// Promote a scored pattern to the Crystal tier.
    pub fn promote_to_crystal(&self, pattern: &ScoredPattern) {
        self.crystal_cache
            .lock()
            .put(pattern.pattern.id.clone(), pattern.clone());
    }

    /// Evict a pattern from all staging tiers by its ID.
    pub fn evict(&mut self, pattern_id: &str) {
        // Remove from Reflex: scan for entries whose pattern.id matches.
        self.reflex_index
            .retain(|_, sp| sp.pattern.id != pattern_id);

        // Remove from Crystal.
        self.crystal_cache.lock().pop(pattern_id);
    }

    /// Return a snapshot of current staging statistics.
    pub fn stats(&self) -> StagingStats {
        self.stats.lock().clone()
    }

    /// Number of entries in the Reflex index.
    pub fn reflex_count(&self) -> usize {
        self.reflex_index.len()
    }

    /// Number of entries in the Crystal cache.
    pub fn crystal_count(&self) -> usize {
        self.crystal_cache.lock().len()
    }
}

impl Default for RetrievalStaging {
    fn default() -> Self {
        Self::new()
    }
}

/// Perform a staged retrieval, short-circuiting through Reflex and Crystal
/// before falling back to the full Booster pipeline.
///
/// # Logic
///
/// 1. If the query targets a single domain AND the query text hashes to a
///    Reflex hit, return immediately with O(1) latency.
/// 2. Otherwise check the Crystal LRU cache for a matching entry.
/// 3. Fall through to `retrieve_patterns()` (Booster stage) for full
///    FTS5 + embedding search.
/// 4. After Booster retrieval, promote returned patterns to the appropriate
///    staging tier if they qualify (based on `PatternTier` thresholds).
///
/// # Arguments
///
/// * `staging` -- The staging cache (mutable to allow promotions).
/// * `patterns` -- Full pattern corpus for the Booster stage.
/// * `query_embedding` -- Embedding vector for the query.
/// * `query` -- Query parameters (domains, filters, limit, etc.).
/// * `config` -- Retrieval pipeline configuration.
pub fn staged_retrieve_patterns(
    staging: &mut RetrievalStaging,
    patterns: &[Pattern],
    query_embedding: &ArrayView1<f32>,
    query: &PatternQuery,
    config: &RetrievalConfig,
) -> ReasoningBankResult<RetrievalResult> {
    let start = std::time::Instant::now();

    // Attempt Reflex / Crystal short-circuit when query targets a single domain.
    if query.domains.len() == 1 {
        let domain = &query.domains[0];
        let problem_hash = compute_problem_hash(&query.query_text);

        if let Some(scored) = staging.staged_lookup(domain, problem_hash) {
            let retrieval_time_ms = start.elapsed().as_millis() as u64;
            return Ok(RetrievalResult {
                patterns: vec![scored],
                total_candidates: 1,
                after_domain_filter: 1,
                after_reward_filter: 1,
                retrieval_time_ms,
                query: query.clone(),
            });
        }
    }

    // Booster: full retrieval pipeline.
    {
        let mut stats = staging.stats.lock();
        stats.booster_queries += 1;
        // If no single-domain query was attempted, count the total query here.
        if query.domains.len() != 1 {
            stats.total_queries += 1;
        }
    }

    let mut result = retrieve_patterns(patterns, query_embedding, query, config)?;
    result.retrieval_time_ms = start.elapsed().as_millis() as u64;

    // Post-retrieval promotion: move qualifying patterns into staging caches.
    for scored in &result.patterns {
        let p = &scored.pattern;
        if let Some(target_tier) = PatternTier::check_promotion(p.reward, p.usage_count) {
            match target_tier {
                PatternTier::Reflex => {
                    staging.promote_to_reflex(scored);
                }
                PatternTier::Crystal => {
                    staging.promote_to_crystal(scored);
                }
                PatternTier::Booster => {}
            }
        }
    }

    Ok(result)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    /// Helper: build a `Pattern` (the mod.rs one) with specified fields.
    fn make_pattern(id: &str, problem: &str, domain: &str, reward: f32, tier: PatternTier) -> Pattern {
        let mut p = Pattern::new(problem, "solution text", domain)
            .with_reward(reward)
            .with_tier(tier);
        p.id = id.to_string();
        p
    }

    /// Helper: build a `ScoredPattern` from a `Pattern`.
    fn scored(pattern: Pattern) -> ScoredPattern {
        ScoredPattern {
            similarity: 1.0,
            final_score: pattern.reward,
            pattern,
            factor_scores: super::super::FactorScores::default(),
        }
    }

    // ------------------------------------------------------------------
    // 1. test_staging_new
    // ------------------------------------------------------------------
    #[test]
    fn test_staging_new() {
        let staging = RetrievalStaging::new();
        assert_eq!(staging.reflex_count(), 0);
        assert_eq!(staging.crystal_count(), 0);
        assert_eq!(staging.stats().total_queries, 0);
    }

    // ------------------------------------------------------------------
    // 2. test_compute_problem_hash
    // ------------------------------------------------------------------
    #[test]
    fn test_compute_problem_hash() {
        let h1 = compute_problem_hash("connection timeout");
        let h2 = compute_problem_hash("connection timeout");
        let h3 = compute_problem_hash("memory leak");

        // Deterministic: same input -> same output.
        assert_eq!(h1, h2);
        // Different inputs should (almost certainly) differ.
        assert_ne!(h1, h3);
    }

    // ------------------------------------------------------------------
    // 3. test_populate_from_patterns
    // ------------------------------------------------------------------
    #[test]
    fn test_populate_from_patterns() {
        let patterns = vec![
            make_pattern("r1", "elite problem", "rust", 0.95, PatternTier::Reflex),
            make_pattern("c1", "crystal problem", "rust", 0.8, PatternTier::Crystal),
            make_pattern("b1", "booster problem", "rust", 0.4, PatternTier::Booster),
            make_pattern("r2", "another elite", "python", 0.92, PatternTier::Reflex),
        ];

        let mut staging = RetrievalStaging::new();
        staging.populate_from_patterns(&patterns);

        assert_eq!(staging.reflex_count(), 2);
        assert_eq!(staging.crystal_count(), 1);
    }

    // ------------------------------------------------------------------
    // 4. test_reflex_lookup_hit
    // ------------------------------------------------------------------
    #[test]
    fn test_reflex_lookup_hit() {
        let mut staging = RetrievalStaging::new();
        let p = make_pattern("r1", "elite problem", "rust", 0.95, PatternTier::Reflex);
        staging.populate_from_patterns(&[p.clone()]);

        let hash = compute_problem_hash("elite problem");
        let result = staging.staged_lookup("rust", hash);

        assert!(result.is_some());
        let sp = result.unwrap();
        assert_eq!(sp.pattern.id, "r1");
    }

    // ------------------------------------------------------------------
    // 5. test_reflex_lookup_miss
    // ------------------------------------------------------------------
    #[test]
    fn test_reflex_lookup_miss() {
        let staging = RetrievalStaging::new();
        let hash = compute_problem_hash("nonexistent");
        let result = staging.staged_lookup("rust", hash);
        assert!(result.is_none());
    }

    // ------------------------------------------------------------------
    // 6. test_crystal_lookup_hit
    // ------------------------------------------------------------------
    #[test]
    fn test_crystal_lookup_hit() {
        let mut staging = RetrievalStaging::new();
        let p = make_pattern("c1", "crystal problem", "rust", 0.8, PatternTier::Crystal);
        staging.populate_from_patterns(&[p.clone()]);

        let hash = compute_problem_hash("crystal problem");
        let result = staging.staged_lookup("rust", hash);

        assert!(result.is_some());
        assert_eq!(result.unwrap().pattern.id, "c1");
    }

    // ------------------------------------------------------------------
    // 7. test_crystal_eviction
    // ------------------------------------------------------------------
    #[test]
    fn test_crystal_eviction() {
        // Create a staging with crystal capacity of 2.
        let mut staging = RetrievalStaging::with_capacity(16, 2);

        let patterns = vec![
            make_pattern("c1", "first", "rust", 0.8, PatternTier::Crystal),
            make_pattern("c2", "second", "rust", 0.8, PatternTier::Crystal),
            make_pattern("c3", "third", "rust", 0.8, PatternTier::Crystal), // should evict c1
        ];

        staging.populate_from_patterns(&patterns);

        // Crystal can hold at most 2 entries.
        assert_eq!(staging.crystal_count(), 2);

        // c1 should have been evicted (oldest/LRU).
        let hash1 = compute_problem_hash("first");
        assert!(staging.staged_lookup("rust", hash1).is_none());

        // c2 and c3 should remain.
        let hash2 = compute_problem_hash("second");
        let hash3 = compute_problem_hash("third");
        assert!(staging.staged_lookup("rust", hash2).is_some());
        assert!(staging.staged_lookup("rust", hash3).is_some());
    }

    // ------------------------------------------------------------------
    // 8. test_promote_to_reflex
    // ------------------------------------------------------------------
    #[test]
    fn test_promote_to_reflex() {
        let mut staging = RetrievalStaging::new();

        // Start with a Crystal-tier pattern.
        let p = make_pattern("c1", "crystal problem", "rust", 0.95, PatternTier::Crystal);
        staging.populate_from_patterns(&[p.clone()]);
        assert_eq!(staging.reflex_count(), 0);
        assert_eq!(staging.crystal_count(), 1);

        // Promote to Reflex.
        let sp = scored(p);
        staging.promote_to_reflex(&sp);

        assert_eq!(staging.reflex_count(), 1);

        // Verify O(1) Reflex lookup works.
        let hash = compute_problem_hash("crystal problem");
        let result = staging.staged_lookup("rust", hash);
        assert!(result.is_some());
        assert_eq!(result.unwrap().pattern.id, "c1");

        // Check that reflex hit was recorded.
        assert_eq!(staging.stats().reflex_hits, 1);
    }

    // ------------------------------------------------------------------
    // 9. test_promote_to_crystal
    // ------------------------------------------------------------------
    #[test]
    fn test_promote_to_crystal() {
        let staging = RetrievalStaging::new();
        assert_eq!(staging.crystal_count(), 0);

        let p = make_pattern("b1", "booster problem", "rust", 0.75, PatternTier::Booster);
        let sp = scored(p);
        staging.promote_to_crystal(&sp);

        assert_eq!(staging.crystal_count(), 1);

        // Verify Crystal lookup works.
        let hash = compute_problem_hash("booster problem");
        let result = staging.staged_lookup("rust", hash);
        assert!(result.is_some());
        assert_eq!(result.unwrap().pattern.id, "b1");
    }

    // ------------------------------------------------------------------
    // 10. test_evict_pattern
    // ------------------------------------------------------------------
    #[test]
    fn test_evict_pattern() {
        let mut staging = RetrievalStaging::new();

        let r = make_pattern("r1", "reflex problem", "rust", 0.95, PatternTier::Reflex);
        let c = make_pattern("c1", "crystal problem", "rust", 0.8, PatternTier::Crystal);
        staging.populate_from_patterns(&[r, c]);

        assert_eq!(staging.reflex_count(), 1);
        assert_eq!(staging.crystal_count(), 1);

        // Evict reflex pattern.
        staging.evict("r1");
        assert_eq!(staging.reflex_count(), 0);

        // Evict crystal pattern.
        staging.evict("c1");
        assert_eq!(staging.crystal_count(), 0);
    }

    // ------------------------------------------------------------------
    // 11. test_staging_stats
    // ------------------------------------------------------------------
    #[test]
    fn test_staging_stats() {
        let mut staging = RetrievalStaging::new();
        let p = make_pattern("r1", "reflex hit", "rust", 0.95, PatternTier::Reflex);
        staging.populate_from_patterns(&[p]);

        let hash_hit = compute_problem_hash("reflex hit");
        let hash_miss = compute_problem_hash("no such thing");

        // One hit.
        staging.staged_lookup("rust", hash_hit);
        // Two misses.
        staging.staged_lookup("rust", hash_miss);
        staging.staged_lookup("python", hash_hit);

        let stats = staging.stats();
        assert_eq!(stats.total_queries, 3);
        assert_eq!(stats.reflex_hits, 1);
        assert_eq!(stats.reflex_misses, 2);
        // Crystal misses for the 2 queries that missed Reflex.
        assert_eq!(stats.crystal_misses, 2);
    }

    // ------------------------------------------------------------------
    // 12. test_staged_retrieve_reflex_shortcircuit
    // ------------------------------------------------------------------
    #[test]
    fn test_staged_retrieve_reflex_shortcircuit() {
        let mut staging = RetrievalStaging::new();

        let p = make_pattern("r1", "elite pattern", "rust", 0.95, PatternTier::Reflex);
        let embedding = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
        let mut p_with_emb = p.clone();
        p_with_emb.embedding = Some(embedding.clone());
        staging.populate_from_patterns(&[p_with_emb.clone()]);

        let query = PatternQuery::new("elite pattern")
            .with_domains(vec!["rust"]);
        let config = RetrievalConfig::default();

        let result = staged_retrieve_patterns(
            &mut staging,
            &[p_with_emb],
            &embedding.view(),
            &query,
            &config,
        )
        .unwrap();

        assert_eq!(result.patterns.len(), 1);
        assert_eq!(result.patterns[0].pattern.id, "r1");
        // Should have been served from Reflex, not Booster.
        assert_eq!(staging.stats().reflex_hits, 1);
        assert_eq!(staging.stats().booster_queries, 0);
    }

    // ------------------------------------------------------------------
    // 13. test_staged_retrieve_falls_through
    // ------------------------------------------------------------------
    #[test]
    fn test_staged_retrieve_falls_through() {
        let mut staging = RetrievalStaging::new();

        // Only Booster-tier patterns -- no staging cache hits.
        let embedding = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
        let mut p = make_pattern("b1", "booster only", "rust", 0.5, PatternTier::Booster);
        p.embedding = Some(embedding.clone());

        let query = PatternQuery::new("booster only")
            .with_domains(vec!["rust"])
            .with_limit(5);
        let config = RetrievalConfig::default();

        let result = staged_retrieve_patterns(
            &mut staging,
            &[p],
            &embedding.view(),
            &query,
            &config,
        )
        .unwrap();

        // Should fall through to Booster.
        assert_eq!(staging.stats().booster_queries, 1);
        // The pattern should be returned by the full retrieval pipeline.
        assert!(!result.is_empty());
    }

    // ------------------------------------------------------------------
    // Additional: stat rate helpers
    // ------------------------------------------------------------------
    #[test]
    fn test_staging_stats_hit_rates() {
        let stats = StagingStats {
            reflex_hits: 3,
            reflex_misses: 7,
            crystal_hits: 5,
            crystal_misses: 5,
            booster_queries: 7,
            total_queries: 10,
        };

        assert!((stats.reflex_hit_rate() - 0.3).abs() < 0.001);
        assert!((stats.crystal_hit_rate() - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_staging_stats_zero_division() {
        let stats = StagingStats::default();
        assert_eq!(stats.reflex_hit_rate(), 0.0);
        assert_eq!(stats.crystal_hit_rate(), 0.0);
    }

    // ------------------------------------------------------------------
    // with_capacity edge case
    // ------------------------------------------------------------------
    #[test]
    fn test_with_capacity_minimum() {
        // Crystal cap is clamped to at least 1.
        let staging = RetrievalStaging::with_capacity(0, 0);
        assert_eq!(staging.reflex_count(), 0);
        assert_eq!(staging.crystal_count(), 0);
    }

    // ------------------------------------------------------------------
    // Default trait
    // ------------------------------------------------------------------
    #[test]
    fn test_staging_default() {
        let staging = RetrievalStaging::default();
        assert_eq!(staging.reflex_count(), 0);
        assert_eq!(staging.crystal_count(), 0);
    }
}
