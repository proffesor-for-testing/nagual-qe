//! Wormhole Detector - Analyzes trajectories for co-access patterns.
//!
//! The detector scans trajectories and co-access events to identify candidate
//! wormholes. It considers:
//!
//! - **Frequency**: How often patterns are accessed together
//! - **Path Distance**: How far apart patterns are in the graph
//! - **Traversal Savings**: Whether a wormhole would provide significant shortcuts
//!
//! # Detection Process
//!
//! 1. Collect co-access events from trajectories
//! 2. Filter candidates by frequency threshold
//! 3. Calculate path distance for each candidate pair
//! 4. Filter by minimum traversal savings
//! 5. Rank by combined score (frequency * savings)
//! 6. Trigger wormhole creation for top candidates
//!
//! # Example
//!
//! ```ignore
//! use nagual::profdag::wormhole_detector::{WormholeDetector, DetectorConfig};
//!
//! let detector = WormholeDetector::new(adapter, DetectorConfig::default()).await?;
//!
//! // Analyze recent trajectories
//! let candidates = detector.analyze_recent_trajectories(100).await?;
//!
//! // Create wormholes for top candidates
//! for candidate in candidates.iter().take(10) {
//!     detector.create_wormhole_for_candidate(candidate).await?;
//! }
//! ```

use std::collections::HashMap;
use std::sync::Arc;

use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};

use crate::db::DualWriteAdapter;
use crate::error::{NagualError, Result};
use crate::learning::trajectory::Trajectory;
use crate::reasoning_bank::pattern::PatternId;

use super::wormhole::{CoAccessRecord, Wormhole, WormholeCreationReason, WormholeManager};

/// Priority boost multiplier for cross-domain wormhole candidates.
/// Cross-domain wormholes represent valuable knowledge bridges between
/// different areas, so they receive a 1.5x weight boost in scoring.
pub const CROSS_DOMAIN_BOOST: f32 = 1.5;

/// Extract the root domain from a dot-separated domain string.
///
/// For example, "rust.async.tokio" returns "rust", and "python.ml" returns "python".
/// If there is no dot separator, the entire string is returned.
///
/// # Examples
///
/// ```ignore
/// assert_eq!(root_domain("rust.async"), "rust");
/// assert_eq!(root_domain("python"), "python");
/// assert_eq!(root_domain(""), "");
/// ```
pub fn root_domain(domain: &str) -> &str {
    domain.split('.').next().unwrap_or(domain)
}

/// Configuration for the wormhole detector.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectorConfig {
    /// Minimum co-access frequency to consider for wormhole.
    /// Default: 3
    #[serde(default = "default_min_frequency")]
    pub min_frequency: u32,

    /// Minimum path distance to warrant a wormhole.
    /// Default: 3 edges
    #[serde(default = "default_min_path_distance")]
    pub min_path_distance: u32,

    /// Minimum traversal savings (as fraction, 0.0-1.0).
    /// Default: 0.5 (50% reduction)
    #[serde(default = "default_min_savings")]
    pub min_traversal_savings: f32,

    /// Maximum candidates to evaluate per run.
    /// Default: 1000
    #[serde(default = "default_max_candidates")]
    pub max_candidates: usize,

    /// Time window for trajectory analysis (hours).
    /// Default: 168 (7 days)
    #[serde(default = "default_time_window_hours")]
    pub time_window_hours: u32,

    /// Weight for frequency in candidate scoring.
    /// Default: 0.6
    #[serde(default = "default_frequency_weight")]
    pub frequency_weight: f32,

    /// Weight for traversal savings in candidate scoring.
    /// Default: 0.4
    #[serde(default = "default_savings_weight")]
    pub savings_weight: f32,

    /// Whether to analyze completed trajectories only.
    /// Default: true
    #[serde(default = "default_completed_only")]
    pub completed_only: bool,

    /// Minimum trajectory reward to include in analysis.
    /// Default: 0.5
    #[serde(default = "default_min_trajectory_reward")]
    pub min_trajectory_reward: f32,
}

fn default_min_frequency() -> u32 { 3 }
fn default_min_path_distance() -> u32 { 3 }
fn default_min_savings() -> f32 { 0.5 }
fn default_max_candidates() -> usize { 1000 }
fn default_time_window_hours() -> u32 { 168 }
fn default_frequency_weight() -> f32 { 0.6 }
fn default_savings_weight() -> f32 { 0.4 }
fn default_completed_only() -> bool { true }
fn default_min_trajectory_reward() -> f32 { 0.5 }

impl Default for DetectorConfig {
    fn default() -> Self {
        Self {
            min_frequency: default_min_frequency(),
            min_path_distance: default_min_path_distance(),
            min_traversal_savings: default_min_savings(),
            max_candidates: default_max_candidates(),
            time_window_hours: default_time_window_hours(),
            frequency_weight: default_frequency_weight(),
            savings_weight: default_savings_weight(),
            completed_only: default_completed_only(),
            min_trajectory_reward: default_min_trajectory_reward(),
        }
    }
}

impl DetectorConfig {
    /// Set minimum frequency.
    pub fn with_min_frequency(mut self, freq: u32) -> Self {
        self.min_frequency = freq.max(1);
        self
    }

    /// Set minimum path distance.
    pub fn with_min_path_distance(mut self, dist: u32) -> Self {
        self.min_path_distance = dist.max(2);
        self
    }

    /// Set time window in hours.
    pub fn with_time_window(mut self, hours: u32) -> Self {
        self.time_window_hours = hours;
        self
    }
}

/// A candidate wormhole identified by the detector.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WormholeCandidate {
    /// Source pattern ID.
    pub source_id: String,

    /// Target pattern ID.
    pub target_id: String,

    /// Number of co-accesses observed.
    pub co_access_count: u32,

    /// Estimated path distance without wormhole.
    pub path_distance: Option<u32>,

    /// Calculated traversal savings (0.0-1.0).
    pub traversal_savings: f32,

    /// Combined score for ranking candidates.
    pub score: f32,

    /// Session IDs where co-access was observed.
    pub session_ids: Vec<String>,

    /// Trajectory IDs where co-access was observed.
    pub trajectory_ids: Vec<String>,

    /// First observation time.
    pub first_observed: DateTime<Utc>,

    /// Most recent observation time.
    pub last_observed: DateTime<Utc>,
}

impl WormholeCandidate {
    /// Calculate combined score from frequency and savings.
    pub fn calculate_score(
        &mut self,
        frequency_weight: f32,
        savings_weight: f32,
        max_frequency: u32,
    ) {
        let normalized_frequency = if max_frequency > 0 {
            self.co_access_count as f32 / max_frequency as f32
        } else {
            0.0
        };

        self.score = frequency_weight * normalized_frequency
            + savings_weight * self.traversal_savings;
    }

    /// Check if this candidate meets the minimum requirements.
    pub fn meets_requirements(&self, config: &DetectorConfig) -> bool {
        self.co_access_count >= config.min_frequency
            && self.traversal_savings >= config.min_traversal_savings
            && self.path_distance.map_or(true, |d| d >= config.min_path_distance)
    }
}

/// Result of a detection run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectionResult {
    /// Candidates that passed all filters.
    pub candidates: Vec<WormholeCandidate>,

    /// Total co-access pairs analyzed.
    pub pairs_analyzed: usize,

    /// Pairs that passed frequency filter.
    pub passed_frequency: usize,

    /// Pairs that passed traversal savings filter.
    pub passed_savings: usize,

    /// Trajectories analyzed.
    pub trajectories_analyzed: usize,

    /// Duration of analysis in milliseconds.
    pub duration_ms: u64,

    /// Time window analyzed.
    pub time_window: (DateTime<Utc>, DateTime<Utc>),
}

impl DetectionResult {
    /// Get top N candidates by score.
    pub fn top_candidates(&self, n: usize) -> Vec<&WormholeCandidate> {
        self.candidates.iter().take(n).collect()
    }

    /// Get candidates above a score threshold.
    pub fn candidates_above_score(&self, min_score: f32) -> Vec<&WormholeCandidate> {
        self.candidates
            .iter()
            .filter(|c| c.score >= min_score)
            .collect()
    }
}

/// Statistics about detection activity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectorStats {
    /// Total detection runs performed.
    pub total_runs: u64,

    /// Total candidates identified.
    pub total_candidates: u64,

    /// Total wormholes created from detections.
    pub wormholes_created: u64,

    /// Average candidates per run.
    pub avg_candidates_per_run: f32,

    /// Last detection run time.
    pub last_run: Option<DateTime<Utc>>,
}

/// Wormhole detector that analyzes trajectories and co-access patterns.
pub struct WormholeDetector {
    /// Database adapter.
    adapter: Arc<DualWriteAdapter>,

    /// Detector configuration.
    config: DetectorConfig,

    /// Wormhole manager for creating wormholes.
    wormhole_manager: Arc<WormholeManager>,

    /// Statistics.
    stats: parking_lot::RwLock<DetectorStats>,
}

impl WormholeDetector {
    /// Create a new wormhole detector.
    pub async fn new(
        adapter: Arc<DualWriteAdapter>,
        wormhole_manager: Arc<WormholeManager>,
        config: DetectorConfig,
    ) -> Result<Self> {
        Ok(Self {
            adapter,
            config,
            wormhole_manager,
            stats: parking_lot::RwLock::new(DetectorStats {
                total_runs: 0,
                total_candidates: 0,
                wormholes_created: 0,
                avg_candidates_per_run: 0.0,
                last_run: None,
            }),
        })
    }

    /// Create with default configuration.
    pub async fn with_defaults(
        adapter: Arc<DualWriteAdapter>,
        wormhole_manager: Arc<WormholeManager>,
    ) -> Result<Self> {
        Self::new(adapter, wormhole_manager, DetectorConfig::default()).await
    }

    /// Get the configuration.
    pub fn config(&self) -> &DetectorConfig {
        &self.config
    }

    /// Get current statistics.
    pub fn stats(&self) -> DetectorStats {
        self.stats.read().clone()
    }

    /// Analyze a single trajectory for co-access patterns.
    ///
    /// Records co-access events for all pattern pairs in the trajectory.
    pub async fn analyze_trajectory(&self, trajectory: &Trajectory) -> Result<Vec<Wormhole>> {
        let pattern_ids: Vec<String> = trajectory
            .all_pattern_ids()
            .into_iter()
            .map(|p| p.as_str().to_string())
            .collect();

        if pattern_ids.len() < 2 {
            return Ok(Vec::new());
        }

        // Record all co-accesses
        let wormholes = self
            .wormhole_manager
            .record_trajectory_co_accesses(
                &pattern_ids,
                trajectory.session_id.as_deref(),
                Some(trajectory.id.as_str()),
            )
            .await?;

        Ok(wormholes)
    }

    /// Analyze recent trajectories within the time window.
    pub async fn analyze_recent_trajectories(&self, limit: usize) -> Result<DetectionResult> {
        let start_time = std::time::Instant::now();
        let end_time = Utc::now();
        let start_window = end_time - Duration::hours(self.config.time_window_hours as i64);

        // Collect co-access events from stored data
        let co_accesses = self.collect_co_accesses(start_window, limit).await?;

        // Build candidates from co-access data
        let mut candidates = self.build_candidates(co_accesses).await?;

        // Calculate path distances for candidates
        self.calculate_path_distances(&mut candidates).await?;

        // Calculate scores
        let max_frequency = candidates
            .iter()
            .map(|c| c.co_access_count)
            .max()
            .unwrap_or(1);

        for candidate in &mut candidates {
            candidate.calculate_score(
                self.config.frequency_weight,
                self.config.savings_weight,
                max_frequency,
            );
        }

        // Filter and sort
        let passed_frequency = candidates.len();
        candidates.retain(|c| c.meets_requirements(&self.config));
        let passed_savings = candidates.len();

        candidates.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        candidates.truncate(self.config.max_candidates);

        // Update stats
        {
            let mut stats = self.stats.write();
            stats.total_runs += 1;
            stats.total_candidates += candidates.len() as u64;
            stats.avg_candidates_per_run = stats.total_candidates as f32 / stats.total_runs as f32;
            stats.last_run = Some(Utc::now());
        }

        let result = DetectionResult {
            candidates,
            pairs_analyzed: passed_frequency,
            passed_frequency,
            passed_savings,
            trajectories_analyzed: limit,
            duration_ms: start_time.elapsed().as_millis() as u64,
            time_window: (start_window, end_time),
        };

        info!(
            candidates = result.candidates.len(),
            pairs_analyzed = result.pairs_analyzed,
            duration_ms = result.duration_ms,
            "Detection run completed"
        );

        Ok(result)
    }

    /// Run detection and automatically create wormholes for top candidates.
    pub async fn detect_and_create_wormholes(&self, max_create: usize) -> Result<Vec<Wormhole>> {
        let result = self.analyze_recent_trajectories(1000).await?;

        let mut created = Vec::new();
        for candidate in result.candidates.iter().take(max_create) {
            match self.create_wormhole_for_candidate(candidate).await {
                Ok(wormhole) => {
                    created.push(wormhole);
                    self.stats.write().wormholes_created += 1;
                }
                Err(e) => {
                    warn!(
                        source = %candidate.source_id,
                        target = %candidate.target_id,
                        error = %e,
                        "Failed to create wormhole for candidate"
                    );
                }
            }
        }

        info!(
            created = created.len(),
            max_create = max_create,
            "Wormhole creation completed"
        );

        Ok(created)
    }

    /// Create a wormhole for a specific candidate.
    pub async fn create_wormhole_for_candidate(
        &self,
        candidate: &WormholeCandidate,
    ) -> Result<Wormhole> {
        self.wormhole_manager
            .create_wormhole(
                &candidate.source_id,
                &candidate.target_id,
                WormholeCreationReason::CoAccess {
                    count: candidate.co_access_count,
                    avg_path_distance: candidate.path_distance.map(|d| d as f32),
                },
            )
            .await
    }

    /// Analyze patterns accessed in a session.
    ///
    /// Records co-accesses for patterns accessed within the same session.
    pub async fn analyze_session_patterns(
        &self,
        session_id: &str,
        pattern_ids: &[PatternId],
    ) -> Result<Vec<Wormhole>> {
        let ids: Vec<String> = pattern_ids.iter().map(|p| p.as_str().to_string()).collect();

        self.wormhole_manager
            .record_trajectory_co_accesses(&ids, Some(session_id), None)
            .await
    }

    /// Get candidates that already meet the threshold but don't have wormholes.
    pub async fn get_pending_candidates(&self, limit: usize) -> Result<Vec<WormholeCandidate>> {
        let sql = format!(
            r#"
            SELECT
                ca.pattern_a, ca.pattern_b, ca.count,
                ca.first_accessed, ca.last_accessed
            FROM wormhole_co_access ca
            WHERE ca.count >= {}
            AND NOT EXISTS (
                SELECT 1 FROM wormholes w
                WHERE (w.source_id = ca.pattern_a AND w.target_id = ca.pattern_b)
                   OR (w.source_id = ca.pattern_b AND w.target_id = ca.pattern_a)
            )
            ORDER BY ca.count DESC
            LIMIT {}
            "#,
            self.config.min_frequency,
            limit
        );

        let candidates: Vec<WormholeCandidate> = self
            .adapter
            .sqlite()
            .query(&sql, &[], |row| {
                let pattern_a: String = row.get(0)?;
                let pattern_b: String = row.get(1)?;
                let count: i64 = row.get(2)?;
                let first_accessed_str: String = row.get(3)?;
                let last_accessed_str: String = row.get(4)?;

                let first_observed = DateTime::parse_from_rfc3339(&first_accessed_str)
                    .map(|dt| dt.with_timezone(&Utc))
                    .unwrap_or_else(|_| Utc::now());

                let last_observed = DateTime::parse_from_rfc3339(&last_accessed_str)
                    .map(|dt| dt.with_timezone(&Utc))
                    .unwrap_or_else(|_| Utc::now());

                Ok(WormholeCandidate {
                    source_id: pattern_a,
                    target_id: pattern_b,
                    co_access_count: count as u32,
                    path_distance: None,
                    traversal_savings: 0.5, // Default, will be calculated
                    score: 0.0,
                    session_ids: Vec::new(),
                    trajectory_ids: Vec::new(),
                    first_observed,
                    last_observed,
                })
            })
            .await
            .map_err(|e| NagualError::internal(format!("Failed to get pending candidates: {}", e)))?;

        Ok(candidates)
    }

    /// Detect wormhole candidates between different root domains.
    ///
    /// Cross-domain wormholes get a 1.5x weight boost as they represent
    /// valuable knowledge bridges between different areas.
    ///
    /// This method queries the `reasoning_patterns` table to resolve the
    /// `category` (domain) for each pattern in the co-access records,
    /// then filters to only pairs where the root domains differ.
    pub async fn detect_cross_domain_wormholes(&self) -> Result<Vec<WormholeCandidate>> {
        let end_time = Utc::now();
        let start_window = end_time - Duration::hours(self.config.time_window_hours as i64);

        // Collect co-access events
        let co_accesses = self.collect_co_accesses(start_window, 10000).await?;

        if co_accesses.is_empty() {
            debug!("No co-access records found for cross-domain detection");
            return Ok(Vec::new());
        }

        // Build candidates from co-access data
        let candidates = self.build_candidates(co_accesses).await?;

        if candidates.is_empty() {
            debug!("No candidates after filtering existing wormholes");
            return Ok(Vec::new());
        }

        // Collect all unique pattern IDs from candidates
        let mut pattern_ids: Vec<String> = Vec::new();
        for candidate in &candidates {
            if !pattern_ids.contains(&candidate.source_id) {
                pattern_ids.push(candidate.source_id.clone());
            }
            if !pattern_ids.contains(&candidate.target_id) {
                pattern_ids.push(candidate.target_id.clone());
            }
        }

        // Query domains for all pattern IDs from reasoning_patterns
        let domain_map = self.lookup_pattern_domains(&pattern_ids).await?;

        // Filter and boost cross-domain candidates
        let cross_domain = filter_cross_domain_candidates(candidates, &domain_map);

        info!(
            total_cross_domain = cross_domain.len(),
            pattern_ids_resolved = domain_map.len(),
            "Cross-domain wormhole detection completed"
        );

        Ok(cross_domain)
    }

    /// Look up domain (category) for a set of pattern IDs from the database.
    ///
    /// Returns a map from pattern ID to its domain/category string.
    async fn lookup_pattern_domains(
        &self,
        pattern_ids: &[String],
    ) -> Result<HashMap<String, String>> {
        if pattern_ids.is_empty() {
            return Ok(HashMap::new());
        }

        // Build query with placeholders for each pattern ID.
        // SQLite does not support array parameters, so we build the IN clause manually.
        let placeholders: Vec<String> = pattern_ids.iter().map(|_| "?".to_string()).collect();
        let sql = format!(
            "SELECT id, category FROM reasoning_patterns WHERE id IN ({})",
            placeholders.join(", ")
        );

        // Convert to trait-object references for the query
        let params: Vec<&dyn rusqlite::ToSql> = pattern_ids
            .iter()
            .map(|id| id as &dyn rusqlite::ToSql)
            .collect();

        let rows: Vec<(String, String)> = self
            .adapter
            .sqlite()
            .query(&sql, &params, |row| {
                Ok((row.get(0)?, row.get(1)?))
            })
            .await
            .map_err(|e| {
                NagualError::internal(format!("Failed to lookup pattern domains: {}", e))
            })?;

        let mut map = HashMap::with_capacity(rows.len());
        for (id, category) in rows {
            map.insert(id, category);
        }

        Ok(map)
    }

    // ========================================================================
    // Private Helper Methods
    // ========================================================================

    /// Collect co-access events from the database.
    async fn collect_co_accesses(
        &self,
        since: DateTime<Utc>,
        _limit: usize,
    ) -> Result<Vec<CoAccessRecord>> {
        let sql = r#"
            SELECT pattern_a, pattern_b, count, first_accessed, last_accessed
            FROM wormhole_co_access
            WHERE last_accessed >= ?
            AND count >= ?
            ORDER BY count DESC
            LIMIT 10000
        "#;

        let records: Vec<CoAccessRecord> = self
            .adapter
            .sqlite()
            .query(sql, &[&since.to_rfc3339(), &(self.config.min_frequency as i64)], |row| {
                let pattern_a: String = row.get(0)?;
                let pattern_b: String = row.get(1)?;
                let count: i64 = row.get(2)?;
                let first_accessed_str: String = row.get(3)?;
                let last_accessed_str: String = row.get(4)?;

                let first_accessed = DateTime::parse_from_rfc3339(&first_accessed_str)
                    .map(|dt| dt.with_timezone(&Utc))
                    .unwrap_or_else(|_| Utc::now());

                let last_accessed = DateTime::parse_from_rfc3339(&last_accessed_str)
                    .map(|dt| dt.with_timezone(&Utc))
                    .unwrap_or_else(|_| Utc::now());

                Ok(CoAccessRecord {
                    pattern_a,
                    pattern_b,
                    count: count as u32,
                    first_accessed,
                    last_accessed,
                    session_ids: Vec::new(),
                    trajectory_ids: Vec::new(),
                })
            })
            .await
            .map_err(|e| NagualError::internal(format!("Failed to collect co-accesses: {}", e)))?;

        Ok(records)
    }

    /// Build candidates from co-access records.
    async fn build_candidates(
        &self,
        co_accesses: Vec<CoAccessRecord>,
    ) -> Result<Vec<WormholeCandidate>> {
        let mut candidates = Vec::with_capacity(co_accesses.len());

        for record in co_accesses {
            // Skip if wormhole already exists
            if self
                .wormhole_manager
                .wormhole_exists(&record.pattern_a, &record.pattern_b)
                .await?
            {
                continue;
            }

            candidates.push(WormholeCandidate {
                source_id: record.pattern_a,
                target_id: record.pattern_b,
                co_access_count: record.count,
                path_distance: None,
                traversal_savings: 0.5, // Will be calculated based on path distance
                score: 0.0,
                session_ids: record.session_ids,
                trajectory_ids: record.trajectory_ids,
                first_observed: record.first_accessed,
                last_observed: record.last_accessed,
            });
        }

        Ok(candidates)
    }

    /// Calculate path distances for candidates.
    ///
    /// Uses BFS to find shortest path between pattern pairs in the graph.
    async fn calculate_path_distances(
        &self,
        candidates: &mut Vec<WormholeCandidate>,
    ) -> Result<()> {
        // For each candidate, try to find the shortest path
        for candidate in candidates.iter_mut() {
            let distance = self
                .find_shortest_path(&candidate.source_id, &candidate.target_id)
                .await?;

            candidate.path_distance = distance;
            candidate.traversal_savings = if let Some(dist) = distance {
                if dist > 1 {
                    // Savings = (distance - 1) / distance
                    // E.g., distance of 4 gives savings of 3/4 = 0.75
                    (dist - 1) as f32 / dist as f32
                } else {
                    0.0
                }
            } else {
                // If no path exists, assume high savings value to encourage wormhole
                0.8
            };
        }

        Ok(())
    }

    /// Find shortest path between two nodes using BFS.
    async fn find_shortest_path(
        &self,
        source_id: &str,
        target_id: &str,
    ) -> Result<Option<u32>> {
        // Simple BFS implementation for path finding
        // Limited to MAX_DEPTH to avoid performance issues
        const MAX_DEPTH: u32 = 10;

        let mut visited = std::collections::HashSet::new();
        let mut queue = std::collections::VecDeque::new();

        visited.insert(source_id.to_string());
        queue.push_back((source_id.to_string(), 0u32));

        while let Some((current_id, depth)) = queue.pop_front() {
            if depth >= MAX_DEPTH {
                continue;
            }

            // Get neighbors from profdag_edges
            let sql = r#"
                SELECT target_id FROM profdag_edges
                WHERE source_id = ? AND edge_type != 'wormhole'
                UNION
                SELECT source_id FROM profdag_edges
                WHERE target_id = ? AND edge_type != 'wormhole'
            "#;

            let neighbors: Vec<String> = self
                .adapter
                .sqlite()
                .query(sql, &[&current_id, &current_id], |row| row.get(0))
                .await
                .unwrap_or_else(|_| Vec::new());

            for neighbor in neighbors {
                if neighbor == target_id {
                    return Ok(Some(depth + 1));
                }

                if !visited.contains(&neighbor) {
                    visited.insert(neighbor.clone());
                    queue.push_back((neighbor, depth + 1));
                }
            }
        }

        // No path found within depth limit
        Ok(None)
    }
}

/// Filter a list of wormhole candidates to only cross-domain pairs and apply
/// the [`CROSS_DOMAIN_BOOST`] multiplier to their scores.
///
/// A candidate is cross-domain when the root domains (first segment of a
/// dot-separated category string) of its source and target patterns differ.
///
/// Candidates whose pattern IDs are not present in `domain_map` are excluded
/// because their domain cannot be verified.
///
/// # Arguments
///
/// * `candidates` - The full list of candidates to filter.
/// * `domain_map` - A mapping from pattern ID to its domain/category string.
///
/// # Returns
///
/// A new `Vec<WormholeCandidate>` containing only cross-domain pairs, sorted
/// descending by boosted score.
pub fn filter_cross_domain_candidates(
    candidates: Vec<WormholeCandidate>,
    domain_map: &HashMap<String, String>,
) -> Vec<WormholeCandidate> {
    let mut cross_domain: Vec<WormholeCandidate> = candidates
        .into_iter()
        .filter_map(|mut candidate| {
            let source_domain = domain_map.get(&candidate.source_id)?;
            let target_domain = domain_map.get(&candidate.target_id)?;

            let source_root = root_domain(source_domain);
            let target_root = root_domain(target_domain);

            if source_root != target_root {
                // Apply 1.5x priority boost
                candidate.score *= CROSS_DOMAIN_BOOST;
                Some(candidate)
            } else {
                None
            }
        })
        .collect();

    // Sort by boosted score descending
    cross_domain.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    cross_domain
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detector_config_defaults() {
        let config = DetectorConfig::default();
        assert_eq!(config.min_frequency, 3);
        assert_eq!(config.min_path_distance, 3);
        assert!((config.min_traversal_savings - 0.5).abs() < 0.001);
        assert_eq!(config.time_window_hours, 168);
    }

    #[test]
    fn test_detector_config_builder() {
        let config = DetectorConfig::default()
            .with_min_frequency(5)
            .with_min_path_distance(4)
            .with_time_window(24);

        assert_eq!(config.min_frequency, 5);
        assert_eq!(config.min_path_distance, 4);
        assert_eq!(config.time_window_hours, 24);
    }

    #[test]
    fn test_candidate_score_calculation() {
        let mut candidate = WormholeCandidate {
            source_id: "A".to_string(),
            target_id: "B".to_string(),
            co_access_count: 5,
            path_distance: Some(4),
            traversal_savings: 0.75,
            score: 0.0,
            session_ids: vec![],
            trajectory_ids: vec![],
            first_observed: Utc::now(),
            last_observed: Utc::now(),
        };

        candidate.calculate_score(0.6, 0.4, 10);

        // Expected: 0.6 * (5/10) + 0.4 * 0.75 = 0.3 + 0.3 = 0.6
        assert!((candidate.score - 0.6).abs() < 0.001);
    }

    #[test]
    fn test_candidate_meets_requirements() {
        let config = DetectorConfig::default();

        let good_candidate = WormholeCandidate {
            source_id: "A".to_string(),
            target_id: "B".to_string(),
            co_access_count: 5,
            path_distance: Some(4),
            traversal_savings: 0.6,
            score: 0.5,
            session_ids: vec![],
            trajectory_ids: vec![],
            first_observed: Utc::now(),
            last_observed: Utc::now(),
        };

        assert!(good_candidate.meets_requirements(&config));

        let low_frequency = WormholeCandidate {
            co_access_count: 2,
            ..good_candidate.clone()
        };
        assert!(!low_frequency.meets_requirements(&config));

        let low_savings = WormholeCandidate {
            traversal_savings: 0.3,
            ..good_candidate.clone()
        };
        assert!(!low_savings.meets_requirements(&config));
    }

    #[test]
    fn test_detection_result_top_candidates() {
        let candidates = vec![
            WormholeCandidate {
                source_id: "A".to_string(),
                target_id: "B".to_string(),
                co_access_count: 10,
                path_distance: Some(4),
                traversal_savings: 0.75,
                score: 0.9,
                session_ids: vec![],
                trajectory_ids: vec![],
                first_observed: Utc::now(),
                last_observed: Utc::now(),
            },
            WormholeCandidate {
                source_id: "C".to_string(),
                target_id: "D".to_string(),
                co_access_count: 5,
                path_distance: Some(3),
                traversal_savings: 0.67,
                score: 0.7,
                session_ids: vec![],
                trajectory_ids: vec![],
                first_observed: Utc::now(),
                last_observed: Utc::now(),
            },
        ];

        let result = DetectionResult {
            candidates,
            pairs_analyzed: 100,
            passed_frequency: 50,
            passed_savings: 2,
            trajectories_analyzed: 10,
            duration_ms: 100,
            time_window: (Utc::now() - Duration::hours(24), Utc::now()),
        };

        let top = result.top_candidates(1);
        assert_eq!(top.len(), 1);
        assert_eq!(top[0].source_id, "A");

        let above_score = result.candidates_above_score(0.8);
        assert_eq!(above_score.len(), 1);
    }

    #[test]
    fn test_traversal_savings_calculation() {
        // Distance 4: savings = (4-1)/4 = 0.75
        let savings_4: f64 = (4.0 - 1.0) / 4.0;
        assert!((savings_4 - 0.75).abs() < 0.001);

        // Distance 2: savings = (2-1)/2 = 0.5
        let savings_2: f64 = (2.0 - 1.0) / 2.0;
        assert!((savings_2 - 0.5).abs() < 0.001);

        // Distance 10: savings = (10-1)/10 = 0.9
        let savings_10: f64 = (10.0 - 1.0) / 10.0;
        assert!((savings_10 - 0.9).abs() < 0.001);
    }

    // ====================================================================
    // Cross-domain detection tests
    // ====================================================================

    #[test]
    fn test_root_domain_extracts_first_segment() {
        assert_eq!(root_domain("rust.async.tokio"), "rust");
        assert_eq!(root_domain("python.ml.pytorch"), "python");
        assert_eq!(root_domain("rust"), "rust");
        assert_eq!(root_domain(""), "");
    }

    #[test]
    fn test_root_domain_single_segment() {
        assert_eq!(root_domain("devops"), "devops");
        assert_eq!(root_domain("testing"), "testing");
    }

    #[test]
    fn test_cross_domain_boost_constant() {
        assert!((CROSS_DOMAIN_BOOST - 1.5).abs() < f32::EPSILON);
    }

    /// Helper to build a candidate with a pre-set score for testing.
    fn make_candidate(source: &str, target: &str, score: f32) -> WormholeCandidate {
        WormholeCandidate {
            source_id: source.to_string(),
            target_id: target.to_string(),
            co_access_count: 5,
            path_distance: Some(4),
            traversal_savings: 0.75,
            score,
            session_ids: vec![],
            trajectory_ids: vec![],
            first_observed: Utc::now(),
            last_observed: Utc::now(),
        }
    }

    #[test]
    fn test_filter_cross_domain_returns_only_cross_domain() {
        let candidates = vec![
            make_candidate("p1", "p2", 0.8), // rust <-> python  (cross-domain)
            make_candidate("p3", "p4", 0.7), // rust <-> rust    (same-domain)
            make_candidate("p5", "p6", 0.6), // python <-> devops (cross-domain)
        ];

        let mut domain_map = HashMap::new();
        domain_map.insert("p1".to_string(), "rust.async".to_string());
        domain_map.insert("p2".to_string(), "python.ml".to_string());
        domain_map.insert("p3".to_string(), "rust.web".to_string());
        domain_map.insert("p4".to_string(), "rust.cli".to_string());
        domain_map.insert("p5".to_string(), "python.data".to_string());
        domain_map.insert("p6".to_string(), "devops.docker".to_string());

        let result = filter_cross_domain_candidates(candidates, &domain_map);

        assert_eq!(result.len(), 2, "Only cross-domain candidates should remain");

        // Verify the same-domain pair (p3-p4, both rust.*) was filtered out
        assert!(
            result.iter().all(|c| !(c.source_id == "p3" && c.target_id == "p4")),
            "Same-domain candidate should be filtered out"
        );
    }

    #[test]
    fn test_filter_cross_domain_applies_boost() {
        let candidates = vec![
            make_candidate("p1", "p2", 0.8),
        ];

        let mut domain_map = HashMap::new();
        domain_map.insert("p1".to_string(), "rust.async".to_string());
        domain_map.insert("p2".to_string(), "python.ml".to_string());

        let result = filter_cross_domain_candidates(candidates, &domain_map);

        assert_eq!(result.len(), 1);
        let boosted_score = result[0].score;
        let expected = 0.8 * CROSS_DOMAIN_BOOST; // 0.8 * 1.5 = 1.2
        assert!(
            (boosted_score - expected).abs() < 0.001,
            "Score should be boosted by {}x: expected {}, got {}",
            CROSS_DOMAIN_BOOST,
            expected,
            boosted_score
        );
    }

    #[test]
    fn test_filter_cross_domain_excludes_unknown_patterns() {
        let candidates = vec![
            make_candidate("p1", "p_unknown", 0.8),
        ];

        let mut domain_map = HashMap::new();
        domain_map.insert("p1".to_string(), "rust".to_string());
        // p_unknown is not in the domain map

        let result = filter_cross_domain_candidates(candidates, &domain_map);

        assert!(
            result.is_empty(),
            "Candidates with unknown domains should be excluded"
        );
    }

    #[test]
    fn test_filter_cross_domain_sorted_by_boosted_score() {
        let candidates = vec![
            make_candidate("p1", "p2", 0.5), // rust <-> python
            make_candidate("p3", "p4", 0.9), // devops <-> python
            make_candidate("p5", "p6", 0.7), // rust <-> devops
        ];

        let mut domain_map = HashMap::new();
        domain_map.insert("p1".to_string(), "rust".to_string());
        domain_map.insert("p2".to_string(), "python".to_string());
        domain_map.insert("p3".to_string(), "devops".to_string());
        domain_map.insert("p4".to_string(), "python".to_string());
        domain_map.insert("p5".to_string(), "rust".to_string());
        domain_map.insert("p6".to_string(), "devops".to_string());

        let result = filter_cross_domain_candidates(candidates, &domain_map);

        assert_eq!(result.len(), 3);
        // All scores are boosted by 1.5x: 0.5*1.5=0.75, 0.9*1.5=1.35, 0.7*1.5=1.05
        // Sorted descending: 1.35, 1.05, 0.75
        assert!(
            result[0].score > result[1].score && result[1].score > result[2].score,
            "Results should be sorted by boosted score descending: {}, {}, {}",
            result[0].score,
            result[1].score,
            result[2].score
        );
        // The highest-scored candidate should be p3-p4 (0.9 * 1.5 = 1.35)
        assert_eq!(result[0].source_id, "p3");
        assert_eq!(result[0].target_id, "p4");
    }

    #[test]
    fn test_filter_cross_domain_empty_input() {
        let candidates: Vec<WormholeCandidate> = vec![];
        let domain_map = HashMap::new();

        let result = filter_cross_domain_candidates(candidates, &domain_map);

        assert!(result.is_empty());
    }

    #[test]
    fn test_filter_cross_domain_same_root_different_sub() {
        // Both patterns have root domain "rust" even though sub-domains differ
        let candidates = vec![
            make_candidate("p1", "p2", 0.8),
        ];

        let mut domain_map = HashMap::new();
        domain_map.insert("p1".to_string(), "rust.async.tokio".to_string());
        domain_map.insert("p2".to_string(), "rust.web.actix".to_string());

        let result = filter_cross_domain_candidates(candidates, &domain_map);

        assert!(
            result.is_empty(),
            "Same root domain (rust) should be filtered out even if sub-domains differ"
        );
    }
}
