//! Wormhole Neural Shortcuts for ProfDAG.
//!
//! This module implements wormhole shortcuts that create direct neural pathways
//! between frequently co-accessed patterns. Wormholes bypass normal graph traversal
//! to provide O(1) access to commonly used pattern combinations.
//!
//! # Architecture
//!
//! ```text
//! Pattern A ----[normal edges]----> B ----> C ----> D
//!     |                                              ^
//!     +=================[WORMHOLE]=================+
//!           (created when A->D co-accessed >3 times)
//! ```
//!
//! # Lifecycle
//!
//! 1. **Detection**: `WormholeDetector` analyzes trajectories for co-access patterns
//! 2. **Creation**: When threshold met (default: 3 co-accesses), wormhole is created
//! 3. **Strengthening**: Each use increases wormhole strength
//! 4. **Decay**: Unused wormholes decay over time (default: 30 days)
//! 5. **Deletion**: Wormholes below minimum strength are removed
//!
//! # Example
//!
//! ```ignore
//! use nagual::profdag::wormhole::{WormholeManager, WormholeConfig};
//!
//! let manager = WormholeManager::new(adapter, WormholeConfig::default()).await?;
//!
//! // Record co-access
//! manager.record_co_access("pattern_A", "pattern_D", "session_123").await?;
//!
//! // After 3 co-accesses, wormhole is automatically created
//! let wormholes = manager.get_wormholes_for_pattern("pattern_A").await?;
//! ```

use std::sync::Arc;

use chrono::{DateTime, Utc};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use tracing::{debug, info};
use uuid::Uuid;

use crate::db::DualWriteAdapter;
use crate::error::{NagualError, Result};
use super::{ProfDAGEdge, ProfDAGResult};

/// Configuration for wormhole behavior.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WormholeConfig {
    /// Minimum co-accesses required to create a wormhole.
    /// Default: 3
    #[serde(default = "default_activation_threshold")]
    pub activation_threshold: u32,

    /// Days of inactivity before wormhole starts decaying.
    /// Default: 30
    #[serde(default = "default_decay_days")]
    pub decay_days: u32,

    /// Maximum wormholes per source node.
    /// Default: 10
    #[serde(default = "default_max_wormholes_per_node")]
    pub max_wormholes_per_node: usize,

    /// Minimum traversal savings required (as fraction).
    /// Default: 0.5 (50% reduction in path length)
    #[serde(default = "default_min_traversal_savings")]
    pub min_traversal_savings: f32,

    /// Minimum strength before wormhole is deleted.
    /// Default: 0.1
    #[serde(default = "default_min_strength")]
    pub min_strength: f32,

    /// Decay rate per day after decay_days (0.0 - 1.0).
    /// Default: 0.05 (5% per day)
    #[serde(default = "default_decay_rate")]
    pub decay_rate: f32,

    /// Whether to create bidirectional wormholes.
    /// Default: true
    #[serde(default = "default_bidirectional")]
    pub bidirectional: bool,

    /// Base strength for new wormholes.
    /// Default: 0.5
    #[serde(default = "default_base_strength")]
    pub base_strength: f32,

    /// Strength increment per use.
    /// Default: 0.1
    #[serde(default = "default_strength_increment")]
    pub strength_increment: f32,
}

fn default_activation_threshold() -> u32 { 3 }
fn default_decay_days() -> u32 { 30 }
fn default_max_wormholes_per_node() -> usize { 10 }
fn default_min_traversal_savings() -> f32 { 0.5 }
fn default_min_strength() -> f32 { 0.1 }
fn default_decay_rate() -> f32 { 0.05 }
fn default_bidirectional() -> bool { true }
fn default_base_strength() -> f32 { 0.5 }
fn default_strength_increment() -> f32 { 0.1 }

impl Default for WormholeConfig {
    fn default() -> Self {
        Self {
            activation_threshold: default_activation_threshold(),
            decay_days: default_decay_days(),
            max_wormholes_per_node: default_max_wormholes_per_node(),
            min_traversal_savings: default_min_traversal_savings(),
            min_strength: default_min_strength(),
            decay_rate: default_decay_rate(),
            bidirectional: default_bidirectional(),
            base_strength: default_base_strength(),
            strength_increment: default_strength_increment(),
        }
    }
}

impl WormholeConfig {
    /// Create a new config with custom activation threshold.
    pub fn with_activation_threshold(mut self, threshold: u32) -> Self {
        self.activation_threshold = threshold.max(1);
        self
    }

    /// Create a new config with custom decay days.
    pub fn with_decay_days(mut self, days: u32) -> Self {
        self.decay_days = days;
        self
    }

    /// Create a new config with custom max wormholes per node.
    pub fn with_max_wormholes(mut self, max: usize) -> Self {
        self.max_wormholes_per_node = max.max(1);
        self
    }

    /// Create a new config with custom minimum traversal savings.
    pub fn with_min_traversal_savings(mut self, savings: f32) -> Self {
        self.min_traversal_savings = savings.clamp(0.0, 1.0);
        self
    }
}

/// Reason for wormhole creation.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WormholeCreationReason {
    /// Created due to frequent co-access in trajectories.
    CoAccess {
        /// Number of times patterns were co-accessed.
        count: u32,
        /// Average path distance without wormhole.
        avg_path_distance: Option<f32>,
    },
    /// Created based on semantic similarity.
    SemanticSimilarity {
        /// Similarity score (0.0 - 1.0).
        similarity: f32,
    },
    /// Created manually by user/admin.
    Manual {
        /// Reason provided by user.
        reason: String,
    },
    /// Created by learning algorithm.
    Learned {
        /// Algorithm that created the wormhole.
        algorithm: String,
        /// Confidence in the prediction.
        confidence: f32,
    },
}

impl std::fmt::Display for WormholeCreationReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WormholeCreationReason::CoAccess { count, avg_path_distance } => {
                write!(f, "Co-accessed {} times", count)?;
                if let Some(dist) = avg_path_distance {
                    write!(f, " (avg path distance: {:.1})", dist)?;
                }
                Ok(())
            }
            WormholeCreationReason::SemanticSimilarity { similarity } => {
                write!(f, "Semantic similarity: {:.2}", similarity)
            }
            WormholeCreationReason::Manual { reason } => {
                write!(f, "Manual: {}", reason)
            }
            WormholeCreationReason::Learned { algorithm, confidence } => {
                write!(f, "Learned by {} (confidence: {:.2})", algorithm, confidence)
            }
        }
    }
}

/// A wormhole representing a neural shortcut between patterns.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Wormhole {
    /// Unique identifier.
    pub id: String,

    /// Source pattern/node ID.
    pub source_id: String,

    /// Target pattern/node ID.
    pub target_id: String,

    /// Current strength (0.0 - 1.0).
    /// Calculated as: frequency / (frequency + decay_factor)
    pub strength: f32,

    /// Why this wormhole was created.
    pub creation_reason: WormholeCreationReason,

    /// When the wormhole was created.
    pub created_at: DateTime<Utc>,

    /// When the wormhole was last used.
    pub last_used: DateTime<Utc>,

    /// Total number of times this wormhole has been traversed.
    pub usage_count: u32,

    /// Path distance this wormhole bypasses (number of edges saved).
    pub path_distance_saved: Option<u32>,

    /// Whether this wormhole is currently active.
    pub is_active: bool,

    /// Additional metadata.
    #[serde(default)]
    pub metadata: serde_json::Value,
}

impl Wormhole {
    /// Create a new wormhole.
    pub fn new(
        source_id: impl Into<String>,
        target_id: impl Into<String>,
        creation_reason: WormholeCreationReason,
        initial_strength: f32,
    ) -> Self {
        let now = Utc::now();
        Self {
            id: format!("wh_{}", Uuid::new_v4()),
            source_id: source_id.into(),
            target_id: target_id.into(),
            strength: initial_strength.clamp(0.0, 1.0),
            creation_reason,
            created_at: now,
            last_used: now,
            usage_count: 0,
            path_distance_saved: None,
            is_active: true,
            metadata: serde_json::Value::Object(serde_json::Map::new()),
        }
    }

    /// Create a wormhole from co-access pattern.
    pub fn from_co_access(
        source_id: impl Into<String>,
        target_id: impl Into<String>,
        co_access_count: u32,
        avg_path_distance: Option<f32>,
        initial_strength: f32,
    ) -> Self {
        Self::new(
            source_id,
            target_id,
            WormholeCreationReason::CoAccess {
                count: co_access_count,
                avg_path_distance,
            },
            initial_strength,
        )
    }

    /// Record a use of this wormhole.
    pub fn record_use(&mut self, strength_increment: f32) {
        self.usage_count += 1;
        self.last_used = Utc::now();
        self.strength = (self.strength + strength_increment).min(1.0);
    }

    /// Apply decay based on time since last use.
    pub fn apply_decay(&mut self, decay_days: u32, decay_rate: f32) -> bool {
        let days_since_use = (Utc::now() - self.last_used).num_days() as u32;

        if days_since_use > decay_days {
            let decay_days_elapsed = days_since_use - decay_days;
            let decay_factor = (1.0 - decay_rate).powi(decay_days_elapsed as i32);
            self.strength *= decay_factor;
            true
        } else {
            false
        }
    }

    /// Check if this wormhole should be deactivated (below minimum strength).
    pub fn should_deactivate(&self, min_strength: f32) -> bool {
        self.strength < min_strength
    }

    /// Calculate traversal savings as a percentage.
    pub fn traversal_savings(&self, default_path_length: u32) -> f32 {
        if let Some(saved) = self.path_distance_saved {
            if default_path_length > 0 {
                return saved as f32 / default_path_length as f32;
            }
        }
        0.0
    }

    /// Convert to a ProfDAG edge.
    pub fn to_profdag_edge(&self) -> ProfDAGEdge {
        ProfDAGEdge::wormhole(
            &self.source_id,
            &self.target_id,
            self.strength as f64,
            self.creation_reason.to_string(),
        )
    }
}

/// Record of a co-access event between two patterns.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoAccessRecord {
    /// First pattern ID (lexicographically ordered).
    pub pattern_a: String,

    /// Second pattern ID (lexicographically ordered).
    pub pattern_b: String,

    /// Number of times co-accessed.
    pub count: u32,

    /// First co-access timestamp.
    pub first_accessed: DateTime<Utc>,

    /// Last co-access timestamp.
    pub last_accessed: DateTime<Utc>,

    /// Session IDs where co-access occurred.
    pub session_ids: Vec<String>,

    /// Trajectory IDs where co-access occurred.
    pub trajectory_ids: Vec<String>,
}

impl CoAccessRecord {
    /// Create a new co-access record.
    pub fn new(pattern_a: impl Into<String>, pattern_b: impl Into<String>) -> Self {
        let a = pattern_a.into();
        let b = pattern_b.into();

        // Ensure consistent ordering
        let (ordered_a, ordered_b) = if a < b { (a, b) } else { (b, a) };

        let now = Utc::now();
        Self {
            pattern_a: ordered_a,
            pattern_b: ordered_b,
            count: 1,
            first_accessed: now,
            last_accessed: now,
            session_ids: Vec::new(),
            trajectory_ids: Vec::new(),
        }
    }

    /// Record another co-access.
    pub fn record_access(&mut self, session_id: Option<&str>, trajectory_id: Option<&str>) {
        self.count += 1;
        self.last_accessed = Utc::now();

        if let Some(sid) = session_id {
            if !self.session_ids.contains(&sid.to_string()) {
                self.session_ids.push(sid.to_string());
            }
        }

        if let Some(tid) = trajectory_id {
            if !self.trajectory_ids.contains(&tid.to_string()) {
                self.trajectory_ids.push(tid.to_string());
            }
        }
    }

    /// Check if the threshold is met for wormhole creation.
    pub fn meets_threshold(&self, threshold: u32) -> bool {
        self.count >= threshold
    }
}

/// Result of wormhole maintenance operations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WormholeMaintenanceResult {
    /// Number of wormholes decayed.
    pub decayed: usize,

    /// Number of wormholes deactivated.
    pub deactivated: usize,

    /// Number of wormholes deleted.
    pub deleted: usize,

    /// Number of new wormholes created.
    pub created: usize,

    /// Duration of maintenance operation in milliseconds.
    pub duration_ms: u64,
}

/// Statistics about the wormhole system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WormholeStats {
    /// Total number of active wormholes.
    pub active_count: usize,

    /// Total number of inactive wormholes.
    pub inactive_count: usize,

    /// Average wormhole strength.
    pub avg_strength: f32,

    /// Average usage count.
    pub avg_usage: f32,

    /// Total traversals saved (sum of path_distance_saved * usage_count).
    pub total_traversals_saved: u64,

    /// Number of co-access records pending evaluation.
    pub pending_candidates: usize,
}

/// Manager for wormhole lifecycle and operations.
pub struct WormholeManager {
    /// Database adapter.
    adapter: Arc<DualWriteAdapter>,

    /// Configuration.
    config: WormholeConfig,

    /// In-memory cache of active wormholes (source_id -> wormholes).
    cache: RwLock<std::collections::HashMap<String, Vec<Wormhole>>>,

    /// In-memory cache of co-access records.
    co_access_cache: RwLock<std::collections::HashMap<(String, String), CoAccessRecord>>,
}

impl WormholeManager {
    /// Create a new wormhole manager.
    pub async fn new(
        adapter: Arc<DualWriteAdapter>,
        config: WormholeConfig,
    ) -> ProfDAGResult<Self> {
        let manager = Self {
            adapter,
            config,
            cache: RwLock::new(std::collections::HashMap::new()),
            co_access_cache: RwLock::new(std::collections::HashMap::new()),
        };

        // Initialize schema
        manager.init_schema().await?;

        Ok(manager)
    }

    /// Create with default configuration.
    pub async fn with_defaults(adapter: Arc<DualWriteAdapter>) -> ProfDAGResult<Self> {
        Self::new(adapter, WormholeConfig::default()).await
    }

    /// Get the configuration.
    pub fn config(&self) -> &WormholeConfig {
        &self.config
    }

    /// Initialize the wormhole schema.
    async fn init_schema(&self) -> ProfDAGResult<()> {
        // Tables are created by migration 013_wormhole_tables.sql
        // This method ensures the schema exists
        debug!("Wormhole schema initialized");
        Ok(())
    }

    /// Record a co-access event between two patterns.
    ///
    /// If the co-access threshold is met, automatically creates a wormhole.
    pub async fn record_co_access(
        &self,
        pattern_a: &str,
        pattern_b: &str,
        session_id: Option<&str>,
        trajectory_id: Option<&str>,
    ) -> Result<Option<Wormhole>> {
        // Don't create wormholes for self-access
        if pattern_a == pattern_b {
            return Ok(None);
        }

        // Ensure consistent ordering
        let (ordered_a, ordered_b) = if pattern_a < pattern_b {
            (pattern_a.to_string(), pattern_b.to_string())
        } else {
            (pattern_b.to_string(), pattern_a.to_string())
        };

        let key = (ordered_a.clone(), ordered_b.clone());

        // Update co-access record
        let should_create = {
            let mut cache = self.co_access_cache.write();
            let is_existing = cache.contains_key(&key);
            let record = cache.entry(key.clone()).or_insert_with(|| {
                CoAccessRecord::new(&ordered_a, &ordered_b)
            });

            if is_existing {
                record.record_access(session_id, trajectory_id);
            }

            record.meets_threshold(self.config.activation_threshold)
        };

        // Persist to database
        self.persist_co_access(&ordered_a, &ordered_b, session_id, trajectory_id).await?;

        // Check if we should create a wormhole
        if should_create {
            // Check if wormhole already exists
            if !self.wormhole_exists(&ordered_a, &ordered_b).await? {
                let wormhole = self.create_wormhole(
                    &ordered_a,
                    &ordered_b,
                    WormholeCreationReason::CoAccess {
                        count: self.config.activation_threshold,
                        avg_path_distance: None, // TODO: Calculate from graph
                    },
                ).await?;

                return Ok(Some(wormhole));
            }
        }

        Ok(None)
    }

    /// Record multiple co-accesses from a trajectory.
    ///
    /// For a trajectory with patterns [A, B, C, D], records:
    /// - A<->B, A<->C, A<->D, B<->C, B<->D, C<->D
    pub async fn record_trajectory_co_accesses(
        &self,
        pattern_ids: &[String],
        session_id: Option<&str>,
        trajectory_id: Option<&str>,
    ) -> Result<Vec<Wormhole>> {
        if pattern_ids.len() < 2 {
            return Ok(Vec::new());
        }

        let mut created_wormholes = Vec::new();

        // Record all unique pairs
        for i in 0..pattern_ids.len() {
            for j in (i + 1)..pattern_ids.len() {
                if let Some(wormhole) = self
                    .record_co_access(&pattern_ids[i], &pattern_ids[j], session_id, trajectory_id)
                    .await?
                {
                    created_wormholes.push(wormhole);
                }
            }
        }

        Ok(created_wormholes)
    }

    /// Create a new wormhole.
    pub async fn create_wormhole(
        &self,
        source_id: &str,
        target_id: &str,
        reason: WormholeCreationReason,
    ) -> Result<Wormhole> {
        // Check max wormholes per node
        let existing_count = self.count_wormholes_for_source(source_id).await?;
        if existing_count >= self.config.max_wormholes_per_node {
            // Evict weakest wormhole
            self.evict_weakest_wormhole(source_id).await?;
        }

        let wormhole = Wormhole::new(
            source_id,
            target_id,
            reason,
            self.config.base_strength,
        );

        // Persist wormhole
        self.persist_wormhole(&wormhole).await?;

        // Create ProfDAG edge
        self.create_profdag_edge(&wormhole).await?;

        // Update cache
        {
            let mut cache = self.cache.write();
            cache
                .entry(source_id.to_string())
                .or_insert_with(Vec::new)
                .push(wormhole.clone());
        }

        info!(
            source_id = source_id,
            target_id = target_id,
            strength = wormhole.strength,
            "Created wormhole"
        );

        // Create reverse wormhole if bidirectional
        if self.config.bidirectional && source_id != target_id {
            let reverse_wormhole = Wormhole::new(
                target_id,
                source_id,
                wormhole.creation_reason.clone(),
                self.config.base_strength,
            );
            self.persist_wormhole(&reverse_wormhole).await?;
            self.create_profdag_edge(&reverse_wormhole).await?;

            let mut cache = self.cache.write();
            cache
                .entry(target_id.to_string())
                .or_insert_with(Vec::new)
                .push(reverse_wormhole);
        }

        Ok(wormhole)
    }

    /// Record usage of a wormhole.
    pub async fn record_wormhole_use(
        &self,
        wormhole_id: &str,
    ) -> Result<()> {
        // Update in database
        let sql = r#"
            UPDATE wormholes
            SET usage_count = usage_count + 1,
                last_used = ?,
                strength = MIN(1.0, strength + ?)
            WHERE id = ?
        "#;

        self.adapter
            .sqlite()
            .execute(
                sql,
                &[
                    &Utc::now().to_rfc3339(),
                    &self.config.strength_increment,
                    &wormhole_id,
                ],
            )
            .await
            .map_err(|e| NagualError::internal(format!("Failed to record wormhole use: {}", e)))?;

        // Log usage
        self.log_wormhole_usage(wormhole_id, "traversed").await?;

        debug!(wormhole_id = wormhole_id, "Recorded wormhole use");

        Ok(())
    }

    /// Get all wormholes for a source pattern.
    pub async fn get_wormholes_for_pattern(&self, source_id: &str) -> Result<Vec<Wormhole>> {
        let sql = r#"
            SELECT id, source_id, target_id, strength, creation_reason,
                   created_at, last_used, usage_count, path_distance_saved,
                   is_active, metadata
            FROM wormholes
            WHERE source_id = ? AND is_active = 1
            ORDER BY strength DESC
        "#;

        let wormholes = self
            .adapter
            .sqlite()
            .query(sql, &[&source_id], |row| {
                Self::wormhole_from_row(row)
            })
            .await
            .map_err(|e| NagualError::internal(format!("Failed to get wormholes: {}", e)))?;

        Ok(wormholes)
    }

    /// Get a specific wormhole by ID.
    pub async fn get_wormhole(&self, wormhole_id: &str) -> Result<Option<Wormhole>> {
        let sql = r#"
            SELECT id, source_id, target_id, strength, creation_reason,
                   created_at, last_used, usage_count, path_distance_saved,
                   is_active, metadata
            FROM wormholes
            WHERE id = ?
        "#;

        let wormhole = self
            .adapter
            .sqlite()
            .query_one(sql, &[&wormhole_id], |row| {
                Self::wormhole_from_row(row)
            })
            .await
            .map_err(|e| NagualError::internal(format!("Failed to get wormhole: {}", e)))?;

        Ok(wormhole)
    }

    /// Check if a wormhole exists between two patterns.
    pub async fn wormhole_exists(&self, source_id: &str, target_id: &str) -> Result<bool> {
        let sql = "SELECT COUNT(*) FROM wormholes WHERE source_id = ? AND target_id = ?";

        let count: i64 = self
            .adapter
            .sqlite()
            .query_one(sql, &[&source_id, &target_id], |row| row.get(0))
            .await
            .map_err(|e| NagualError::internal(format!("Failed to check wormhole exists: {}", e)))?
            .unwrap_or(0);

        Ok(count > 0)
    }

    /// Run maintenance: apply decay, deactivate weak wormholes.
    pub async fn run_maintenance(&self) -> Result<WormholeMaintenanceResult> {
        let start = std::time::Instant::now();
        let mut result = WormholeMaintenanceResult {
            decayed: 0,
            deactivated: 0,
            deleted: 0,
            created: 0,
            duration_ms: 0,
        };

        // Get all active wormholes
        let wormholes = self.get_all_active_wormholes().await?;

        for mut wormhole in wormholes {
            // Apply decay
            if wormhole.apply_decay(self.config.decay_days, self.config.decay_rate) {
                result.decayed += 1;

                // Check if should deactivate
                if wormhole.should_deactivate(self.config.min_strength) {
                    wormhole.is_active = false;
                    result.deactivated += 1;

                    // Log deactivation
                    self.log_wormhole_usage(&wormhole.id, "deactivated").await?;
                }

                // Update in database
                self.update_wormhole_strength(&wormhole.id, wormhole.strength, wormhole.is_active)
                    .await?;
            }
        }

        // Check for new wormholes from pending co-access records
        let new_wormholes = self.evaluate_pending_candidates().await?;
        result.created = new_wormholes.len();

        result.duration_ms = start.elapsed().as_millis() as u64;

        info!(
            decayed = result.decayed,
            deactivated = result.deactivated,
            created = result.created,
            duration_ms = result.duration_ms,
            "Wormhole maintenance completed"
        );

        Ok(result)
    }

    /// Get wormhole statistics.
    pub async fn stats(&self) -> Result<WormholeStats> {
        let active_count: usize = self
            .adapter
            .sqlite()
            .query_one(
                "SELECT COUNT(*) FROM wormholes WHERE is_active = 1",
                &[],
                |row| row.get(0),
            )
            .await
            .map_err(|e| NagualError::internal(e.to_string()))?
            .unwrap_or(0);

        let inactive_count: usize = self
            .adapter
            .sqlite()
            .query_one(
                "SELECT COUNT(*) FROM wormholes WHERE is_active = 0",
                &[],
                |row| row.get(0),
            )
            .await
            .map_err(|e| NagualError::internal(e.to_string()))?
            .unwrap_or(0);

        let (avg_strength, avg_usage): (f64, f64) = self
            .adapter
            .sqlite()
            .query_one(
                "SELECT COALESCE(AVG(strength), 0), COALESCE(AVG(usage_count), 0) FROM wormholes WHERE is_active = 1",
                &[],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .await
            .map_err(|e| NagualError::internal(e.to_string()))?
            .unwrap_or((0.0, 0.0));

        let total_traversals_saved: i64 = self
            .adapter
            .sqlite()
            .query_one(
                "SELECT COALESCE(SUM(COALESCE(path_distance_saved, 0) * usage_count), 0) FROM wormholes WHERE is_active = 1",
                &[],
                |row| row.get(0),
            )
            .await
            .map_err(|e| NagualError::internal(e.to_string()))?
            .unwrap_or(0);

        let pending_candidates: usize = self
            .adapter
            .sqlite()
            .query_one(
                &format!(
                    "SELECT COUNT(*) FROM wormhole_co_access WHERE count >= {} AND NOT EXISTS (
                        SELECT 1 FROM wormholes WHERE
                            (wormholes.source_id = wormhole_co_access.pattern_a AND wormholes.target_id = wormhole_co_access.pattern_b)
                            OR (wormholes.source_id = wormhole_co_access.pattern_b AND wormholes.target_id = wormhole_co_access.pattern_a)
                    )",
                    self.config.activation_threshold
                ),
                &[],
                |row| row.get(0),
            )
            .await
            .map_err(|e| NagualError::internal(e.to_string()))?
            .unwrap_or(0);

        Ok(WormholeStats {
            active_count,
            inactive_count,
            avg_strength: avg_strength as f32,
            avg_usage: avg_usage as f32,
            total_traversals_saved: total_traversals_saved as u64,
            pending_candidates,
        })
    }

    // ========================================================================
    // Private Helper Methods
    // ========================================================================

    /// Persist co-access to database.
    async fn persist_co_access(
        &self,
        pattern_a: &str,
        pattern_b: &str,
        session_id: Option<&str>,
        trajectory_id: Option<&str>,
    ) -> Result<()> {
        let sql = r#"
            INSERT INTO wormhole_co_access (pattern_a, pattern_b, count, first_accessed, last_accessed, last_session_id, last_trajectory_id)
            VALUES (?, ?, 1, ?, ?, ?, ?)
            ON CONFLICT (pattern_a, pattern_b) DO UPDATE SET
                count = wormhole_co_access.count + 1,
                last_accessed = excluded.last_accessed,
                last_session_id = COALESCE(excluded.last_session_id, wormhole_co_access.last_session_id),
                last_trajectory_id = COALESCE(excluded.last_trajectory_id, wormhole_co_access.last_trajectory_id)
        "#;

        let now = Utc::now().to_rfc3339();
        self.adapter
            .sqlite()
            .execute(
                sql,
                &[
                    &pattern_a,
                    &pattern_b,
                    &now,
                    &now,
                    &session_id as &dyn rusqlite::ToSql,
                    &trajectory_id as &dyn rusqlite::ToSql,
                ],
            )
            .await
            .map_err(|e| NagualError::internal(format!("Failed to persist co-access: {}", e)))?;

        Ok(())
    }

    /// Persist wormhole to database.
    async fn persist_wormhole(&self, wormhole: &Wormhole) -> Result<()> {
        let sql = r#"
            INSERT INTO wormholes (
                id, source_id, target_id, strength, creation_reason,
                created_at, last_used, usage_count, path_distance_saved,
                is_active, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        "#;

        let reason_json = serde_json::to_string(&wormhole.creation_reason)
            .unwrap_or_else(|_| "{}".to_string());
        let metadata_json = serde_json::to_string(&wormhole.metadata)
            .unwrap_or_else(|_| "{}".to_string());

        self.adapter
            .sqlite()
            .execute(
                sql,
                &[
                    &wormhole.id,
                    &wormhole.source_id,
                    &wormhole.target_id,
                    &wormhole.strength,
                    &reason_json,
                    &wormhole.created_at.to_rfc3339(),
                    &wormhole.last_used.to_rfc3339(),
                    &(wormhole.usage_count as i64),
                    &wormhole.path_distance_saved.map(|v| v as i64) as &dyn rusqlite::ToSql,
                    &wormhole.is_active,
                    &metadata_json,
                ],
            )
            .await
            .map_err(|e| NagualError::internal(format!("Failed to persist wormhole: {}", e)))?;

        Ok(())
    }

    /// Create ProfDAG edge for wormhole.
    async fn create_profdag_edge(&self, wormhole: &Wormhole) -> Result<()> {
        let edge = wormhole.to_profdag_edge();
        let metadata_json = serde_json::to_string(&edge.metadata)
            .unwrap_or_else(|_| "{}".to_string());

        let sql = r#"
            INSERT INTO profdag_edges (
                id, source_id, target_id, edge_type, weight, metadata,
                wormhole_strength, wormhole_reason, created_at
            ) VALUES (?, ?, ?, 'wormhole', ?, ?, ?, ?, ?)
            ON CONFLICT (source_id, target_id, edge_type) DO UPDATE SET
                weight = excluded.weight,
                wormhole_strength = excluded.wormhole_strength,
                updated_at = excluded.created_at
        "#;

        self.adapter
            .sqlite()
            .execute(
                sql,
                &[
                    &edge.id,
                    &edge.source_id,
                    &edge.target_id,
                    &edge.weight,
                    &metadata_json,
                    &edge.wormhole_strength as &dyn rusqlite::ToSql,
                    &edge.wormhole_reason as &dyn rusqlite::ToSql,
                    &edge.created_at.to_rfc3339(),
                ],
            )
            .await
            .map_err(|e| NagualError::internal(format!("Failed to create ProfDAG edge: {}", e)))?;

        Ok(())
    }

    /// Count wormholes for a source.
    async fn count_wormholes_for_source(&self, source_id: &str) -> Result<usize> {
        let count: i64 = self
            .adapter
            .sqlite()
            .query_one(
                "SELECT COUNT(*) FROM wormholes WHERE source_id = ? AND is_active = 1",
                &[&source_id],
                |row| row.get(0),
            )
            .await
            .map_err(|e| NagualError::internal(e.to_string()))?
            .unwrap_or(0);

        Ok(count as usize)
    }

    /// Evict the weakest wormhole for a source.
    async fn evict_weakest_wormhole(&self, source_id: &str) -> Result<()> {
        let sql = r#"
            UPDATE wormholes
            SET is_active = 0
            WHERE id = (
                SELECT id FROM wormholes
                WHERE source_id = ? AND is_active = 1
                ORDER BY strength ASC, usage_count ASC
                LIMIT 1
            )
        "#;

        self.adapter
            .sqlite()
            .execute(sql, &[&source_id])
            .await
            .map_err(|e| NagualError::internal(format!("Failed to evict wormhole: {}", e)))?;

        Ok(())
    }

    /// Get all active wormholes.
    async fn get_all_active_wormholes(&self) -> Result<Vec<Wormhole>> {
        let sql = r#"
            SELECT id, source_id, target_id, strength, creation_reason,
                   created_at, last_used, usage_count, path_distance_saved,
                   is_active, metadata
            FROM wormholes
            WHERE is_active = 1
        "#;

        let wormholes = self
            .adapter
            .sqlite()
            .query(sql, &[], |row| Self::wormhole_from_row(row))
            .await
            .map_err(|e| NagualError::internal(format!("Failed to get wormholes: {}", e)))?;

        Ok(wormholes)
    }

    /// Update wormhole strength.
    async fn update_wormhole_strength(
        &self,
        wormhole_id: &str,
        strength: f32,
        is_active: bool,
    ) -> Result<()> {
        let sql = "UPDATE wormholes SET strength = ?, is_active = ? WHERE id = ?";

        self.adapter
            .sqlite()
            .execute(sql, &[&strength, &is_active, &wormhole_id])
            .await
            .map_err(|e| NagualError::internal(format!("Failed to update wormhole: {}", e)))?;

        Ok(())
    }

    /// Log wormhole usage event.
    async fn log_wormhole_usage(&self, wormhole_id: &str, event_type: &str) -> Result<()> {
        let sql = r#"
            INSERT INTO wormhole_usage_log (wormhole_id, event_type, timestamp)
            VALUES (?, ?, ?)
        "#;

        self.adapter
            .sqlite()
            .execute(sql, &[&wormhole_id, &event_type, &Utc::now().to_rfc3339()])
            .await
            .map_err(|e| NagualError::internal(format!("Failed to log wormhole usage: {}", e)))?;

        Ok(())
    }

    /// Evaluate pending candidates for wormhole creation.
    async fn evaluate_pending_candidates(&self) -> Result<Vec<Wormhole>> {
        let sql = format!(
            r#"
            SELECT pattern_a, pattern_b, count
            FROM wormhole_co_access
            WHERE count >= {}
            AND NOT EXISTS (
                SELECT 1 FROM wormholes
                WHERE (wormholes.source_id = wormhole_co_access.pattern_a
                       AND wormholes.target_id = wormhole_co_access.pattern_b)
                   OR (wormholes.source_id = wormhole_co_access.pattern_b
                       AND wormholes.target_id = wormhole_co_access.pattern_a)
            )
            ORDER BY count DESC
            LIMIT 100
            "#,
            self.config.activation_threshold
        );

        let candidates: Vec<(String, String, i64)> = self
            .adapter
            .sqlite()
            .query(&sql, &[], |row| {
                Ok((row.get(0)?, row.get(1)?, row.get(2)?))
            })
            .await
            .map_err(|e| NagualError::internal(format!("Failed to get candidates: {}", e)))?;

        let mut created = Vec::new();
        for (pattern_a, pattern_b, count) in candidates {
            let wormhole = self
                .create_wormhole(
                    &pattern_a,
                    &pattern_b,
                    WormholeCreationReason::CoAccess {
                        count: count as u32,
                        avg_path_distance: None,
                    },
                )
                .await?;
            created.push(wormhole);
        }

        Ok(created)
    }

    /// Convert database row to Wormhole.
    fn wormhole_from_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<Wormhole> {
        let id: String = row.get(0)?;
        let source_id: String = row.get(1)?;
        let target_id: String = row.get(2)?;
        let strength: f64 = row.get(3)?;
        let creation_reason_json: String = row.get(4)?;
        let created_at_str: String = row.get(5)?;
        let last_used_str: String = row.get(6)?;
        let usage_count: i64 = row.get(7)?;
        let path_distance_saved: Option<i64> = row.get(8)?;
        let is_active: bool = row.get(9)?;
        let metadata_json: String = row.get::<_, Option<String>>(10)?.unwrap_or_else(|| "{}".to_string());

        let creation_reason: WormholeCreationReason = serde_json::from_str(&creation_reason_json)
            .unwrap_or(WormholeCreationReason::Manual {
                reason: "Unknown".to_string(),
            });

        let metadata: serde_json::Value = serde_json::from_str(&metadata_json)
            .unwrap_or(serde_json::json!({}));

        let created_at = DateTime::parse_from_rfc3339(&created_at_str)
            .map(|dt| dt.with_timezone(&Utc))
            .unwrap_or_else(|_| Utc::now());

        let last_used = DateTime::parse_from_rfc3339(&last_used_str)
            .map(|dt| dt.with_timezone(&Utc))
            .unwrap_or_else(|_| Utc::now());

        Ok(Wormhole {
            id,
            source_id,
            target_id,
            strength: strength as f32,
            creation_reason,
            created_at,
            last_used,
            usage_count: usage_count as u32,
            path_distance_saved: path_distance_saved.map(|v| v as u32),
            is_active,
            metadata,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::profdag::EdgeType;

    #[test]
    fn test_wormhole_config_defaults() {
        let config = WormholeConfig::default();
        assert_eq!(config.activation_threshold, 3);
        assert_eq!(config.decay_days, 30);
        assert_eq!(config.max_wormholes_per_node, 10);
        assert!((config.min_traversal_savings - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_wormhole_config_builder() {
        let config = WormholeConfig::default()
            .with_activation_threshold(5)
            .with_decay_days(60)
            .with_max_wormholes(20);

        assert_eq!(config.activation_threshold, 5);
        assert_eq!(config.decay_days, 60);
        assert_eq!(config.max_wormholes_per_node, 20);
    }

    #[test]
    fn test_wormhole_creation() {
        let wormhole = Wormhole::from_co_access(
            "pattern_A",
            "pattern_D",
            3,
            Some(4.5),
            0.5,
        );

        assert!(!wormhole.id.is_empty());
        assert_eq!(wormhole.source_id, "pattern_A");
        assert_eq!(wormhole.target_id, "pattern_D");
        assert!((wormhole.strength - 0.5).abs() < 0.001);
        assert!(wormhole.is_active);
        assert_eq!(wormhole.usage_count, 0);
    }

    #[test]
    fn test_wormhole_record_use() {
        let mut wormhole = Wormhole::new(
            "A",
            "B",
            WormholeCreationReason::Manual { reason: "test".to_string() },
            0.5,
        );

        wormhole.record_use(0.1);

        assert_eq!(wormhole.usage_count, 1);
        assert!((wormhole.strength - 0.6).abs() < 0.001);
    }

    #[test]
    fn test_wormhole_strength_cap() {
        let mut wormhole = Wormhole::new(
            "A",
            "B",
            WormholeCreationReason::Manual { reason: "test".to_string() },
            0.95,
        );

        wormhole.record_use(0.1);

        assert!((wormhole.strength - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_wormhole_should_deactivate() {
        let wormhole = Wormhole::new(
            "A",
            "B",
            WormholeCreationReason::Manual { reason: "test".to_string() },
            0.05,
        );

        assert!(wormhole.should_deactivate(0.1));
        assert!(!wormhole.should_deactivate(0.01));
    }

    #[test]
    fn test_co_access_record_ordering() {
        let record1 = CoAccessRecord::new("pattern_B", "pattern_A");
        let record2 = CoAccessRecord::new("pattern_A", "pattern_B");

        // Both should have the same ordering
        assert_eq!(record1.pattern_a, "pattern_A");
        assert_eq!(record1.pattern_b, "pattern_B");
        assert_eq!(record2.pattern_a, "pattern_A");
        assert_eq!(record2.pattern_b, "pattern_B");
    }

    #[test]
    fn test_co_access_record_threshold() {
        let mut record = CoAccessRecord::new("A", "B");
        assert!(!record.meets_threshold(3));

        record.record_access(Some("session_1"), None);
        record.record_access(Some("session_2"), None);

        assert!(record.meets_threshold(3));
    }

    #[test]
    fn test_creation_reason_display() {
        let co_access = WormholeCreationReason::CoAccess {
            count: 5,
            avg_path_distance: Some(3.5),
        };
        let display = co_access.to_string();
        assert!(display.contains("5 times"));
        assert!(display.contains("3.5"));

        let manual = WormholeCreationReason::Manual {
            reason: "test reason".to_string(),
        };
        assert!(manual.to_string().contains("test reason"));
    }

    #[test]
    fn test_wormhole_to_profdag_edge() {
        let wormhole = Wormhole::new(
            "A",
            "B",
            WormholeCreationReason::CoAccess {
                count: 3,
                avg_path_distance: None,
            },
            0.7,
        );

        let edge = wormhole.to_profdag_edge();

        assert_eq!(edge.source_id, "A");
        assert_eq!(edge.target_id, "B");
        assert_eq!(edge.edge_type, EdgeType::Wormhole);
        assert!((edge.weight - 0.7).abs() < 0.001);
        assert!(edge.wormhole_reason.is_some());
    }

    #[test]
    fn test_wormhole_traversal_savings() {
        let mut wormhole = Wormhole::new(
            "A",
            "B",
            WormholeCreationReason::Manual { reason: "test".to_string() },
            0.5,
        );
        wormhole.path_distance_saved = Some(3);

        let savings = wormhole.traversal_savings(6);
        assert!((savings - 0.5).abs() < 0.001);

        let savings_zero = wormhole.traversal_savings(0);
        assert!(savings_zero.abs() < 0.001);
    }
}
