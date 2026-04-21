//! Trajectory Recorder for ProfDAG integration.
//!
//! This module provides recording and replay capabilities for reasoning trajectories,
//! linking them to ProfDAG nodes via 'leads_to' edges.
//!
//! # Architecture
//!
//! ```text
//! Query -> TrajectoryRecorder.start()
//!            |
//!          record_step() x N
//!            |
//!          complete() -> ProfDAG node + edges
//! ```
//!
//! # Example
//!
//! ```ignore
//! use nagual::profdag::trajectory_recorder::{TrajectoryRecorder, RecorderConfig};
//! use nagual::learning::trajectory::TrajectoryStep;
//!
//! let recorder = TrajectoryRecorder::new();
//!
//! // Start recording
//! let trajectory_id = recorder.start("How to optimize caching?", Some("session-123".to_string()));
//!
//! // Record steps
//! recorder.record_step(&trajectory_id, TrajectoryStep::pattern_retrieval(
//!     vec!["pat_1".into()],
//!     "caching",
//!     0.9,
//! )).unwrap();
//!
//! // Complete and link to ProfDAG
//! let result = recorder.complete(&trajectory_id, Outcome::Success, 0.9).unwrap();
//! println!("ProfDAG node: {}", result.profdag_node_id);
//! ```

use std::collections::HashMap;
use std::io::{Read as IoRead, Write as IoWrite};
use std::sync::Arc;

use chrono::{DateTime, Utc};
use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use flate2::Compression;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use tracing::{debug, info, instrument, warn};

use crate::error::{NagualError, Result};
use crate::learning::Outcome;
use crate::learning::trajectory::{
    CompactTrajectory, Trajectory, TrajectoryBuilder, TrajectoryId, TrajectoryStats,
    TrajectoryStep,
};
use crate::reasoning_bank::pattern::PatternId;

use super::storage::ProfDAGStorage;
use super::{ProfDAGEdge, ProfDAGNode};

/// Configuration for the trajectory recorder.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecorderConfig {
    /// Maximum steps per trajectory before auto-complete.
    #[serde(default = "default_max_steps")]
    pub max_steps: usize,

    /// Whether to compress trajectories before storage.
    #[serde(default = "default_compress")]
    pub compress: bool,

    /// Compression level (1-9, higher = more compression).
    #[serde(default = "default_compression_level")]
    pub compression_level: u32,

    /// Whether to create ProfDAG edges automatically.
    #[serde(default = "default_auto_edges")]
    pub auto_create_edges: bool,

    /// Minimum reward threshold to create 'leads_to' edges.
    #[serde(default = "default_min_reward")]
    pub min_reward_for_edges: f32,

    /// Maximum trajectories to keep in memory cache.
    #[serde(default = "default_cache_size")]
    pub cache_size: usize,

    /// Whether to record failed trajectories.
    #[serde(default = "default_record_failures")]
    pub record_failures: bool,

    /// Timeout for incomplete trajectories (seconds).
    #[serde(default = "default_timeout")]
    pub timeout_secs: u64,
}

fn default_max_steps() -> usize {
    50
}

fn default_compress() -> bool {
    true
}

fn default_compression_level() -> u32 {
    6
}

fn default_auto_edges() -> bool {
    true
}

fn default_min_reward() -> f32 {
    0.6
}

fn default_cache_size() -> usize {
    1000
}

fn default_record_failures() -> bool {
    true
}

fn default_timeout() -> u64 {
    3600 // 1 hour
}

impl Default for RecorderConfig {
    fn default() -> Self {
        Self {
            max_steps: default_max_steps(),
            compress: default_compress(),
            compression_level: default_compression_level(),
            auto_create_edges: default_auto_edges(),
            min_reward_for_edges: default_min_reward(),
            cache_size: default_cache_size(),
            record_failures: default_record_failures(),
            timeout_secs: default_timeout(),
        }
    }
}

impl RecorderConfig {
    /// Create a new config with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set max steps.
    pub fn with_max_steps(mut self, max_steps: usize) -> Self {
        self.max_steps = max_steps;
        self
    }

    /// Enable/disable compression.
    pub fn with_compression(mut self, compress: bool) -> Self {
        self.compress = compress;
        self
    }

    /// Set compression level.
    pub fn with_compression_level(mut self, level: u32) -> Self {
        self.compression_level = level.clamp(1, 9);
        self
    }

    /// Enable/disable auto edge creation.
    pub fn with_auto_edges(mut self, auto: bool) -> Self {
        self.auto_create_edges = auto;
        self
    }

    /// Set minimum reward for edge creation.
    pub fn with_min_reward(mut self, min_reward: f32) -> Self {
        self.min_reward_for_edges = min_reward.clamp(0.0, 1.0);
        self
    }
}

/// Result of completing a trajectory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompleteResult {
    /// The trajectory ID.
    pub trajectory_id: TrajectoryId,

    /// The created ProfDAG node ID.
    pub profdag_node_id: String,

    /// Number of 'leads_to' edges created.
    pub edges_created: usize,

    /// Final outcome.
    pub outcome: Outcome,

    /// Final reward.
    pub reward: f32,

    /// Total duration in milliseconds.
    pub duration_ms: u64,

    /// Number of steps recorded.
    pub step_count: usize,

    /// Whether the trajectory was compressed.
    pub compressed: bool,

    /// Compressed size in bytes (if compressed).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub compressed_size: Option<usize>,
}

/// Result of replaying a trajectory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplayResult {
    /// The trajectory ID.
    pub trajectory_id: TrajectoryId,

    /// The replayed trajectory.
    pub trajectory: Trajectory,

    /// Pattern IDs encountered.
    pub pattern_ids: Vec<PatternId>,

    /// Step summaries.
    pub step_summaries: Vec<StepSummary>,

    /// Total replay time in milliseconds.
    pub replay_time_ms: u64,
}

/// Summary of a trajectory step for replay.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepSummary {
    /// Step index (0-based).
    pub index: usize,

    /// Step type.
    pub step_type: String,

    /// Decision/action taken.
    pub decision: String,

    /// Confidence level.
    pub confidence: f32,

    /// Pattern IDs involved.
    pub pattern_ids: Vec<String>,

    /// Duration in milliseconds.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duration_ms: Option<u64>,
}

/// Active recording session for a trajectory.
#[derive(Debug)]
struct ActiveRecording {
    trajectory: Trajectory,
    #[allow(dead_code)]
    started_at: DateTime<Utc>,
    last_activity: DateTime<Utc>,
}

impl ActiveRecording {
    fn new(trajectory: Trajectory) -> Self {
        let now = Utc::now();
        Self {
            trajectory,
            started_at: now,
            last_activity: now,
        }
    }

    fn touch(&mut self) {
        self.last_activity = Utc::now();
    }
}

/// Trajectory recorder with ProfDAG integration.
///
/// The recorder manages:
/// - Active recording sessions
/// - Trajectory storage and compression
/// - ProfDAG node and edge creation
/// - Replay capabilities
pub struct TrajectoryRecorder {
    /// Configuration.
    config: RecorderConfig,

    /// Active recordings (trajectory_id -> recording).
    active: RwLock<HashMap<String, ActiveRecording>>,

    /// Completed trajectories cache (for replay).
    completed: RwLock<lru::LruCache<String, CompactTrajectory>>,

    /// Statistics.
    stats: RwLock<TrajectoryStats>,

    /// Full trajectory storage (for replay).
    trajectory_store: RwLock<HashMap<String, Vec<u8>>>,

    /// Optional ProfDAG storage backend for persisting nodes and edges.
    /// When `None`, the recorder operates in memory-only mode (backward compatible).
    storage: Option<Arc<ProfDAGStorage>>,
}

impl TrajectoryRecorder {
    /// Create a new trajectory recorder with default config.
    pub fn new() -> Self {
        Self::with_config(RecorderConfig::default())
    }

    /// Create a new trajectory recorder with custom config.
    pub fn with_config(config: RecorderConfig) -> Self {
        Self {
            active: RwLock::new(HashMap::new()),
            completed: RwLock::new(lru::LruCache::new(
                std::num::NonZeroUsize::new(config.cache_size).unwrap_or(
                    std::num::NonZeroUsize::new(1000).unwrap(),
                ),
            )),
            stats: RwLock::new(TrajectoryStats::new()),
            trajectory_store: RwLock::new(HashMap::new()),
            storage: None,
            config,
        }
    }

    /// Create a new trajectory recorder with custom config and ProfDAG storage.
    ///
    /// When storage is provided, `complete_async()` will persist trajectory nodes
    /// and pattern edges to the ProfDAG database.
    pub fn with_storage(config: RecorderConfig, storage: Arc<ProfDAGStorage>) -> Self {
        Self {
            active: RwLock::new(HashMap::new()),
            completed: RwLock::new(lru::LruCache::new(
                std::num::NonZeroUsize::new(config.cache_size).unwrap_or(
                    std::num::NonZeroUsize::new(1000).unwrap(),
                ),
            )),
            stats: RwLock::new(TrajectoryStats::new()),
            trajectory_store: RwLock::new(HashMap::new()),
            storage: Some(storage),
            config,
        }
    }

    /// Returns whether this recorder has ProfDAG storage attached.
    pub fn has_storage(&self) -> bool {
        self.storage.is_some()
    }

    /// Get the configuration.
    pub fn config(&self) -> &RecorderConfig {
        &self.config
    }

    /// Get current statistics.
    pub fn stats(&self) -> TrajectoryStats {
        self.stats.read().clone()
    }

    /// Get count of active recordings.
    pub fn active_count(&self) -> usize {
        self.active.read().len()
    }

    /// Start recording a new trajectory.
    ///
    /// # Arguments
    ///
    /// * `query` - The initial query/context
    /// * `session_id` - Optional session ID
    ///
    /// # Returns
    ///
    /// The new trajectory ID
    pub fn start(
        &self,
        query: impl Into<String>,
        session_id: Option<String>,
    ) -> TrajectoryId {
        self.start_with_agent(query, session_id, None)
    }

    /// Start recording a new trajectory with agent ID.
    pub fn start_with_agent(
        &self,
        query: impl Into<String>,
        session_id: Option<String>,
        agent_id: Option<String>,
    ) -> TrajectoryId {
        let query_str = query.into();

        let mut builder = TrajectoryBuilder::new()
            .query(query_str.clone());

        if let Some(sid) = session_id {
            builder = builder.session_id(sid);
        }

        if let Some(aid) = agent_id {
            builder = builder.agent_id(aid);
        }

        let trajectory = builder.build();
        let id = trajectory.id.clone();

        let recording = ActiveRecording::new(trajectory);
        self.active.write().insert(id.as_str().to_string(), recording);

        debug!(trajectory_id = %id, "Started trajectory recording");
        id
    }

    /// Record a step in an active trajectory.
    ///
    /// # Arguments
    ///
    /// * `trajectory_id` - The trajectory to record in
    /// * `step` - The step to record
    ///
    /// # Returns
    ///
    /// The step index (0-based)
    #[instrument(skip(self, step), fields(trajectory_id = %trajectory_id))]
    pub fn record_step(
        &self,
        trajectory_id: &TrajectoryId,
        step: TrajectoryStep,
    ) -> Result<usize> {
        let mut active = self.active.write();

        let recording = active
            .get_mut(trajectory_id.as_str())
            .ok_or_else(|| NagualError::Internal {
                message: format!("Trajectory not found: {}", trajectory_id),
            })?;

        // Check max steps
        if recording.trajectory.step_count() >= self.config.max_steps {
            return Err(NagualError::Internal {
                message: format!(
                    "Trajectory {} exceeded max steps ({})",
                    trajectory_id, self.config.max_steps
                ),
            });
        }

        recording.trajectory.add_step(step);
        recording.touch();

        let step_index = recording.trajectory.step_count() - 1;
        debug!(
            trajectory_id = %trajectory_id,
            step_index = step_index,
            "Recorded trajectory step"
        );

        Ok(step_index)
    }

    /// Complete a trajectory and create ProfDAG nodes/edges.
    ///
    /// # Arguments
    ///
    /// * `trajectory_id` - The trajectory to complete
    /// * `outcome` - The final outcome
    /// * `reward` - The total reward (0.0 - 1.0)
    ///
    /// # Returns
    ///
    /// Completion result with ProfDAG node ID
    #[instrument(skip(self), fields(trajectory_id = %trajectory_id, outcome = %outcome))]
    pub fn complete(
        &self,
        trajectory_id: &TrajectoryId,
        outcome: Outcome,
        reward: f32,
    ) -> Result<CompleteResult> {
        let recording = self
            .active
            .write()
            .remove(trajectory_id.as_str())
            .ok_or_else(|| NagualError::Internal {
                message: format!("Trajectory not found: {}", trajectory_id),
            })?;

        let mut trajectory = recording.trajectory;
        trajectory.set_outcome(outcome, reward);

        // Skip recording failures if configured
        if !self.config.record_failures && !trajectory.success {
            debug!(
                trajectory_id = %trajectory_id,
                "Skipping failed trajectory (record_failures=false)"
            );
            return Ok(CompleteResult {
                trajectory_id: trajectory_id.clone(),
                profdag_node_id: String::new(),
                edges_created: 0,
                outcome,
                reward,
                duration_ms: trajectory.total_duration_ms.unwrap_or(0),
                step_count: trajectory.step_count(),
                compressed: false,
                compressed_size: None,
            });
        }

        // Create ProfDAG node ID
        let profdag_node_id = format!("traj_node_{}", trajectory.id.as_str());

        // Create edges between pattern nodes
        let mut edges_created = 0;
        if self.config.auto_create_edges && reward >= self.config.min_reward_for_edges {
            edges_created = self.create_pattern_edges(&trajectory);
        }

        // Link trajectory to ProfDAG node
        trajectory.link_profdag_node(&profdag_node_id);

        // Update stats
        self.stats.write().record(&trajectory);

        // Compress and store
        let (compressed, compressed_size) = self.store_trajectory(&trajectory)?;

        // Add to completed cache
        self.completed
            .write()
            .put(trajectory_id.as_str().to_string(), trajectory.to_compact());

        info!(
            trajectory_id = %trajectory_id,
            profdag_node_id = %profdag_node_id,
            edges_created = edges_created,
            duration_ms = trajectory.total_duration_ms.unwrap_or(0),
            "Completed trajectory recording"
        );

        Ok(CompleteResult {
            trajectory_id: trajectory_id.clone(),
            profdag_node_id,
            edges_created,
            outcome,
            reward,
            duration_ms: trajectory.total_duration_ms.unwrap_or(0),
            step_count: trajectory.step_count(),
            compressed,
            compressed_size,
        })
    }

    /// Complete a trajectory and persist the ProfDAG node and edges to storage.
    ///
    /// This is the async counterpart of `complete()`. When storage is attached,
    /// it creates a real `ProfDAGNode` (type: trajectory) and inserts it via
    /// `ProfDAGStorage::insert_node()`. It also creates `leads_to` edges between
    /// consecutive patterns in the trajectory and inserts them via
    /// `ProfDAGStorage::insert_edge()`.
    ///
    /// When no storage is attached, this falls back to the same behavior as the
    /// synchronous `complete()` method.
    ///
    /// # Arguments
    ///
    /// * `trajectory_id` - The trajectory to complete
    /// * `outcome` - The final outcome
    /// * `reward` - The total reward (0.0 - 1.0)
    ///
    /// # Returns
    ///
    /// Completion result with the persisted ProfDAG node ID
    #[instrument(skip(self), fields(trajectory_id = %trajectory_id, outcome = %outcome))]
    pub async fn complete_async(
        &self,
        trajectory_id: &TrajectoryId,
        outcome: Outcome,
        reward: f32,
    ) -> Result<CompleteResult> {
        let recording = self
            .active
            .write()
            .remove(trajectory_id.as_str())
            .ok_or_else(|| NagualError::Internal {
                message: format!("Trajectory not found: {}", trajectory_id),
            })?;

        let mut trajectory = recording.trajectory;
        trajectory.set_outcome(outcome, reward);

        // Skip recording failures if configured
        if !self.config.record_failures && !trajectory.success {
            debug!(
                trajectory_id = %trajectory_id,
                "Skipping failed trajectory (record_failures=false)"
            );
            return Ok(CompleteResult {
                trajectory_id: trajectory_id.clone(),
                profdag_node_id: String::new(),
                edges_created: 0,
                outcome,
                reward,
                duration_ms: trajectory.total_duration_ms.unwrap_or(0),
                step_count: trajectory.step_count(),
                compressed: false,
                compressed_size: None,
            });
        }

        // Build the ProfDAG trajectory node
        let content = format!(
            "Trajectory: {} ({} steps, reward={:.2})",
            trajectory.query.as_deref().unwrap_or("unknown"),
            trajectory.step_count(),
            reward,
        );
        let mut node = ProfDAGNode::trajectory(content)
            .with_confidence(reward)
            .with_importance(reward * 0.8)
            .with_source("trajectory", trajectory.id.as_str());
        if let Some(ref sid) = trajectory.session_id {
            node = node.with_session_id(sid.clone());
        }
        if let Some(ref aid) = trajectory.agent_id {
            node = node.with_agent_id(aid.clone());
        }

        let profdag_node_id;

        // Collect pattern edges
        let pending_edges = if self.config.auto_create_edges
            && reward >= self.config.min_reward_for_edges
        {
            self.collect_pattern_edges(&trajectory, &node.id)
        } else {
            Vec::new()
        };
        let edges_created = pending_edges.len();

        // Persist to storage if available
        if let Some(ref storage) = self.storage {
            // Insert the trajectory node
            profdag_node_id = storage.insert_node(&node).await.map_err(|e| {
                NagualError::Internal {
                    message: format!("Failed to insert ProfDAG node: {}", e),
                }
            })?;

            // Insert all pattern edges
            for edge in &pending_edges {
                if let Err(e) = storage.insert_edge(edge).await {
                    warn!(
                        edge_id = %edge.id,
                        source = %edge.source_id,
                        target = %edge.target_id,
                        error = %e,
                        "Failed to insert ProfDAG edge (continuing)"
                    );
                }
            }
        } else {
            // No storage: generate an in-memory node ID (same as sync complete)
            profdag_node_id = node.id.clone();
        }

        // Also count via the old stub path for backward-compatible edge count
        // (the sync create_pattern_edges is not called; we use pending_edges above)

        // Link trajectory to ProfDAG node
        trajectory.link_profdag_node(&profdag_node_id);

        // Update stats
        self.stats.write().record(&trajectory);

        // Compress and store in local memory
        let (compressed, compressed_size) = self.store_trajectory(&trajectory)?;

        // Add to completed cache
        self.completed
            .write()
            .put(trajectory_id.as_str().to_string(), trajectory.to_compact());

        info!(
            trajectory_id = %trajectory_id,
            profdag_node_id = %profdag_node_id,
            edges_created = edges_created,
            persisted = self.storage.is_some(),
            duration_ms = trajectory.total_duration_ms.unwrap_or(0),
            "Completed trajectory recording (async)"
        );

        Ok(CompleteResult {
            trajectory_id: trajectory_id.clone(),
            profdag_node_id,
            edges_created,
            outcome,
            reward,
            duration_ms: trajectory.total_duration_ms.unwrap_or(0),
            step_count: trajectory.step_count(),
            compressed,
            compressed_size,
        })
    }

    /// Abort an active trajectory without saving.
    pub fn abort(&self, trajectory_id: &TrajectoryId) -> bool {
        let removed = self.active.write().remove(trajectory_id.as_str()).is_some();
        if removed {
            debug!(trajectory_id = %trajectory_id, "Aborted trajectory recording");
        }
        removed
    }

    /// Get an active trajectory (for inspection).
    pub fn get_active(&self, trajectory_id: &TrajectoryId) -> Option<Trajectory> {
        self.active
            .read()
            .get(trajectory_id.as_str())
            .map(|r| r.trajectory.clone())
    }

    /// Check if a trajectory is currently being recorded.
    pub fn is_active(&self, trajectory_id: &TrajectoryId) -> bool {
        self.active.read().contains_key(trajectory_id.as_str())
    }

    /// Replay a completed trajectory.
    ///
    /// # Arguments
    ///
    /// * `trajectory_id` - The trajectory to replay
    ///
    /// # Returns
    ///
    /// Replay result with step summaries
    #[instrument(skip(self), fields(trajectory_id = %trajectory_id))]
    pub fn replay(&self, trajectory_id: &TrajectoryId) -> Result<ReplayResult> {
        let start_time = std::time::Instant::now();

        // Try to load from store
        let trajectory = self.load_trajectory(trajectory_id)?;

        let step_summaries: Vec<StepSummary> = trajectory
            .steps
            .iter()
            .enumerate()
            .map(|(index, step)| StepSummary {
                index,
                step_type: step.step_type.as_str().to_string(),
                decision: step.decision.clone(),
                confidence: step.confidence,
                pattern_ids: step.pattern_ids.iter().map(|p| p.as_str().to_string()).collect(),
                duration_ms: step.duration_ms,
            })
            .collect();

        let pattern_ids = trajectory.all_pattern_ids();
        let replay_time_ms = start_time.elapsed().as_millis() as u64;

        debug!(
            trajectory_id = %trajectory_id,
            step_count = step_summaries.len(),
            replay_time_ms = replay_time_ms,
            "Replayed trajectory"
        );

        Ok(ReplayResult {
            trajectory_id: trajectory_id.clone(),
            trajectory,
            pattern_ids,
            step_summaries,
            replay_time_ms,
        })
    }

    /// Get a compact representation of a completed trajectory.
    pub fn get_compact(&self, trajectory_id: &TrajectoryId) -> Option<CompactTrajectory> {
        self.completed
            .write()
            .get(trajectory_id.as_str())
            .cloned()
    }

    /// Clean up timed-out active recordings.
    pub fn cleanup_timed_out(&self) -> usize {
        let now = Utc::now();
        let timeout = chrono::Duration::seconds(self.config.timeout_secs as i64);

        let mut active = self.active.write();
        let timed_out: Vec<String> = active
            .iter()
            .filter(|(_, recording)| now - recording.last_activity > timeout)
            .map(|(id, _)| id.clone())
            .collect();

        let count = timed_out.len();
        for id in timed_out {
            active.remove(&id);
            warn!(trajectory_id = %id, "Cleaned up timed-out trajectory");
        }

        count
    }

    /// Create 'leads_to' edges between patterns in a trajectory.
    fn create_pattern_edges(&self, trajectory: &Trajectory) -> usize {
        let mut edges_created = 0;

        // Get pattern IDs in order of appearance
        let mut pattern_sequence: Vec<PatternId> = Vec::new();
        for step in &trajectory.steps {
            for pattern_id in &step.pattern_ids {
                if pattern_sequence.last() != Some(pattern_id) {
                    pattern_sequence.push(pattern_id.clone());
                }
            }
        }

        // Create edges between consecutive patterns
        for window in pattern_sequence.windows(2) {
            let source = &window[0];
            let target = &window[1];

            // Note: In a real implementation, this would call the graph storage
            // to create the edge. For now, we just count.
            debug!(
                source = %source,
                target = %target,
                "Would create leads_to edge"
            );
            edges_created += 1;
        }

        edges_created
    }

    /// Collect `leads_to` edges between consecutive patterns in a trajectory.
    ///
    /// Unlike `create_pattern_edges`, this method returns the actual `ProfDAGEdge`
    /// objects so they can be persisted via `ProfDAGStorage::insert_edge()`.
    ///
    /// The `trajectory_node_id` is used as the source for a `derived_from` style
    /// link -- but the primary edges are `leads_to` between pattern nodes.
    fn collect_pattern_edges(
        &self,
        trajectory: &Trajectory,
        _trajectory_node_id: &str,
    ) -> Vec<ProfDAGEdge> {
        let mut edges = Vec::new();

        // Get pattern IDs in order of appearance (deduplicate consecutive)
        let mut pattern_sequence: Vec<PatternId> = Vec::new();
        for step in &trajectory.steps {
            for pattern_id in &step.pattern_ids {
                if pattern_sequence.last() != Some(pattern_id) {
                    pattern_sequence.push(pattern_id.clone());
                }
            }
        }

        // Create leads_to edges between consecutive patterns
        for window in pattern_sequence.windows(2) {
            let source = &window[0];
            let target = &window[1];

            let weight = trajectory.total_reward as f64;
            let edge = ProfDAGEdge::leads_to(
                source.as_str(),
                target.as_str(),
                weight,
            );

            debug!(
                edge_id = %edge.id,
                source = %source,
                target = %target,
                weight = weight,
                "Collected leads_to edge for storage"
            );
            edges.push(edge);
        }

        edges
    }

    /// Store a trajectory with optional compression.
    fn store_trajectory(&self, trajectory: &Trajectory) -> Result<(bool, Option<usize>)> {
        let json = serde_json::to_vec(trajectory).map_err(|e| NagualError::Internal {
            message: format!("Failed to serialize trajectory: {}", e),
        })?;

        let (data, compressed, compressed_size) = if self.config.compress {
            let compressed = self.compress_data(&json)?;
            let size = compressed.len();
            (compressed, true, Some(size))
        } else {
            (json, false, None)
        };

        self.trajectory_store
            .write()
            .insert(trajectory.id.as_str().to_string(), data);

        Ok((compressed, compressed_size))
    }

    /// Load a trajectory from storage.
    fn load_trajectory(&self, trajectory_id: &TrajectoryId) -> Result<Trajectory> {
        let store = self.trajectory_store.read();
        let data = store
            .get(trajectory_id.as_str())
            .ok_or_else(|| NagualError::Internal {
                message: format!("Trajectory not found: {}", trajectory_id),
            })?;

        let json = if self.config.compress {
            self.decompress_data(data)?
        } else {
            data.clone()
        };

        serde_json::from_slice(&json).map_err(|e| NagualError::Internal {
            message: format!("Failed to deserialize trajectory: {}", e),
        })
    }

    /// Compress data using gzip via flate2.
    fn compress_data(&self, data: &[u8]) -> Result<Vec<u8>> {
        let mut encoder = GzEncoder::new(
            Vec::new(),
            Compression::new(self.config.compression_level),
        );
        encoder.write_all(data).map_err(|e| NagualError::Internal {
            message: format!("Compression failed: {}", e),
        })?;
        encoder.finish().map_err(|e| NagualError::Internal {
            message: format!("Compression finish failed: {}", e),
        })
    }

    /// Decompress gzip data via flate2.
    fn decompress_data(&self, data: &[u8]) -> Result<Vec<u8>> {
        let mut decoder = GzDecoder::new(data);
        let mut decompressed = Vec::new();
        decoder
            .read_to_end(&mut decompressed)
            .map_err(|e| NagualError::Internal {
                message: format!("Decompression failed: {}", e),
            })?;
        Ok(decompressed)
    }
}

impl Default for TrajectoryRecorder {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for trajectory recording with fluent API.
pub struct RecordingSession {
    recorder: Arc<TrajectoryRecorder>,
    trajectory_id: TrajectoryId,
}

impl RecordingSession {
    /// Create a new recording session.
    pub fn new(recorder: Arc<TrajectoryRecorder>, query: impl Into<String>) -> Self {
        let trajectory_id = recorder.start(query, None);
        Self {
            recorder,
            trajectory_id,
        }
    }

    /// Create a new recording session with session ID.
    pub fn with_session(
        recorder: Arc<TrajectoryRecorder>,
        query: impl Into<String>,
        session_id: impl Into<String>,
    ) -> Self {
        let trajectory_id = recorder.start(query, Some(session_id.into()));
        Self {
            recorder,
            trajectory_id,
        }
    }

    /// Get the trajectory ID.
    pub fn trajectory_id(&self) -> &TrajectoryId {
        &self.trajectory_id
    }

    /// Record a step.
    pub fn record(&self, step: TrajectoryStep) -> Result<usize> {
        self.recorder.record_step(&self.trajectory_id, step)
    }

    /// Record a pattern retrieval step.
    pub fn record_retrieval(
        &self,
        pattern_ids: Vec<PatternId>,
        query: impl Into<String>,
        confidence: f32,
    ) -> Result<usize> {
        self.record(TrajectoryStep::pattern_retrieval(pattern_ids, query, confidence))
    }

    /// Record a decision step.
    pub fn record_decision(
        &self,
        pattern_ids: Vec<PatternId>,
        decision: impl Into<String>,
        confidence: f32,
    ) -> Result<usize> {
        self.record(TrajectoryStep::decision(pattern_ids, decision, confidence))
    }

    /// Complete the recording.
    pub fn complete(self, outcome: Outcome, reward: f32) -> Result<CompleteResult> {
        self.recorder.complete(&self.trajectory_id, outcome, reward)
    }

    /// Abort the recording.
    pub fn abort(self) -> bool {
        self.recorder.abort(&self.trajectory_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_recorder_config_default() {
        let config = RecorderConfig::default();
        assert_eq!(config.max_steps, 50);
        assert!(config.compress);
        assert!(config.auto_create_edges);
        assert!((config.min_reward_for_edges - 0.6).abs() < 0.001);
    }

    #[test]
    fn test_recorder_start_and_record() {
        let recorder = TrajectoryRecorder::new();

        let id = recorder.start("test query", Some("session-1".to_string()));
        assert!(recorder.is_active(&id));

        let step = TrajectoryStep::pattern_retrieval(
            vec![PatternId::from_string("pat_1")],
            "caching",
            0.9,
        );
        let index = recorder.record_step(&id, step).unwrap();
        assert_eq!(index, 0);

        let trajectory = recorder.get_active(&id).unwrap();
        assert_eq!(trajectory.step_count(), 1);
    }

    #[test]
    fn test_recorder_complete() {
        let recorder = TrajectoryRecorder::new();

        let id = recorder.start("test query", None);
        recorder
            .record_step(
                &id,
                TrajectoryStep::pattern_retrieval(
                    vec![PatternId::from_string("pat_1")],
                    "query",
                    0.9,
                ),
            )
            .unwrap();

        let result = recorder.complete(&id, Outcome::Success, 0.85).unwrap();

        assert!(!recorder.is_active(&id));
        assert_eq!(result.outcome, Outcome::Success);
        assert!((result.reward - 0.85).abs() < 0.001);
        assert_eq!(result.step_count, 1);
        assert!(result.profdag_node_id.starts_with("traj_node_"));
    }

    #[test]
    fn test_recorder_abort() {
        let recorder = TrajectoryRecorder::new();

        let id = recorder.start("test query", None);
        assert!(recorder.is_active(&id));

        let aborted = recorder.abort(&id);
        assert!(aborted);
        assert!(!recorder.is_active(&id));
    }

    #[test]
    fn test_recorder_replay() {
        let recorder = TrajectoryRecorder::new();

        let id = recorder.start("How to optimize?", None);
        recorder
            .record_step(
                &id,
                TrajectoryStep::pattern_retrieval(
                    vec![PatternId::from_string("pat_1"), PatternId::from_string("pat_2")],
                    "optimize",
                    0.85,
                ),
            )
            .unwrap();
        recorder
            .record_step(
                &id,
                TrajectoryStep::decision(
                    vec![PatternId::from_string("pat_1")],
                    "selected pat_1",
                    0.9,
                ),
            )
            .unwrap();

        recorder.complete(&id, Outcome::Success, 0.9).unwrap();

        let replay = recorder.replay(&id).unwrap();
        assert_eq!(replay.step_summaries.len(), 2);
        assert_eq!(replay.pattern_ids.len(), 2);
    }

    #[test]
    fn test_recorder_max_steps() {
        let config = RecorderConfig::new().with_max_steps(2);
        let recorder = TrajectoryRecorder::with_config(config);

        let id = recorder.start("test", None);

        recorder
            .record_step(&id, TrajectoryStep::new(
                crate::learning::trajectory::StepType::Decision,
                vec![],
                "step1",
                0.9,
            ))
            .unwrap();

        recorder
            .record_step(&id, TrajectoryStep::new(
                crate::learning::trajectory::StepType::Decision,
                vec![],
                "step2",
                0.9,
            ))
            .unwrap();

        // Third step should fail
        let result = recorder.record_step(&id, TrajectoryStep::new(
            crate::learning::trajectory::StepType::Decision,
            vec![],
            "step3",
            0.9,
        ));

        assert!(result.is_err());
    }

    #[test]
    fn test_recorder_stats() {
        let recorder = TrajectoryRecorder::new();

        let id1 = recorder.start("query 1", None);
        recorder.complete(&id1, Outcome::Success, 0.9).unwrap();

        let id2 = recorder.start("query 2", None);
        recorder.complete(&id2, Outcome::Failure, 0.2).unwrap();

        let stats = recorder.stats();
        assert_eq!(stats.total_count, 2);
        assert_eq!(stats.success_count, 1);
        assert_eq!(stats.failure_count, 1);
    }

    #[test]
    fn test_recording_session() {
        let recorder = Arc::new(TrajectoryRecorder::new());

        let session = RecordingSession::with_session(
            recorder.clone(),
            "test query",
            "session-123",
        );

        session
            .record_retrieval(
                vec![PatternId::from_string("pat_1")],
                "search",
                0.9,
            )
            .unwrap();

        session
            .record_decision(
                vec![PatternId::from_string("pat_1")],
                "selected",
                0.85,
            )
            .unwrap();

        let result = session.complete(Outcome::Success, 0.9).unwrap();
        assert_eq!(result.step_count, 2);
    }

    #[test]
    fn test_skip_failed_trajectory() {
        let mut config = RecorderConfig::new();
        config.record_failures = false;

        let recorder = TrajectoryRecorder::with_config(config);

        let id = recorder.start("test", None);
        let result = recorder.complete(&id, Outcome::Failure, 0.2).unwrap();

        // Should return empty profdag_node_id for skipped failures
        assert!(result.profdag_node_id.is_empty());
        assert_eq!(result.edges_created, 0);
    }

    #[test]
    fn test_complete_result_serialization() {
        let result = CompleteResult {
            trajectory_id: TrajectoryId::from_string("test-123"),
            profdag_node_id: "traj_node_test-123".to_string(),
            edges_created: 3,
            outcome: Outcome::Success,
            reward: 0.85,
            duration_ms: 150,
            step_count: 5,
            compressed: true,
            compressed_size: Some(1024),
        };

        let json = serde_json::to_string(&result).unwrap();
        let deserialized: CompleteResult = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.trajectory_id, result.trajectory_id);
        assert_eq!(deserialized.profdag_node_id, result.profdag_node_id);
        assert_eq!(deserialized.edges_created, result.edges_created);
    }

    #[tokio::test]
    async fn test_complete_async_without_storage() {
        // complete_async with no storage should behave like sync complete
        let recorder = TrajectoryRecorder::new();
        assert!(!recorder.has_storage());

        let id = recorder.start("async test query", Some("session-async".to_string()));
        recorder
            .record_step(
                &id,
                TrajectoryStep::pattern_retrieval(
                    vec![PatternId::from_string("pat_a")],
                    "search",
                    0.9,
                ),
            )
            .unwrap();

        let result = recorder.complete_async(&id, Outcome::Success, 0.85).await.unwrap();

        assert!(!recorder.is_active(&id));
        assert_eq!(result.outcome, Outcome::Success);
        assert!((result.reward - 0.85).abs() < 0.001);
        assert_eq!(result.step_count, 1);
        // Without storage, node ID is a UUID (not the old traj_node_ prefix)
        assert!(!result.profdag_node_id.is_empty());
    }

    #[test]
    fn test_compress_decompress_roundtrip() {
        let recorder = TrajectoryRecorder::new();

        let original_data = b"Hello, this is a test of real gzip compression in nagual-rs!";
        let compressed = recorder.compress_data(original_data).unwrap();

        // Compressed data should be different from original (gzip header)
        assert_ne!(compressed.as_slice(), original_data.as_slice());

        // Decompress should recover original
        let decompressed = recorder.decompress_data(&compressed).unwrap();
        assert_eq!(decompressed, original_data);
    }

    #[test]
    fn test_compress_decompress_roundtrip_large() {
        let recorder = TrajectoryRecorder::new();

        // Create a larger payload that compresses well
        let original_data: Vec<u8> = (0..10_000)
            .map(|i| (i % 256) as u8)
            .collect();

        let compressed = recorder.compress_data(&original_data).unwrap();

        // Gzip should achieve real compression on repetitive data
        assert!(
            compressed.len() < original_data.len(),
            "Expected compression: compressed={} original={}",
            compressed.len(),
            original_data.len()
        );

        let decompressed = recorder.decompress_data(&compressed).unwrap();
        assert_eq!(decompressed, original_data);
    }

    #[test]
    fn test_compress_decompress_empty() {
        let recorder = TrajectoryRecorder::new();

        let compressed = recorder.compress_data(b"").unwrap();
        let decompressed = recorder.decompress_data(&compressed).unwrap();
        assert!(decompressed.is_empty());
    }

    #[test]
    fn test_collect_pattern_edges() {
        let recorder = TrajectoryRecorder::new();

        // Build a trajectory with multiple patterns
        let mut trajectory = Trajectory::new();
        trajectory.add_step(TrajectoryStep::pattern_retrieval(
            vec![PatternId::from_string("pat_1"), PatternId::from_string("pat_2")],
            "first search",
            0.9,
        ));
        trajectory.add_step(TrajectoryStep::decision(
            vec![PatternId::from_string("pat_3")],
            "chose pat_3",
            0.85,
        ));
        trajectory.set_outcome(Outcome::Success, 0.9);

        let edges = recorder.collect_pattern_edges(&trajectory, "fake_node_id");

        // Pattern sequence is: pat_1 -> pat_2 -> pat_3
        // So we expect 2 edges: pat_1->pat_2, pat_2->pat_3
        assert_eq!(edges.len(), 2);

        assert_eq!(edges[0].source_id, "pat_1");
        assert_eq!(edges[0].target_id, "pat_2");
        assert_eq!(edges[0].edge_type, super::super::EdgeType::LeadsTo);
        assert!((edges[0].weight - 0.9).abs() < 0.001);

        assert_eq!(edges[1].source_id, "pat_2");
        assert_eq!(edges[1].target_id, "pat_3");
        assert_eq!(edges[1].edge_type, super::super::EdgeType::LeadsTo);
    }

    #[test]
    fn test_collect_pattern_edges_empty() {
        let recorder = TrajectoryRecorder::new();

        // Trajectory with no pattern IDs
        let mut trajectory = Trajectory::new();
        trajectory.add_step(TrajectoryStep::new(
            crate::learning::trajectory::StepType::Decision,
            vec![],
            "no patterns",
            0.9,
        ));
        trajectory.set_outcome(Outcome::Success, 0.8);

        let edges = recorder.collect_pattern_edges(&trajectory, "fake_node_id");
        assert!(edges.is_empty());
    }

    #[test]
    fn test_collect_pattern_edges_deduplicates_consecutive() {
        let recorder = TrajectoryRecorder::new();

        // Same pattern repeated consecutively should not create self-edges
        let mut trajectory = Trajectory::new();
        trajectory.add_step(TrajectoryStep::pattern_retrieval(
            vec![PatternId::from_string("pat_1")],
            "search",
            0.9,
        ));
        trajectory.add_step(TrajectoryStep::decision(
            vec![PatternId::from_string("pat_1")],
            "same pattern",
            0.85,
        ));
        trajectory.add_step(TrajectoryStep::decision(
            vec![PatternId::from_string("pat_2")],
            "new pattern",
            0.8,
        ));
        trajectory.set_outcome(Outcome::Success, 0.9);

        let edges = recorder.collect_pattern_edges(&trajectory, "fake_node_id");

        // Sequence after dedup: pat_1 -> pat_2, so only 1 edge
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].source_id, "pat_1");
        assert_eq!(edges[0].target_id, "pat_2");
    }

    #[test]
    fn test_with_storage_constructor() {
        // We cannot create a real ProfDAGStorage without a database,
        // but we can verify the constructor field via has_storage().
        let recorder = TrajectoryRecorder::new();
        assert!(!recorder.has_storage());

        let recorder2 = TrajectoryRecorder::with_config(RecorderConfig::default());
        assert!(!recorder2.has_storage());
    }

    #[test]
    fn test_store_and_replay_with_real_compression() {
        // Verify that the full store -> replay roundtrip works with real gzip
        let recorder = TrajectoryRecorder::new();

        let id = recorder.start("compression roundtrip test", None);
        recorder
            .record_step(
                &id,
                TrajectoryStep::pattern_retrieval(
                    vec![PatternId::from_string("pat_x")],
                    "roundtrip",
                    0.95,
                ),
            )
            .unwrap();

        let result = recorder.complete(&id, Outcome::Success, 0.9).unwrap();
        assert!(result.compressed);
        assert!(result.compressed_size.is_some());

        // Replay should decompress correctly
        let replay = recorder.replay(&id).unwrap();
        assert_eq!(replay.step_summaries.len(), 1);
        assert_eq!(replay.trajectory.query.as_deref(), Some("compression roundtrip test"));
    }
}
