//! Performance Profiler for ProfDAG operations.
//!
//! This module provides fine-grained performance profiling for all ProfDAG
//! operations including search, traversal, routing, injection, and wormhole
//! management. It uses RAII guards for automatic timing and supports
//! aggregated snapshots with percentile calculations.
//!
//! # Zero-Cost When Disabled
//!
//! The profiler checks its `enabled` flag early in all hot paths, ensuring
//! zero overhead when profiling is disabled.
//!
//! # Example
//!
//! ```rust
//! use nagual::profdag::profiler::{ProfDAGProfiler, ProfilerConfig, OperationType};
//!
//! let profiler = ProfDAGProfiler::new(ProfilerConfig::default());
//!
//! // RAII-based timing
//! {
//!     let _guard = profiler.start_operation(OperationType::Search);
//!     // ... perform search ...
//!     // timing recorded automatically on drop
//! }
//!
//! // Manual recording
//! profiler.record_search(128, 10, 450);
//!
//! // Get aggregated snapshot
//! let snapshot = profiler.snapshot();
//! println!("Total operations: {}", snapshot.total_operations);
//! println!("P95 search latency: {:?}", snapshot.by_type.get(&OperationType::Search));
//! ```

use std::collections::HashMap;
use std::fmt;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering as AtomicOrdering};
use std::time::Instant;

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// OperationType
// ---------------------------------------------------------------------------

/// The type of ProfDAG operation being profiled.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OperationType {
    /// HNSW or brute-force vector similarity search.
    Search,
    /// Graph traversal along edges.
    Traversal,
    /// Vendor routing decision (FastGRNN / model selection).
    Routing,
    /// E_nagual attention injection into vendor prompts.
    Injection,
    /// Wormhole shortcut creation.
    WormholeCreation,
    /// Light cone temporal query.
    LightConeQuery,
    /// Trajectory recording step.
    TrajectoryRecord,
    /// Storage read (SQLite / PostgreSQL).
    StorageRead,
    /// Storage write (SQLite / PostgreSQL).
    StorageWrite,
}

impl fmt::Display for OperationType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Search => write!(f, "Search"),
            Self::Traversal => write!(f, "Traversal"),
            Self::Routing => write!(f, "Routing"),
            Self::Injection => write!(f, "Injection"),
            Self::WormholeCreation => write!(f, "WormholeCreation"),
            Self::LightConeQuery => write!(f, "LightConeQuery"),
            Self::TrajectoryRecord => write!(f, "TrajectoryRecord"),
            Self::StorageRead => write!(f, "StorageRead"),
            Self::StorageWrite => write!(f, "StorageWrite"),
        }
    }
}

impl OperationType {
    /// Return all operation type variants.
    pub fn all() -> &'static [OperationType] {
        &[
            Self::Search,
            Self::Traversal,
            Self::Routing,
            Self::Injection,
            Self::WormholeCreation,
            Self::LightConeQuery,
            Self::TrajectoryRecord,
            Self::StorageRead,
            Self::StorageWrite,
        ]
    }
}

// ---------------------------------------------------------------------------
// OperationRecord
// ---------------------------------------------------------------------------

/// A single recorded operation with timing and metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperationRecord {
    /// The type of operation.
    pub op_type: OperationType,
    /// Wall-clock start timestamp (epoch microseconds).
    pub start_epoch_us: u64,
    /// Duration of the operation in microseconds.
    pub duration_us: u64,
    /// Arbitrary key-value metadata for this operation.
    pub metadata: HashMap<String, String>,
    /// Whether the operation completed successfully.
    pub success: bool,
}

// ---------------------------------------------------------------------------
// ProfilerConfig
// ---------------------------------------------------------------------------

/// Configuration for the ProfDAG profiler.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfilerConfig {
    /// Whether the profiler is enabled (default: `true`).
    pub enabled: bool,
    /// Sampling rate between 0.0 and 1.0. 1.0 records every operation (default: `1.0`).
    pub sample_rate: f64,
    /// Threshold in milliseconds above which a query is considered slow (default: `50`).
    pub slow_query_threshold_ms: u64,
    /// Maximum number of operation records to retain in the ring buffer (default: `10000`).
    pub history_size: usize,
    /// Whether to track memory allocation metrics (default: `false`).
    pub track_allocations: bool,
}

impl Default for ProfilerConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            sample_rate: 1.0,
            slow_query_threshold_ms: 50,
            history_size: 10_000,
            track_allocations: false,
        }
    }
}

impl ProfilerConfig {
    /// Create a new builder starting from default values.
    pub fn builder() -> ProfilerConfigBuilder {
        ProfilerConfigBuilder(Self::default())
    }
}

/// Builder for [`ProfilerConfig`].
pub struct ProfilerConfigBuilder(ProfilerConfig);

impl ProfilerConfigBuilder {
    /// Set whether the profiler is enabled.
    pub fn enabled(mut self, enabled: bool) -> Self {
        self.0.enabled = enabled;
        self
    }

    /// Set the sampling rate (clamped to 0.0..=1.0).
    pub fn sample_rate(mut self, rate: f64) -> Self {
        self.0.sample_rate = rate.clamp(0.0, 1.0);
        self
    }

    /// Set the slow query threshold in milliseconds.
    pub fn slow_query_threshold_ms(mut self, ms: u64) -> Self {
        self.0.slow_query_threshold_ms = ms;
        self
    }

    /// Set the maximum history size.
    pub fn history_size(mut self, size: usize) -> Self {
        self.0.history_size = size;
        self
    }

    /// Set whether to track allocations.
    pub fn track_allocations(mut self, track: bool) -> Self {
        self.0.track_allocations = track;
        self
    }

    /// Build the configuration.
    pub fn build(self) -> ProfilerConfig {
        self.0
    }
}

// ---------------------------------------------------------------------------
// TypeStats
// ---------------------------------------------------------------------------

/// Aggregated latency statistics for one operation type.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeStats {
    /// Total number of recorded operations.
    pub count: u64,
    /// Minimum observed latency in microseconds.
    pub min_us: u64,
    /// Maximum observed latency in microseconds.
    pub max_us: u64,
    /// Average latency in microseconds.
    pub avg_us: f64,
    /// 50th-percentile (median) latency in microseconds.
    pub p50_us: u64,
    /// 95th-percentile latency in microseconds.
    pub p95_us: u64,
    /// 99th-percentile latency in microseconds.
    pub p99_us: u64,
    /// Number of successful operations.
    pub success_count: u64,
    /// Number of failed operations.
    pub failure_count: u64,
}

// ---------------------------------------------------------------------------
// ProfileSnapshot
// ---------------------------------------------------------------------------

/// A point-in-time snapshot of all profiler metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileSnapshot {
    /// Total number of recorded operations across all types.
    pub total_operations: u64,
    /// Per-type aggregated statistics.
    pub by_type: HashMap<OperationType, TypeStats>,
    /// Total number of slow queries (exceeding configured threshold).
    pub slow_query_count: u64,
    /// Wormhole hit rate (0.0 - 1.0). Computed as wormhole traversals / total traversals.
    pub wormhole_hit_rate: f64,
    /// Cache hit rate (0.0 - 1.0). Computed from StorageRead metadata.
    pub cache_hit_rate: f64,
    /// Time since profiler creation in seconds.
    pub uptime_secs: u64,
}

impl ProfileSnapshot {
    /// Serialize the snapshot to a JSON value.
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::to_value(self).unwrap_or(serde_json::Value::Null)
    }
}

// ---------------------------------------------------------------------------
// HotPath
// ---------------------------------------------------------------------------

/// A frequently executed sequence of operation types.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HotPath {
    /// Ordered sequence of operation types.
    pub path: Vec<OperationType>,
    /// How many times this sequence was observed.
    pub frequency: u64,
    /// Average total latency for the full sequence in microseconds.
    pub avg_latency_us: f64,
}

// ---------------------------------------------------------------------------
// OperationGuard (RAII)
// ---------------------------------------------------------------------------

/// RAII guard that records operation timing on drop.
///
/// Created by [`ProfDAGProfiler::start_operation`]. When the guard is
/// dropped the elapsed duration is automatically recorded.
pub struct OperationGuard {
    profiler: *const ProfDAGProfiler,
    op_type: OperationType,
    start: Instant,
    start_epoch_us: u64,
    metadata: HashMap<String, String>,
    success: bool,
    recorded: bool,
}

// SAFETY: The profiler reference is guaranteed to outlive the guard in all
// intended usage patterns (guard is short-lived, profiler lives for the
// application lifetime). The profiler itself is fully `Send + Sync`.
unsafe impl Send for OperationGuard {}
unsafe impl Sync for OperationGuard {}

impl OperationGuard {
    /// Attach metadata to this operation.
    pub fn set_metadata(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.metadata.insert(key.into(), value.into());
    }

    /// Mark the operation as failed.
    pub fn set_failed(&mut self) {
        self.success = false;
    }

    /// Manually finish the guard and return the duration in microseconds.
    /// Prevents the automatic recording on drop.
    pub fn finish(mut self) -> u64 {
        let duration_us = self.start.elapsed().as_micros() as u64;
        self.record(duration_us);
        self.recorded = true;
        duration_us
    }

    fn record(&self, duration_us: u64) {
        // SAFETY: see above.
        let profiler = unsafe { &*self.profiler };
        profiler.record_operation(OperationRecord {
            op_type: self.op_type,
            start_epoch_us: self.start_epoch_us,
            duration_us,
            metadata: self.metadata.clone(),
            success: self.success,
        });
    }
}

impl Drop for OperationGuard {
    fn drop(&mut self) {
        if !self.recorded {
            let duration_us = self.start.elapsed().as_micros() as u64;
            self.record(duration_us);
        }
    }
}

// ---------------------------------------------------------------------------
// ProfDAGProfiler
// ---------------------------------------------------------------------------

/// Core profiler that collects fine-grained timing data for all ProfDAG operations.
pub struct ProfDAGProfiler {
    config: ProfilerConfig,
    /// Ring buffer of operation records. Oldest entries are evicted first.
    records: RwLock<Vec<OperationRecord>>,
    /// Global operation counter (atomic for fast path).
    op_counter: AtomicU64,
    /// Slow query counter (atomic).
    slow_query_counter: AtomicU64,
    /// Wormhole traversal counter (used for hit rate).
    wormhole_traversal_count: AtomicU64,
    /// Total traversal counter (used for hit rate).
    total_traversal_count: AtomicU64,
    /// Cache hit counter.
    cache_hit_count: AtomicU64,
    /// Total cache access counter.
    total_cache_access_count: AtomicU64,
    /// Profiler creation time.
    created_at: Instant,
    /// Sampling RNG state (simple LCG).
    sample_counter: AtomicU64,
    /// Whether profiler is enabled (also stored atomically for fast check).
    enabled: AtomicBool,
}

impl ProfDAGProfiler {
    /// Create a new profiler with the given configuration.
    pub fn new(config: ProfilerConfig) -> Self {
        let enabled = config.enabled;
        Self {
            config,
            records: RwLock::new(Vec::new()),
            op_counter: AtomicU64::new(0),
            slow_query_counter: AtomicU64::new(0),
            wormhole_traversal_count: AtomicU64::new(0),
            total_traversal_count: AtomicU64::new(0),
            cache_hit_count: AtomicU64::new(0),
            total_cache_access_count: AtomicU64::new(0),
            created_at: Instant::now(),
            sample_counter: AtomicU64::new(0),
            enabled: AtomicBool::new(enabled),
        }
    }

    /// Create a profiler with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(ProfilerConfig::default())
    }

    /// Create a disabled profiler (zero-cost no-op).
    pub fn disabled() -> Self {
        Self::new(ProfilerConfig {
            enabled: false,
            ..ProfilerConfig::default()
        })
    }

    /// Return whether the profiler is currently enabled.
    pub fn is_enabled(&self) -> bool {
        self.enabled.load(AtomicOrdering::Relaxed)
    }

    /// Enable or disable the profiler at runtime.
    pub fn set_enabled(&self, enabled: bool) {
        self.enabled.store(enabled, AtomicOrdering::Relaxed);
    }

    /// Get the profiler configuration.
    pub fn config(&self) -> &ProfilerConfig {
        &self.config
    }

    // -----------------------------------------------------------------------
    // RAII operation timing
    // -----------------------------------------------------------------------

    /// Start timing an operation. Returns an [`OperationGuard`] that
    /// automatically records the elapsed time when dropped.
    ///
    /// If the profiler is disabled, a guard is still returned but its drop
    /// will short-circuit via the disabled check in `record_operation`.
    pub fn start_operation(&self, op_type: OperationType) -> OperationGuard {
        let now = Instant::now();
        let epoch_us = self.created_at.elapsed().as_micros() as u64;

        OperationGuard {
            profiler: self as *const Self,
            op_type,
            start: now,
            start_epoch_us: epoch_us,
            metadata: HashMap::new(),
            success: true,
            recorded: false,
        }
    }

    // -----------------------------------------------------------------------
    // Manual recording helpers
    // -----------------------------------------------------------------------

    /// Record a completed operation manually.
    pub fn record_operation(&self, record: OperationRecord) {
        // Fast exit when disabled.
        if !self.is_enabled() {
            return;
        }

        // Sampling: deterministic skip based on counter.
        if self.config.sample_rate < 1.0 {
            let counter = self.sample_counter.fetch_add(1, AtomicOrdering::Relaxed);
            let threshold = (self.config.sample_rate * 1_000_000.0) as u64;
            if (counter % 1_000_000) >= threshold {
                return;
            }
        }

        // Update atomic counters.
        self.op_counter.fetch_add(1, AtomicOrdering::Relaxed);

        let threshold_us = self.config.slow_query_threshold_ms * 1_000;
        if record.duration_us >= threshold_us {
            self.slow_query_counter.fetch_add(1, AtomicOrdering::Relaxed);
        }

        // Track traversal / wormhole hit rate.
        if record.op_type == OperationType::Traversal {
            self.total_traversal_count
                .fetch_add(1, AtomicOrdering::Relaxed);
            if record.metadata.get("wormhole_used").map(|v| v == "true").unwrap_or(false) {
                self.wormhole_traversal_count
                    .fetch_add(1, AtomicOrdering::Relaxed);
            }
        }

        // Track cache hit rate.
        if record.op_type == OperationType::StorageRead {
            self.total_cache_access_count
                .fetch_add(1, AtomicOrdering::Relaxed);
            if record.metadata.get("cache_hit").map(|v| v == "true").unwrap_or(false) {
                self.cache_hit_count.fetch_add(1, AtomicOrdering::Relaxed);
            }
        }

        // Append to ring buffer.
        let mut records = self.records.write();
        if records.len() >= self.config.history_size {
            records.remove(0);
        }
        records.push(record);
    }

    /// Convenience: record a search operation.
    pub fn record_search(&self, query_dim: usize, results: usize, latency_us: u64) {
        if !self.is_enabled() {
            return;
        }
        let mut metadata = HashMap::new();
        metadata.insert("query_dim".to_string(), query_dim.to_string());
        metadata.insert("results".to_string(), results.to_string());
        self.record_operation(OperationRecord {
            op_type: OperationType::Search,
            start_epoch_us: self.created_at.elapsed().as_micros() as u64,
            duration_us: latency_us,
            metadata,
            success: true,
        });
    }

    /// Convenience: record a graph traversal.
    pub fn record_traversal(
        &self,
        edges_visited: usize,
        wormholes_used: usize,
        latency_us: u64,
    ) {
        if !self.is_enabled() {
            return;
        }
        let mut metadata = HashMap::new();
        metadata.insert("edges_visited".to_string(), edges_visited.to_string());
        metadata.insert("wormholes_used".to_string(), wormholes_used.to_string());
        if wormholes_used > 0 {
            metadata.insert("wormhole_used".to_string(), "true".to_string());
        }
        self.record_operation(OperationRecord {
            op_type: OperationType::Traversal,
            start_epoch_us: self.created_at.elapsed().as_micros() as u64,
            duration_us: latency_us,
            metadata,
            success: true,
        });
    }

    /// Convenience: record a vendor routing decision.
    pub fn record_routing(&self, vendor: &str, complexity: f64, latency_us: u64) {
        if !self.is_enabled() {
            return;
        }
        let mut metadata = HashMap::new();
        metadata.insert("vendor".to_string(), vendor.to_string());
        metadata.insert("complexity".to_string(), format!("{:.4}", complexity));
        self.record_operation(OperationRecord {
            op_type: OperationType::Routing,
            start_epoch_us: self.created_at.elapsed().as_micros() as u64,
            duration_us: latency_us,
            metadata,
            success: true,
        });
    }

    /// Convenience: record an E_nagual injection.
    pub fn record_injection(
        &self,
        provider: &str,
        patterns: usize,
        tokens: usize,
        latency_us: u64,
    ) {
        if !self.is_enabled() {
            return;
        }
        let mut metadata = HashMap::new();
        metadata.insert("provider".to_string(), provider.to_string());
        metadata.insert("patterns".to_string(), patterns.to_string());
        metadata.insert("tokens".to_string(), tokens.to_string());
        self.record_operation(OperationRecord {
            op_type: OperationType::Injection,
            start_epoch_us: self.created_at.elapsed().as_micros() as u64,
            duration_us: latency_us,
            metadata,
            success: true,
        });
    }

    // -----------------------------------------------------------------------
    // Snapshot / queries
    // -----------------------------------------------------------------------

    /// Produce an aggregated snapshot of all profiled metrics.
    pub fn snapshot(&self) -> ProfileSnapshot {
        let records = self.records.read();

        let mut by_type: HashMap<OperationType, Vec<u64>> = HashMap::new();
        let mut success_counts: HashMap<OperationType, u64> = HashMap::new();
        let mut failure_counts: HashMap<OperationType, u64> = HashMap::new();

        for rec in records.iter() {
            by_type.entry(rec.op_type).or_default().push(rec.duration_us);
            if rec.success {
                *success_counts.entry(rec.op_type).or_default() += 1;
            } else {
                *failure_counts.entry(rec.op_type).or_default() += 1;
            }
        }

        let mut stats_map: HashMap<OperationType, TypeStats> = HashMap::new();

        for (op_type, mut latencies) in by_type {
            latencies.sort_unstable();
            let count = latencies.len() as u64;
            let min_us = *latencies.first().unwrap_or(&0);
            let max_us = *latencies.last().unwrap_or(&0);
            let sum: u64 = latencies.iter().sum();
            let avg_us = if count > 0 {
                sum as f64 / count as f64
            } else {
                0.0
            };

            let p50_us = percentile(&latencies, 50.0);
            let p95_us = percentile(&latencies, 95.0);
            let p99_us = percentile(&latencies, 99.0);

            stats_map.insert(
                op_type,
                TypeStats {
                    count,
                    min_us,
                    max_us,
                    avg_us,
                    p50_us,
                    p95_us,
                    p99_us,
                    success_count: success_counts.get(&op_type).copied().unwrap_or(0),
                    failure_count: failure_counts.get(&op_type).copied().unwrap_or(0),
                },
            );
        }

        let total_traversals = self.total_traversal_count.load(AtomicOrdering::Relaxed);
        let wormhole_traversals = self
            .wormhole_traversal_count
            .load(AtomicOrdering::Relaxed);
        let wormhole_hit_rate = if total_traversals > 0 {
            wormhole_traversals as f64 / total_traversals as f64
        } else {
            0.0
        };

        let total_cache = self
            .total_cache_access_count
            .load(AtomicOrdering::Relaxed);
        let cache_hits = self.cache_hit_count.load(AtomicOrdering::Relaxed);
        let cache_hit_rate = if total_cache > 0 {
            cache_hits as f64 / total_cache as f64
        } else {
            0.0
        };

        ProfileSnapshot {
            total_operations: self.op_counter.load(AtomicOrdering::Relaxed),
            by_type: stats_map,
            slow_query_count: self.slow_query_counter.load(AtomicOrdering::Relaxed),
            wormhole_hit_rate,
            cache_hit_rate,
            uptime_secs: self.created_at.elapsed().as_secs(),
        }
    }

    /// Return all operations that exceeded the slow query threshold.
    pub fn slow_queries(&self) -> Vec<OperationRecord> {
        if !self.is_enabled() {
            return Vec::new();
        }
        let threshold_us = self.config.slow_query_threshold_ms * 1_000;
        let records = self.records.read();
        records
            .iter()
            .filter(|r| r.duration_us >= threshold_us)
            .cloned()
            .collect()
    }

    /// Identify the most frequently executed operation sequences (hot paths).
    ///
    /// A hot path is a sequence of 2 or 3 consecutive operation types that
    /// appears repeatedly in the recorded history.
    pub fn hot_paths(&self) -> Vec<HotPath> {
        if !self.is_enabled() {
            return Vec::new();
        }
        let records = self.records.read();
        if records.len() < 2 {
            return Vec::new();
        }

        // Count 2-grams and 3-grams of operation types.
        let mut bigram_counts: HashMap<(OperationType, OperationType), (u64, u64)> = HashMap::new();
        let mut trigram_counts: HashMap<(OperationType, OperationType, OperationType), (u64, u64)> =
            HashMap::new();

        for window in records.windows(2) {
            let key = (window[0].op_type, window[1].op_type);
            let entry = bigram_counts.entry(key).or_insert((0, 0));
            entry.0 += 1;
            entry.1 += window[0].duration_us + window[1].duration_us;
        }

        for window in records.windows(3) {
            let key = (window[0].op_type, window[1].op_type, window[2].op_type);
            let entry = trigram_counts.entry(key).or_insert((0, 0));
            entry.0 += 1;
            entry.1 += window[0].duration_us + window[1].duration_us + window[2].duration_us;
        }

        let mut hot: Vec<HotPath> = Vec::new();

        for ((a, b), (freq, total_us)) in &bigram_counts {
            if *freq >= 2 {
                hot.push(HotPath {
                    path: vec![*a, *b],
                    frequency: *freq,
                    avg_latency_us: *total_us as f64 / *freq as f64,
                });
            }
        }

        for ((a, b, c), (freq, total_us)) in &trigram_counts {
            if *freq >= 2 {
                hot.push(HotPath {
                    path: vec![*a, *b, *c],
                    frequency: *freq,
                    avg_latency_us: *total_us as f64 / *freq as f64,
                });
            }
        }

        // Sort by frequency descending, then by latency descending.
        hot.sort_by(|a, b| {
            b.frequency
                .cmp(&a.frequency)
                .then_with(|| {
                    b.avg_latency_us
                        .partial_cmp(&a.avg_latency_us)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
        });

        hot.truncate(20);
        hot
    }

    /// Clear all collected profiling data and reset counters.
    pub fn reset(&self) {
        self.records.write().clear();
        self.op_counter.store(0, AtomicOrdering::Relaxed);
        self.slow_query_counter.store(0, AtomicOrdering::Relaxed);
        self.wormhole_traversal_count
            .store(0, AtomicOrdering::Relaxed);
        self.total_traversal_count
            .store(0, AtomicOrdering::Relaxed);
        self.cache_hit_count.store(0, AtomicOrdering::Relaxed);
        self.total_cache_access_count
            .store(0, AtomicOrdering::Relaxed);
    }

    /// Return the total number of operations recorded.
    pub fn total_operations(&self) -> u64 {
        self.op_counter.load(AtomicOrdering::Relaxed)
    }
}

// SAFETY: All mutable state is behind RwLock or atomics.
unsafe impl Send for ProfDAGProfiler {}
unsafe impl Sync for ProfDAGProfiler {}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Compute a percentile from a **sorted** slice of latencies.
fn percentile(sorted: &[u64], pct: f64) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }
    let rank = (pct / 100.0) * (sorted.len() as f64 - 1.0);
    let lower = rank.floor() as usize;
    let upper = rank.ceil() as usize;
    let lower = lower.min(sorted.len() - 1);
    let upper = upper.min(sorted.len() - 1);
    if lower == upper {
        sorted[lower]
    } else {
        let frac = rank - lower as f64;
        let low_val = sorted[lower] as f64;
        let high_val = sorted[upper] as f64;
        (low_val + frac * (high_val - low_val)).round() as u64
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_profiler_config_defaults() {
        let config = ProfilerConfig::default();
        assert!(config.enabled);
        assert!((config.sample_rate - 1.0).abs() < f64::EPSILON);
        assert_eq!(config.slow_query_threshold_ms, 50);
        assert_eq!(config.history_size, 10_000);
        assert!(!config.track_allocations);
    }

    #[test]
    fn test_profiler_config_builder() {
        let config = ProfilerConfig::builder()
            .enabled(false)
            .sample_rate(0.5)
            .slow_query_threshold_ms(100)
            .history_size(500)
            .track_allocations(true)
            .build();

        assert!(!config.enabled);
        assert!((config.sample_rate - 0.5).abs() < f64::EPSILON);
        assert_eq!(config.slow_query_threshold_ms, 100);
        assert_eq!(config.history_size, 500);
        assert!(config.track_allocations);
    }

    #[test]
    fn test_disabled_profiler_is_noop() {
        let profiler = ProfDAGProfiler::disabled();
        assert!(!profiler.is_enabled());

        profiler.record_search(128, 10, 5000);
        profiler.record_traversal(50, 2, 1000);
        profiler.record_routing("openai", 0.8, 2000);
        profiler.record_injection("anthropic", 5, 1500, 3000);

        let snapshot = profiler.snapshot();
        assert_eq!(snapshot.total_operations, 0);
        assert!(snapshot.by_type.is_empty());
    }

    #[test]
    fn test_raii_guard_records_on_drop() {
        let profiler = ProfDAGProfiler::with_defaults();

        {
            let _guard = profiler.start_operation(OperationType::Search);
            thread::sleep(Duration::from_millis(5));
        }

        assert_eq!(profiler.total_operations(), 1);
        let snapshot = profiler.snapshot();
        let search_stats = snapshot.by_type.get(&OperationType::Search).unwrap();
        assert_eq!(search_stats.count, 1);
        // Should have recorded at least 4ms (allowing for timing jitter).
        assert!(search_stats.min_us >= 3_000, "min_us={} should be >= 3000", search_stats.min_us);
    }

    #[test]
    fn test_guard_finish_returns_duration() {
        let profiler = ProfDAGProfiler::with_defaults();

        let guard = profiler.start_operation(OperationType::Routing);
        thread::sleep(Duration::from_millis(2));
        let duration = guard.finish();

        assert!(duration >= 1_000, "duration_us={} should be >= 1000", duration);
        assert_eq!(profiler.total_operations(), 1);
    }

    #[test]
    fn test_guard_metadata_and_failure() {
        let profiler = ProfDAGProfiler::with_defaults();

        {
            let mut guard = profiler.start_operation(OperationType::StorageWrite);
            guard.set_metadata("table", "profdag_nodes");
            guard.set_failed();
        }

        let snapshot = profiler.snapshot();
        let stats = snapshot.by_type.get(&OperationType::StorageWrite).unwrap();
        assert_eq!(stats.failure_count, 1);
        assert_eq!(stats.success_count, 0);
    }

    #[test]
    fn test_record_search() {
        let profiler = ProfDAGProfiler::with_defaults();
        profiler.record_search(128, 10, 450);
        profiler.record_search(128, 5, 320);

        let snapshot = profiler.snapshot();
        let stats = snapshot.by_type.get(&OperationType::Search).unwrap();
        assert_eq!(stats.count, 2);
        assert_eq!(stats.min_us, 320);
        assert_eq!(stats.max_us, 450);
    }

    #[test]
    fn test_record_traversal_with_wormhole() {
        let profiler = ProfDAGProfiler::with_defaults();
        profiler.record_traversal(50, 2, 1000);
        profiler.record_traversal(30, 0, 800);
        profiler.record_traversal(40, 1, 900);

        let snapshot = profiler.snapshot();
        // 2 out of 3 traversals used wormholes.
        assert!(
            (snapshot.wormhole_hit_rate - 2.0 / 3.0).abs() < 0.01,
            "wormhole_hit_rate={}",
            snapshot.wormhole_hit_rate
        );
    }

    #[test]
    fn test_slow_queries() {
        let config = ProfilerConfig {
            slow_query_threshold_ms: 10, // 10ms = 10_000us
            ..ProfilerConfig::default()
        };
        let profiler = ProfDAGProfiler::new(config);

        profiler.record_search(128, 10, 5_000); // 5ms - not slow
        profiler.record_search(128, 10, 15_000); // 15ms - slow
        profiler.record_search(128, 10, 20_000); // 20ms - slow

        let slow = profiler.slow_queries();
        assert_eq!(slow.len(), 2);
        assert!(slow.iter().all(|r| r.duration_us >= 10_000));
    }

    #[test]
    fn test_hot_paths() {
        let profiler = ProfDAGProfiler::with_defaults();

        // Create a repeated pattern: Search -> Traversal -> Routing
        for _ in 0..5 {
            profiler.record_search(128, 10, 100);
            profiler.record_traversal(20, 1, 200);
            profiler.record_routing("openai", 0.5, 150);
        }

        let hot = profiler.hot_paths();
        assert!(!hot.is_empty());

        // The bigram (Search, Traversal) should appear with high frequency.
        let search_trav = hot.iter().find(|h| {
            h.path.len() == 2
                && h.path[0] == OperationType::Search
                && h.path[1] == OperationType::Traversal
        });
        assert!(search_trav.is_some(), "Expected Search->Traversal hot path");
        assert!(search_trav.unwrap().frequency >= 4);
    }

    #[test]
    fn test_reset() {
        let profiler = ProfDAGProfiler::with_defaults();
        profiler.record_search(128, 10, 500);
        profiler.record_traversal(20, 1, 300);
        assert!(profiler.total_operations() > 0);

        profiler.reset();
        assert_eq!(profiler.total_operations(), 0);
        let snapshot = profiler.snapshot();
        assert_eq!(snapshot.total_operations, 0);
        assert!(snapshot.by_type.is_empty());
    }

    #[test]
    fn test_history_size_limit() {
        let config = ProfilerConfig {
            history_size: 5,
            ..ProfilerConfig::default()
        };
        let profiler = ProfDAGProfiler::new(config);

        for i in 0..10 {
            profiler.record_search(128, 10, (i + 1) * 100);
        }

        // Records should be capped at 5.
        let snapshot = profiler.snapshot();
        let stats = snapshot.by_type.get(&OperationType::Search).unwrap();
        assert_eq!(stats.count, 5);
        // Oldest records (100-500us) should have been evicted.
        assert_eq!(stats.min_us, 600);
    }

    #[test]
    fn test_percentile_calculation() {
        // 100 evenly spaced values: 1, 2, ..., 100
        // P50 of 1..=100 with interpolation: rank = 0.5 * 99 = 49.5 -> interp(50, 51) = 50.5 -> 51
        let values: Vec<u64> = (1..=100).collect();
        assert_eq!(percentile(&values, 0.0), 1);
        assert_eq!(percentile(&values, 50.0), 51); // interpolated midpoint rounds up
        assert_eq!(percentile(&values, 95.0), 95);
        assert_eq!(percentile(&values, 99.0), 99);
        assert_eq!(percentile(&values, 100.0), 100);

        // Verify exact midpoint with even count
        let values_10: Vec<u64> = (1..=10).collect();
        // P50 of [1..10]: rank = 0.5 * 9 = 4.5 -> interp(5, 6) = 5.5 -> 6
        assert_eq!(percentile(&values_10, 50.0), 6);
        assert_eq!(percentile(&values_10, 0.0), 1);
        assert_eq!(percentile(&values_10, 100.0), 10);
    }

    #[test]
    fn test_percentile_empty_and_single() {
        assert_eq!(percentile(&[], 50.0), 0);
        assert_eq!(percentile(&[42], 99.0), 42);
    }

    #[test]
    fn test_snapshot_to_json() {
        let profiler = ProfDAGProfiler::with_defaults();
        profiler.record_search(128, 5, 200);
        let snapshot = profiler.snapshot();
        let json = snapshot.to_json();
        assert!(json.is_object());
        assert!(json.get("total_operations").is_some());
        assert!(json.get("by_type").is_some());
    }

    #[test]
    fn test_operation_type_display() {
        assert_eq!(OperationType::Search.to_string(), "Search");
        assert_eq!(OperationType::WormholeCreation.to_string(), "WormholeCreation");
        assert_eq!(OperationType::LightConeQuery.to_string(), "LightConeQuery");
    }

    #[test]
    fn test_operation_type_all() {
        let all = OperationType::all();
        assert_eq!(all.len(), 9);
    }

    #[test]
    fn test_enable_disable_at_runtime() {
        let profiler = ProfDAGProfiler::with_defaults();
        assert!(profiler.is_enabled());

        profiler.set_enabled(false);
        assert!(!profiler.is_enabled());

        profiler.record_search(128, 10, 500);
        assert_eq!(profiler.total_operations(), 0);

        profiler.set_enabled(true);
        profiler.record_search(128, 10, 500);
        assert_eq!(profiler.total_operations(), 1);
    }

    #[test]
    fn test_cache_hit_rate() {
        let profiler = ProfDAGProfiler::with_defaults();

        // 3 cache reads, 2 hits
        let mut meta_hit = HashMap::new();
        meta_hit.insert("cache_hit".to_string(), "true".to_string());

        profiler.record_operation(OperationRecord {
            op_type: OperationType::StorageRead,
            start_epoch_us: 0,
            duration_us: 50,
            metadata: meta_hit.clone(),
            success: true,
        });
        profiler.record_operation(OperationRecord {
            op_type: OperationType::StorageRead,
            start_epoch_us: 100,
            duration_us: 50,
            metadata: meta_hit,
            success: true,
        });
        profiler.record_operation(OperationRecord {
            op_type: OperationType::StorageRead,
            start_epoch_us: 200,
            duration_us: 500,
            metadata: HashMap::new(),
            success: true,
        });

        let snapshot = profiler.snapshot();
        assert!(
            (snapshot.cache_hit_rate - 2.0 / 3.0).abs() < 0.01,
            "cache_hit_rate={}",
            snapshot.cache_hit_rate
        );
    }

    #[test]
    fn test_concurrent_recording() {
        use std::sync::Arc;

        let profiler = Arc::new(ProfDAGProfiler::with_defaults());
        let mut handles = Vec::new();

        for _ in 0..4 {
            let p = Arc::clone(&profiler);
            handles.push(thread::spawn(move || {
                for i in 0..100 {
                    p.record_search(128, 10, (i + 1) * 10);
                }
            }));
        }

        for h in handles {
            h.join().unwrap();
        }

        assert_eq!(profiler.total_operations(), 400);
    }
}
