//! Performance Optimizer for ProfDAG operations.
//!
//! This module analyzes profiler snapshots and produces actionable
//! recommendations for tuning HNSW parameters, wormhole thresholds,
//! cache sizing, and more.
//!
//! # Example
//!
//! ```rust
//! use nagual::profdag::profiler::{ProfDAGProfiler, ProfilerConfig};
//! use nagual::profdag::optimizer::{ProfDAGOptimizer, OptimizerConfig};
//!
//! let profiler = ProfDAGProfiler::new(ProfilerConfig::default());
//!
//! // ... record operations ...
//!
//! let optimizer = ProfDAGOptimizer::new(OptimizerConfig::default());
//! let snapshot = profiler.snapshot();
//! let recommendations = optimizer.analyze(&snapshot);
//!
//! for rec in &recommendations {
//!     println!("[{:?}] {}: {}", rec.impact, rec.title, rec.description);
//! }
//! ```

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use super::profiler::{OperationType, ProfileSnapshot, TypeStats};

// ---------------------------------------------------------------------------
// OptimizerConfig
// ---------------------------------------------------------------------------

/// Configuration for the ProfDAG optimizer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizerConfig {
    /// Whether to automatically apply recommendations (default: `false`).
    pub auto_optimize: bool,
    /// Interval in seconds between automatic optimization passes (default: `3600`).
    pub optimization_interval_secs: u64,
    /// Minimum number of samples required before generating recommendations (default: `100`).
    pub min_samples: usize,
    /// Target P95 latency in milliseconds (default: `50`).
    pub target_p95_ms: u64,
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        Self {
            auto_optimize: false,
            optimization_interval_secs: 3600,
            min_samples: 100,
            target_p95_ms: 50,
        }
    }
}

impl OptimizerConfig {
    /// Create a new builder starting from default values.
    pub fn builder() -> OptimizerConfigBuilder {
        OptimizerConfigBuilder(Self::default())
    }
}

/// Builder for [`OptimizerConfig`].
pub struct OptimizerConfigBuilder(OptimizerConfig);

impl OptimizerConfigBuilder {
    /// Set whether to auto-optimize.
    pub fn auto_optimize(mut self, auto: bool) -> Self {
        self.0.auto_optimize = auto;
        self
    }

    /// Set the optimization interval in seconds.
    pub fn optimization_interval_secs(mut self, secs: u64) -> Self {
        self.0.optimization_interval_secs = secs;
        self
    }

    /// Set the minimum sample count.
    pub fn min_samples(mut self, samples: usize) -> Self {
        self.0.min_samples = samples;
        self
    }

    /// Set the target P95 latency in milliseconds.
    pub fn target_p95_ms(mut self, ms: u64) -> Self {
        self.0.target_p95_ms = ms;
        self
    }

    /// Build the configuration.
    pub fn build(self) -> OptimizerConfig {
        self.0
    }
}

// ---------------------------------------------------------------------------
// Recommendation / Bottleneck types
// ---------------------------------------------------------------------------

/// The category of an optimization recommendation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OptCategory {
    /// HNSW index parameter tuning.
    HnswTuning,
    /// Wormhole threshold / behavior tuning.
    WormholeTuning,
    /// Cache configuration changes.
    CacheTuning,
    /// Concurrency or batching improvements.
    Concurrency,
    /// Storage / I/O optimization.
    Storage,
    /// General architecture improvements.
    Architecture,
}

/// Impact level of a recommendation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Impact {
    /// Significant performance improvement expected.
    High,
    /// Moderate improvement expected.
    Medium,
    /// Minor improvement expected.
    Low,
}

/// Effort level required to implement a recommendation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Effort {
    /// Configuration change only, no code changes.
    Trivial,
    /// Small code change (< 50 lines).
    Small,
    /// Moderate code change (50-200 lines).
    Medium,
    /// Significant refactoring (> 200 lines).
    Large,
}

/// A specific optimization recommendation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Recommendation {
    /// Unique identifier for this recommendation.
    pub id: String,
    /// Category of optimization.
    pub category: OptCategory,
    /// Short title.
    pub title: String,
    /// Detailed description of the recommendation.
    pub description: String,
    /// Expected impact level.
    pub impact: Impact,
    /// Effort required to implement.
    pub effort: Effort,
    /// Suggested parameter changes (key: parameter name, value: suggested value).
    pub parameters: HashMap<String, String>,
}

/// A detected performance bottleneck.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bottleneck {
    /// The operation type that is the bottleneck.
    pub operation: OperationType,
    /// Severity from 0.0 (minor) to 1.0 (critical).
    pub severity: f64,
    /// Human-readable description.
    pub description: String,
    /// Current P95 latency in microseconds.
    pub current_p95_us: u64,
    /// Target P95 latency in microseconds.
    pub target_p95_us: u64,
}

// ---------------------------------------------------------------------------
// Specific recommendation types
// ---------------------------------------------------------------------------

/// HNSW-specific parameter recommendation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HnswRecommendation {
    /// Suggested HNSW M parameter (connections per layer).
    pub suggested_m: usize,
    /// Suggested ef_construction parameter.
    pub suggested_ef_construction: usize,
    /// Suggested ef_search parameter.
    pub suggested_ef_search: usize,
    /// Explanation of the reasoning.
    pub reasoning: String,
}

/// Wormhole-specific tuning recommendation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WormholeRecommendation {
    /// Suggested activation threshold.
    pub suggested_activation_threshold: u32,
    /// Suggested decay rate.
    pub suggested_decay_rate: f32,
    /// Suggested max wormholes per node.
    pub suggested_max_per_node: usize,
    /// Explanation of the reasoning.
    pub reasoning: String,
}

/// Cache sizing recommendation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheRecommendation {
    /// Suggested cache capacity.
    pub suggested_capacity: usize,
    /// Suggested TTL in seconds.
    pub suggested_ttl_secs: u64,
    /// Whether an LRU or LFU eviction policy is recommended.
    pub suggested_eviction: String,
    /// Explanation of the reasoning.
    pub reasoning: String,
}

/// Estimated impact of applying a recommendation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImpactEstimate {
    /// Estimated speedup as a percentage (e.g., 20.0 = 20% faster).
    pub estimated_speedup_pct: f64,
    /// Confidence in the estimate (0.0 - 1.0).
    pub confidence: f64,
    /// Explanation of how the estimate was derived.
    pub reasoning: String,
}

// ---------------------------------------------------------------------------
// ProfDAGOptimizer
// ---------------------------------------------------------------------------

/// Optimizer that analyzes profiler snapshots and recommends tuning actions.
pub struct ProfDAGOptimizer {
    config: OptimizerConfig,
}

impl ProfDAGOptimizer {
    /// Create a new optimizer with the given configuration.
    pub fn new(config: OptimizerConfig) -> Self {
        Self { config }
    }

    /// Create an optimizer with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(OptimizerConfig::default())
    }

    /// Get the optimizer configuration.
    pub fn config(&self) -> &OptimizerConfig {
        &self.config
    }

    /// Analyze a profiler snapshot and produce optimization recommendations.
    pub fn analyze(&self, snapshot: &ProfileSnapshot) -> Vec<Recommendation> {
        let mut recommendations = Vec::new();

        // Check minimum samples.
        if snapshot.total_operations < self.config.min_samples as u64 {
            return recommendations;
        }

        let target_p95_us = self.config.target_p95_ms * 1_000;

        // Analyze search performance.
        if let Some(search_stats) = snapshot.by_type.get(&OperationType::Search) {
            if search_stats.p95_us > target_p95_us {
                let hnsw_rec = self.suggest_hnsw_params(search_stats);
                let mut params = HashMap::new();
                params.insert("hnsw_m".to_string(), hnsw_rec.suggested_m.to_string());
                params.insert(
                    "ef_construction".to_string(),
                    hnsw_rec.suggested_ef_construction.to_string(),
                );
                params.insert(
                    "ef_search".to_string(),
                    hnsw_rec.suggested_ef_search.to_string(),
                );

                recommendations.push(Recommendation {
                    id: "hnsw-search-latency".to_string(),
                    category: OptCategory::HnswTuning,
                    title: "Reduce HNSW search latency".to_string(),
                    description: format!(
                        "Search P95 is {}us (target: {}us). {}",
                        search_stats.p95_us, target_p95_us, hnsw_rec.reasoning
                    ),
                    impact: Impact::High,
                    effort: Effort::Trivial,
                    parameters: params,
                });
            }

            if search_stats.failure_count > 0 {
                let failure_rate =
                    search_stats.failure_count as f64 / search_stats.count as f64;
                if failure_rate > 0.01 {
                    recommendations.push(Recommendation {
                        id: "search-failures".to_string(),
                        category: OptCategory::Architecture,
                        title: "Investigate search failures".to_string(),
                        description: format!(
                            "Search failure rate is {:.1}% ({} / {}). Check embedding dimensions and index health.",
                            failure_rate * 100.0,
                            search_stats.failure_count,
                            search_stats.count
                        ),
                        impact: Impact::High,
                        effort: Effort::Medium,
                        parameters: HashMap::new(),
                    });
                }
            }
        }

        // Analyze traversal performance and wormhole efficiency.
        if let Some(trav_stats) = snapshot.by_type.get(&OperationType::Traversal) {
            let wormhole_rec =
                self.suggest_wormhole_tuning(snapshot.wormhole_hit_rate, trav_stats);

            if snapshot.wormhole_hit_rate < 0.3 && trav_stats.p95_us > target_p95_us {
                let mut params = HashMap::new();
                params.insert(
                    "activation_threshold".to_string(),
                    wormhole_rec.suggested_activation_threshold.to_string(),
                );
                params.insert(
                    "decay_rate".to_string(),
                    format!("{:.2}", wormhole_rec.suggested_decay_rate),
                );

                recommendations.push(Recommendation {
                    id: "wormhole-low-hit-rate".to_string(),
                    category: OptCategory::WormholeTuning,
                    title: "Improve wormhole hit rate".to_string(),
                    description: format!(
                        "Wormhole hit rate is {:.1}% with traversal P95 at {}us. {}",
                        snapshot.wormhole_hit_rate * 100.0,
                        trav_stats.p95_us,
                        wormhole_rec.reasoning
                    ),
                    impact: Impact::Medium,
                    effort: Effort::Trivial,
                    parameters: params,
                });
            }

            if trav_stats.p95_us > target_p95_us * 2 {
                recommendations.push(Recommendation {
                    id: "traversal-slow".to_string(),
                    category: OptCategory::Architecture,
                    title: "Traversal latency exceeds 2x target".to_string(),
                    description: format!(
                        "Traversal P95 is {}us, which is more than 2x the target ({}us). \
                         Consider adding more wormhole shortcuts or reducing graph depth.",
                        trav_stats.p95_us, target_p95_us
                    ),
                    impact: Impact::High,
                    effort: Effort::Medium,
                    parameters: HashMap::new(),
                });
            }
        }

        // Analyze cache performance.
        if let Some(storage_stats) = snapshot.by_type.get(&OperationType::StorageRead) {
            let cache_rec = self.suggest_cache_sizing(storage_stats);

            if snapshot.cache_hit_rate < 0.5 && storage_stats.count >= 50 {
                let mut params = HashMap::new();
                params.insert(
                    "cache_capacity".to_string(),
                    cache_rec.suggested_capacity.to_string(),
                );
                params.insert(
                    "cache_ttl_secs".to_string(),
                    cache_rec.suggested_ttl_secs.to_string(),
                );
                params.insert(
                    "eviction_policy".to_string(),
                    cache_rec.suggested_eviction.clone(),
                );

                recommendations.push(Recommendation {
                    id: "cache-low-hit-rate".to_string(),
                    category: OptCategory::CacheTuning,
                    title: "Increase cache hit rate".to_string(),
                    description: format!(
                        "Cache hit rate is {:.1}%. {}",
                        snapshot.cache_hit_rate * 100.0,
                        cache_rec.reasoning
                    ),
                    impact: Impact::Medium,
                    effort: Effort::Trivial,
                    parameters: params,
                });
            }
        }

        // Analyze routing performance.
        if let Some(routing_stats) = snapshot.by_type.get(&OperationType::Routing) {
            if routing_stats.p95_us > target_p95_us {
                recommendations.push(Recommendation {
                    id: "routing-slow".to_string(),
                    category: OptCategory::Architecture,
                    title: "Reduce routing decision latency".to_string(),
                    description: format!(
                        "Routing P95 is {}us (target: {}us). Consider caching the \
                         FastGRNN model output or reducing feature computation overhead.",
                        routing_stats.p95_us, target_p95_us
                    ),
                    impact: Impact::Medium,
                    effort: Effort::Small,
                    parameters: HashMap::new(),
                });
            }
        }

        // Analyze injection performance.
        if let Some(inj_stats) = snapshot.by_type.get(&OperationType::Injection) {
            if inj_stats.p95_us > target_p95_us * 3 {
                recommendations.push(Recommendation {
                    id: "injection-slow".to_string(),
                    category: OptCategory::Architecture,
                    title: "Reduce injection formatting latency".to_string(),
                    description: format!(
                        "Injection P95 is {}us. Consider pre-computing E_nagual blocks \
                         or reducing the number of injected patterns.",
                        inj_stats.p95_us
                    ),
                    impact: Impact::Low,
                    effort: Effort::Small,
                    parameters: HashMap::new(),
                });
            }
        }

        // Check for high slow query count.
        if snapshot.slow_query_count > 0 {
            let slow_pct = snapshot.slow_query_count as f64 / snapshot.total_operations as f64;
            if slow_pct > 0.05 {
                recommendations.push(Recommendation {
                    id: "high-slow-query-rate".to_string(),
                    category: OptCategory::Architecture,
                    title: "High slow query rate".to_string(),
                    description: format!(
                        "{:.1}% of operations ({} / {}) exceed the slow query threshold. \
                         Review hot paths and consider index rebuilds.",
                        slow_pct * 100.0,
                        snapshot.slow_query_count,
                        snapshot.total_operations
                    ),
                    impact: Impact::High,
                    effort: Effort::Medium,
                    parameters: HashMap::new(),
                });
            }
        }

        // Sort by impact (High first).
        recommendations.sort_by_key(|r| match r.impact {
            Impact::High => 0,
            Impact::Medium => 1,
            Impact::Low => 2,
        });

        recommendations
    }

    /// Identify performance bottlenecks from a snapshot.
    pub fn identify_bottlenecks(&self, snapshot: &ProfileSnapshot) -> Vec<Bottleneck> {
        let target_p95_us = self.config.target_p95_ms * 1_000;
        let mut bottlenecks = Vec::new();

        for (op_type, stats) in &snapshot.by_type {
            if stats.p95_us > target_p95_us && stats.count >= 10 {
                let overshoot = stats.p95_us as f64 / target_p95_us as f64;
                let severity = (overshoot - 1.0).min(1.0).max(0.0);

                bottlenecks.push(Bottleneck {
                    operation: *op_type,
                    severity,
                    description: format!(
                        "{} P95 latency is {:.1}x the target ({} us vs {} us)",
                        op_type, overshoot, stats.p95_us, target_p95_us
                    ),
                    current_p95_us: stats.p95_us,
                    target_p95_us,
                });
            }
        }

        // Sort by severity descending.
        bottlenecks.sort_by(|a, b| {
            b.severity
                .partial_cmp(&a.severity)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        bottlenecks
    }

    /// Suggest HNSW parameters based on current search statistics.
    pub fn suggest_hnsw_params(&self, search_stats: &TypeStats) -> HnswRecommendation {
        let target_p95_us = self.config.target_p95_ms * 1_000;

        // If P95 is too high, reduce ef_search for speed.
        // If P95 is well below target, we can increase ef_search for better recall.
        let ratio = search_stats.p95_us as f64 / target_p95_us as f64;

        let (m, ef_c, ef_s, reasoning) = if ratio > 2.0 {
            // Very slow: reduce aggressively.
            (
                16,
                128,
                100,
                "Search latency is >2x target. Reducing ef_search and M to \
                 trade recall for speed. Verify recall stays above 0.95."
                    .to_string(),
            )
        } else if ratio > 1.0 {
            // Moderately slow: small reduction.
            (
                20,
                150,
                150,
                "Search latency slightly exceeds target. Moderate reduction \
                 in ef_search should bring P95 under target."
                    .to_string(),
            )
        } else if ratio < 0.3 {
            // Very fast: can increase quality.
            (
                32,
                300,
                300,
                "Search latency is well below target. Increasing parameters \
                 for better recall without exceeding the latency budget."
                    .to_string(),
            )
        } else {
            // Under target: current params are good.
            (
                24,
                200,
                200,
                "Current HNSW parameters are well-balanced for the target latency.".to_string(),
            )
        };

        HnswRecommendation {
            suggested_m: m,
            suggested_ef_construction: ef_c,
            suggested_ef_search: ef_s,
            reasoning,
        }
    }

    /// Suggest wormhole tuning based on hit rate and traversal stats.
    pub fn suggest_wormhole_tuning(
        &self,
        hit_rate: f64,
        traversal_stats: &TypeStats,
    ) -> WormholeRecommendation {
        let target_p95_us = self.config.target_p95_ms * 1_000;

        let (threshold, decay, max_per_node, reasoning) = if hit_rate < 0.1 {
            // Very low hit rate: lower the threshold to create more wormholes.
            (
                2_u32,
                0.03_f32,
                15_usize,
                "Wormhole hit rate is very low (<10%). Lowering the activation \
                 threshold from 3 to 2 and increasing max wormholes per node."
                    .to_string(),
            )
        } else if hit_rate < 0.3 {
            (
                2,
                0.04,
                12,
                "Wormhole hit rate is below 30%. Reducing activation threshold \
                 and slowing decay to retain more wormholes."
                    .to_string(),
            )
        } else if hit_rate > 0.8 && traversal_stats.p95_us < target_p95_us / 2 {
            // High hit rate and fast traversals: can tighten parameters.
            (
                4,
                0.08,
                8,
                "Wormhole hit rate is high and traversals are fast. Tightening \
                 thresholds to reduce memory overhead."
                    .to_string(),
            )
        } else {
            // Balanced.
            (
                3,
                0.05,
                10,
                "Current wormhole parameters appear well-balanced.".to_string(),
            )
        };

        WormholeRecommendation {
            suggested_activation_threshold: threshold,
            suggested_decay_rate: decay,
            suggested_max_per_node: max_per_node,
            reasoning,
        }
    }

    /// Suggest cache sizing based on storage read statistics.
    pub fn suggest_cache_sizing(&self, cache_stats: &TypeStats) -> CacheRecommendation {
        // If average latency is high, we benefit more from caching.
        let (capacity, ttl, eviction, reasoning) = if cache_stats.avg_us > 5_000.0 {
            // Slow reads: large cache, long TTL.
            (
                10_000_usize,
                600_u64,
                "LFU".to_string(),
                "Storage reads are slow (avg > 5ms). A large cache with LFU \
                 eviction will significantly reduce read latency."
                    .to_string(),
            )
        } else if cache_stats.avg_us > 1_000.0 {
            (
                5_000,
                300,
                "LRU".to_string(),
                "Storage reads are moderately slow. Increasing cache capacity \
                 should reduce P95 latency."
                    .to_string(),
            )
        } else {
            // Fast reads: moderate cache.
            (
                2_000,
                180,
                "LRU".to_string(),
                "Storage reads are relatively fast. A moderate cache is sufficient.".to_string(),
            )
        };

        CacheRecommendation {
            suggested_capacity: capacity,
            suggested_ttl_secs: ttl,
            suggested_eviction: eviction,
            reasoning,
        }
    }

    /// Estimate the impact of applying a recommendation.
    pub fn estimate_impact(&self, recommendation: &Recommendation) -> ImpactEstimate {
        let (speedup, confidence, reasoning) = match recommendation.category {
            OptCategory::HnswTuning => {
                let speedup = match recommendation.impact {
                    Impact::High => 30.0,
                    Impact::Medium => 15.0,
                    Impact::Low => 5.0,
                };
                (
                    speedup,
                    0.7,
                    "HNSW parameter tuning typically yields 15-40% latency improvement \
                     based on benchmark data."
                        .to_string(),
                )
            }
            OptCategory::WormholeTuning => {
                let speedup = match recommendation.impact {
                    Impact::High => 25.0,
                    Impact::Medium => 12.0,
                    Impact::Low => 5.0,
                };
                (
                    speedup,
                    0.5,
                    "Wormhole tuning impact depends on access patterns. Estimate is \
                     based on typical co-access distributions."
                        .to_string(),
                )
            }
            OptCategory::CacheTuning => {
                let speedup = match recommendation.impact {
                    Impact::High => 40.0,
                    Impact::Medium => 20.0,
                    Impact::Low => 8.0,
                };
                (
                    speedup,
                    0.6,
                    "Cache improvements have high impact when current hit rate is low. \
                     Estimate assumes the new configuration achieves > 80% hit rate."
                        .to_string(),
                )
            }
            OptCategory::Concurrency => (
                15.0,
                0.4,
                "Concurrency improvements are workload-dependent. Estimate assumes \
                 moderate contention reduction."
                    .to_string(),
            ),
            OptCategory::Storage => (
                20.0,
                0.5,
                "Storage optimizations typically yield 10-30% improvement depending \
                 on I/O patterns."
                    .to_string(),
            ),
            OptCategory::Architecture => {
                let speedup = match recommendation.impact {
                    Impact::High => 35.0,
                    Impact::Medium => 15.0,
                    Impact::Low => 5.0,
                };
                (
                    speedup,
                    0.3,
                    "Architectural changes have variable impact. Lower confidence \
                     reflects the broad scope of possible outcomes."
                        .to_string(),
                )
            }
        };

        ImpactEstimate {
            estimated_speedup_pct: speedup,
            confidence,
            reasoning,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper to create a TypeStats with the given values.
    fn make_stats(count: u64, p95_us: u64, avg_us: f64) -> TypeStats {
        TypeStats {
            count,
            min_us: 10,
            max_us: p95_us + 1000,
            avg_us,
            p50_us: avg_us as u64,
            p95_us,
            p99_us: p95_us + 500,
            success_count: count,
            failure_count: 0,
        }
    }

    fn make_snapshot(
        ops: u64,
        by_type: HashMap<OperationType, TypeStats>,
        wormhole_hit_rate: f64,
        cache_hit_rate: f64,
    ) -> ProfileSnapshot {
        ProfileSnapshot {
            total_operations: ops,
            by_type,
            slow_query_count: 0,
            wormhole_hit_rate,
            cache_hit_rate,
            uptime_secs: 60,
        }
    }

    #[test]
    fn test_optimizer_config_defaults() {
        let config = OptimizerConfig::default();
        assert!(!config.auto_optimize);
        assert_eq!(config.optimization_interval_secs, 3600);
        assert_eq!(config.min_samples, 100);
        assert_eq!(config.target_p95_ms, 50);
    }

    #[test]
    fn test_optimizer_config_builder() {
        let config = OptimizerConfig::builder()
            .auto_optimize(true)
            .optimization_interval_secs(1800)
            .min_samples(50)
            .target_p95_ms(30)
            .build();

        assert!(config.auto_optimize);
        assert_eq!(config.optimization_interval_secs, 1800);
        assert_eq!(config.min_samples, 50);
        assert_eq!(config.target_p95_ms, 30);
    }

    #[test]
    fn test_analyze_insufficient_samples() {
        let optimizer = ProfDAGOptimizer::with_defaults();
        let snapshot = make_snapshot(10, HashMap::new(), 0.0, 0.0);
        let recs = optimizer.analyze(&snapshot);
        assert!(recs.is_empty(), "Should return no recommendations with too few samples");
    }

    #[test]
    fn test_analyze_search_too_slow() {
        let optimizer = ProfDAGOptimizer::with_defaults(); // target_p95_ms = 50

        let mut by_type = HashMap::new();
        // P95 = 120_000us = 120ms, well above 50ms target.
        by_type.insert(OperationType::Search, make_stats(200, 120_000, 80_000.0));

        let snapshot = make_snapshot(200, by_type, 0.5, 0.8);
        let recs = optimizer.analyze(&snapshot);

        let hnsw_rec = recs.iter().find(|r| r.id == "hnsw-search-latency");
        assert!(hnsw_rec.is_some(), "Expected HNSW search latency recommendation");
        assert_eq!(hnsw_rec.unwrap().impact, Impact::High);
    }

    #[test]
    fn test_analyze_low_wormhole_hit_rate() {
        let optimizer = ProfDAGOptimizer::with_defaults();

        let mut by_type = HashMap::new();
        by_type.insert(
            OperationType::Traversal,
            make_stats(200, 80_000, 50_000.0),
        );

        let snapshot = make_snapshot(200, by_type, 0.1, 0.8);
        let recs = optimizer.analyze(&snapshot);

        let wh_rec = recs.iter().find(|r| r.id == "wormhole-low-hit-rate");
        assert!(wh_rec.is_some(), "Expected wormhole recommendation");
    }

    #[test]
    fn test_analyze_low_cache_hit_rate() {
        let optimizer = ProfDAGOptimizer::with_defaults();

        let mut by_type = HashMap::new();
        by_type.insert(
            OperationType::StorageRead,
            make_stats(200, 30_000, 20_000.0),
        );

        let snapshot = make_snapshot(200, by_type, 0.5, 0.2);
        let recs = optimizer.analyze(&snapshot);

        let cache_rec = recs.iter().find(|r| r.id == "cache-low-hit-rate");
        assert!(cache_rec.is_some(), "Expected cache recommendation");
    }

    #[test]
    fn test_identify_bottlenecks() {
        let optimizer = ProfDAGOptimizer::with_defaults();

        let mut by_type = HashMap::new();
        by_type.insert(OperationType::Search, make_stats(100, 30_000, 20_000.0)); // Under target
        by_type.insert(
            OperationType::Traversal,
            make_stats(100, 120_000, 80_000.0),
        ); // Over target

        let snapshot = make_snapshot(200, by_type, 0.5, 0.8);
        let bottlenecks = optimizer.identify_bottlenecks(&snapshot);

        assert_eq!(bottlenecks.len(), 1);
        assert_eq!(bottlenecks[0].operation, OperationType::Traversal);
        assert!(bottlenecks[0].severity > 0.0);
    }

    #[test]
    fn test_suggest_hnsw_params_very_slow() {
        let optimizer = ProfDAGOptimizer::with_defaults();
        let stats = make_stats(100, 150_000, 100_000.0); // 150ms P95, target 50ms => ratio > 2
        let rec = optimizer.suggest_hnsw_params(&stats);
        assert_eq!(rec.suggested_m, 16);
        assert_eq!(rec.suggested_ef_search, 100);
        assert!(rec.reasoning.contains("2x target"));
    }

    #[test]
    fn test_suggest_hnsw_params_fast() {
        let optimizer = ProfDAGOptimizer::with_defaults();
        let stats = make_stats(100, 10_000, 5_000.0); // 10ms P95, target 50ms => ratio < 0.3
        let rec = optimizer.suggest_hnsw_params(&stats);
        assert_eq!(rec.suggested_m, 32);
        assert_eq!(rec.suggested_ef_search, 300);
    }

    #[test]
    fn test_suggest_wormhole_tuning_low_hit_rate() {
        let optimizer = ProfDAGOptimizer::with_defaults();
        let stats = make_stats(100, 80_000, 50_000.0);
        let rec = optimizer.suggest_wormhole_tuning(0.05, &stats);
        assert_eq!(rec.suggested_activation_threshold, 2);
        assert!(rec.reasoning.contains("very low"));
    }

    #[test]
    fn test_suggest_cache_sizing_slow_reads() {
        let optimizer = ProfDAGOptimizer::with_defaults();
        let stats = make_stats(100, 20_000, 8_000.0); // avg > 5ms
        let rec = optimizer.suggest_cache_sizing(&stats);
        assert_eq!(rec.suggested_capacity, 10_000);
        assert_eq!(rec.suggested_eviction, "LFU");
    }

    #[test]
    fn test_estimate_impact() {
        let optimizer = ProfDAGOptimizer::with_defaults();

        let rec = Recommendation {
            id: "test".to_string(),
            category: OptCategory::HnswTuning,
            title: "Test".to_string(),
            description: "Test".to_string(),
            impact: Impact::High,
            effort: Effort::Trivial,
            parameters: HashMap::new(),
        };

        let impact = optimizer.estimate_impact(&rec);
        assert!(impact.estimated_speedup_pct > 0.0);
        assert!(impact.confidence > 0.0 && impact.confidence <= 1.0);
        assert!(!impact.reasoning.is_empty());
    }

    #[test]
    fn test_recommendations_sorted_by_impact() {
        let optimizer = ProfDAGOptimizer::with_defaults();

        let mut by_type = HashMap::new();
        // All slow to trigger multiple recommendations.
        by_type.insert(OperationType::Search, make_stats(200, 120_000, 80_000.0));
        by_type.insert(
            OperationType::Traversal,
            make_stats(200, 120_000, 80_000.0),
        );
        by_type.insert(
            OperationType::StorageRead,
            make_stats(200, 120_000, 80_000.0),
        );

        let snapshot = ProfileSnapshot {
            total_operations: 600,
            by_type,
            slow_query_count: 100,
            wormhole_hit_rate: 0.1,
            cache_hit_rate: 0.2,
            uptime_secs: 60,
        };

        let recs = optimizer.analyze(&snapshot);
        assert!(recs.len() >= 2);

        // First recommendation should be High impact.
        assert_eq!(recs[0].impact, Impact::High);
    }

    #[test]
    fn test_bottleneck_severity_clamped() {
        let optimizer = ProfDAGOptimizer::with_defaults();

        let mut by_type = HashMap::new();
        // Extremely over target: 500ms vs 50ms = 10x.
        by_type.insert(
            OperationType::Search,
            make_stats(100, 500_000, 300_000.0),
        );

        let snapshot = make_snapshot(100, by_type, 0.5, 0.8);
        let bottlenecks = optimizer.identify_bottlenecks(&snapshot);
        assert_eq!(bottlenecks.len(), 1);
        // Severity should be clamped to 1.0.
        assert!((bottlenecks[0].severity - 1.0).abs() < f64::EPSILON);
    }
}
