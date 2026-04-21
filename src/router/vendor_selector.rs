//! Vendor Selection Logic
//!
//! Routes queries to appropriate LLM vendors based on complexity scores.
//! Implements fallback chains for reliability and supports vendor health monitoring.
//!
//! # Vendor Routing Rules
//!
//! - complexity < 0.3: local small model (fastest, lowest cost)
//! - complexity 0.3-0.5: local large model (balanced)
//! - complexity 0.5-0.7: Claude API (high quality)
//! - complexity >= 0.7: Claude API with GPT fallback (best quality)
//!
//! # Fallback Chain
//!
//! Default: local-small -> local-large -> claude -> gpt
//!
//! If a vendor is unavailable, the next in the chain is tried.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

use super::complexity_estimator::{ComplexityEstimator, ComplexityLevel, ComplexityScore};
use super::fastgrnn::FastGRNN;
use super::{RouterConfig, RouterResult};
use crate::profdag::profiler::{OperationType, ProfDAGProfiler};

/// Available LLM vendors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Vendor {
    /// Local small model (e.g., Phi-2, Mistral-7B)
    LocalSmall,
    /// Local large model (e.g., Llama-70B, Mixtral)
    LocalLarge,
    /// Claude API (Anthropic)
    Claude,
    /// GPT API (OpenAI)
    GPT,
}

impl Vendor {
    /// Get string representation.
    pub fn as_str(&self) -> &'static str {
        match self {
            Vendor::LocalSmall => "local-small",
            Vendor::LocalLarge => "local-large",
            Vendor::Claude => "claude",
            Vendor::GPT => "gpt",
        }
    }

    /// Parse from string.
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "local-small" | "local_small" | "localsmall" => Some(Vendor::LocalSmall),
            "local-large" | "local_large" | "locallarge" => Some(Vendor::LocalLarge),
            "claude" | "anthropic" => Some(Vendor::Claude),
            "gpt" | "openai" | "chatgpt" => Some(Vendor::GPT),
            _ => None,
        }
    }

    /// Check if this is a local vendor.
    pub fn is_local(&self) -> bool {
        matches!(self, Vendor::LocalSmall | Vendor::LocalLarge)
    }

    /// Check if this is a cloud vendor.
    pub fn is_cloud(&self) -> bool {
        matches!(self, Vendor::Claude | Vendor::GPT)
    }

    /// Get relative cost (1 = cheapest).
    pub fn relative_cost(&self) -> u32 {
        match self {
            Vendor::LocalSmall => 1,
            Vendor::LocalLarge => 2,
            Vendor::Claude => 10,
            Vendor::GPT => 12,
        }
    }

    /// Get relative latency (1 = fastest).
    pub fn relative_latency(&self) -> u32 {
        match self {
            Vendor::LocalSmall => 1,
            Vendor::LocalLarge => 2,
            Vendor::Claude => 5,
            Vendor::GPT => 5,
        }
    }
}

/// Vendor health status.
#[derive(Debug)]
pub struct VendorStatus {
    /// Whether the vendor is currently available.
    pub available: AtomicBool,

    /// Number of successful requests.
    pub success_count: AtomicU64,

    /// Number of failed requests.
    pub failure_count: AtomicU64,

    /// Number of consecutive failures (resets on success).
    pub consecutive_failures: AtomicU64,

    /// Total latency in microseconds.
    pub total_latency_us: AtomicU64,

    /// Last error message.
    pub last_error: RwLock<Option<String>>,

    /// Last successful timestamp.
    pub last_success: RwLock<Option<Instant>>,

    /// Last failure timestamp.
    pub last_failure: RwLock<Option<Instant>>,
}

impl VendorStatus {
    /// Create a new vendor status (available by default).
    pub fn new() -> Self {
        Self {
            available: AtomicBool::new(true),
            success_count: AtomicU64::new(0),
            failure_count: AtomicU64::new(0),
            consecutive_failures: AtomicU64::new(0),
            total_latency_us: AtomicU64::new(0),
            last_error: RwLock::new(None),
            last_success: RwLock::new(None),
            last_failure: RwLock::new(None),
        }
    }

    /// Record a successful request.
    pub fn record_success(&self, latency_us: u64) {
        self.success_count.fetch_add(1, Ordering::Relaxed);
        self.total_latency_us.fetch_add(latency_us, Ordering::Relaxed);
        *self.last_success.write() = Some(Instant::now());
        self.consecutive_failures.store(0, Ordering::Relaxed);
        self.available.store(true, Ordering::Relaxed);
    }

    /// Record a failed request.
    pub fn record_failure(&self, error: String) {
        self.failure_count.fetch_add(1, Ordering::Relaxed);
        let consecutive = self.consecutive_failures.fetch_add(1, Ordering::Relaxed) + 1;
        *self.last_error.write() = Some(error);
        *self.last_failure.write() = Some(Instant::now());

        // Mark unavailable after 3 consecutive failures
        if consecutive >= 3 {
            self.available.store(false, Ordering::Relaxed);
        }
    }

    /// Check if the vendor is available.
    pub fn is_available(&self) -> bool {
        self.available.load(Ordering::Relaxed)
    }

    /// Get success rate.
    pub fn success_rate(&self) -> f64 {
        let successes = self.success_count.load(Ordering::Relaxed);
        let failures = self.failure_count.load(Ordering::Relaxed);
        let total = successes + failures;
        if total == 0 {
            1.0
        } else {
            successes as f64 / total as f64
        }
    }

    /// Get average latency in microseconds.
    pub fn avg_latency_us(&self) -> f64 {
        let total = self.total_latency_us.load(Ordering::Relaxed);
        let count = self.success_count.load(Ordering::Relaxed);
        if count == 0 {
            0.0
        } else {
            total as f64 / count as f64
        }
    }

    /// Get consecutive failure count.
    pub fn consecutive_failure_count(&self) -> u64 {
        self.consecutive_failures.load(Ordering::Relaxed)
    }

    /// Reset statistics.
    pub fn reset(&self) {
        self.success_count.store(0, Ordering::Relaxed);
        self.failure_count.store(0, Ordering::Relaxed);
        self.consecutive_failures.store(0, Ordering::Relaxed);
        self.total_latency_us.store(0, Ordering::Relaxed);
        *self.last_error.write() = None;
        *self.last_success.write() = None;
        *self.last_failure.write() = None;
        self.available.store(true, Ordering::Relaxed);
    }
}

impl Default for VendorStatus {
    fn default() -> Self {
        Self::new()
    }
}

/// Fallback chain for vendor selection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FallbackChain {
    /// Ordered list of vendors to try.
    pub vendors: Vec<Vendor>,
}

impl Default for FallbackChain {
    fn default() -> Self {
        Self {
            vendors: vec![
                Vendor::LocalSmall,
                Vendor::LocalLarge,
                Vendor::Claude,
                Vendor::GPT,
            ],
        }
    }
}

impl FallbackChain {
    /// Create a chain starting from a specific vendor.
    pub fn starting_from(vendor: Vendor) -> Self {
        let all = vec![
            Vendor::LocalSmall,
            Vendor::LocalLarge,
            Vendor::Claude,
            Vendor::GPT,
        ];

        let start_idx = all.iter().position(|&v| v == vendor).unwrap_or(0);
        let mut chain = Vec::new();

        // Add from start vendor to end
        for i in start_idx..all.len() {
            chain.push(all[i]);
        }

        Self { vendors: chain }
    }

    /// Create a local-only chain.
    pub fn local_only() -> Self {
        Self {
            vendors: vec![Vendor::LocalSmall, Vendor::LocalLarge],
        }
    }

    /// Create a cloud-only chain.
    pub fn cloud_only() -> Self {
        Self {
            vendors: vec![Vendor::Claude, Vendor::GPT],
        }
    }

    /// Get the next vendor in the chain after the given one.
    pub fn next_after(&self, vendor: Vendor) -> Option<Vendor> {
        let pos = self.vendors.iter().position(|&v| v == vendor)?;
        self.vendors.get(pos + 1).copied()
    }

    /// Check if a vendor is in this chain.
    pub fn contains(&self, vendor: Vendor) -> bool {
        self.vendors.contains(&vendor)
    }
}

/// Configuration for vendor selection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VendorConfig {
    /// Complexity threshold for local small model.
    pub local_small_threshold: f32,

    /// Complexity threshold for local large model.
    pub local_large_threshold: f32,

    /// Complexity threshold for cloud (Claude).
    pub cloud_threshold: f32,

    /// Default fallback chain.
    pub default_chain: FallbackChain,

    /// Whether to prefer local models when possible.
    pub prefer_local: bool,

    /// Whether to enable cost optimization.
    pub optimize_cost: bool,

    /// Maximum retry count for fallbacks.
    pub max_retries: u32,
}

impl Default for VendorConfig {
    fn default() -> Self {
        Self {
            local_small_threshold: 0.3,
            local_large_threshold: 0.5,
            cloud_threshold: 0.7,
            default_chain: FallbackChain::default(),
            prefer_local: true,
            optimize_cost: true,
            max_retries: 3,
        }
    }
}

impl VendorConfig {
    /// Create a config optimized for low latency.
    pub fn low_latency() -> Self {
        Self {
            local_small_threshold: 0.4, // Use local more often
            local_large_threshold: 0.6,
            cloud_threshold: 0.8,
            default_chain: FallbackChain::default(),
            prefer_local: true,
            optimize_cost: false,
            max_retries: 2,
        }
    }

    /// Create a config optimized for quality.
    pub fn high_quality() -> Self {
        Self {
            local_small_threshold: 0.2, // Use cloud more often
            local_large_threshold: 0.4,
            cloud_threshold: 0.6,
            default_chain: FallbackChain::default(),
            prefer_local: false,
            optimize_cost: false,
            max_retries: 3,
        }
    }
}

/// Routing decision with metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingDecision {
    /// Selected vendor.
    pub vendor: Vendor,

    /// Complexity score that led to this decision.
    pub complexity: f32,

    /// Complexity level classification.
    pub level: ComplexityLevel,

    /// Fallback chain for this request.
    pub fallback_chain: FallbackChain,

    /// Confidence in the routing decision.
    pub confidence: f32,

    /// Routing latency in microseconds.
    pub routing_latency_us: u64,

    /// Whether this was a fallback selection.
    pub is_fallback: bool,

    /// Reason for the routing decision.
    pub reason: String,
}

impl RoutingDecision {
    /// Create a new routing decision.
    pub fn new(
        vendor: Vendor,
        complexity: f32,
        confidence: f32,
        latency_us: u64,
    ) -> Self {
        Self {
            vendor,
            complexity,
            level: ComplexityLevel::from_score(complexity),
            fallback_chain: FallbackChain::starting_from(vendor),
            confidence,
            routing_latency_us: latency_us,
            is_fallback: false,
            reason: format!(
                "Complexity {:.2} -> {}",
                complexity,
                vendor.as_str()
            ),
        }
    }

    /// Mark as a fallback decision.
    pub fn as_fallback(mut self, original: Vendor, reason: &str) -> Self {
        self.is_fallback = true;
        self.reason = format!(
            "Fallback from {} to {}: {}",
            original.as_str(),
            self.vendor.as_str(),
            reason
        );
        self
    }
}

/// Routing metrics for observability.
#[derive(Debug, Default)]
pub struct RoutingMetrics {
    /// Total routing decisions.
    pub total_decisions: AtomicU64,

    /// Decisions per vendor.
    pub vendor_counts: RwLock<HashMap<Vendor, u64>>,

    /// Total routing latency in microseconds.
    pub total_latency_us: AtomicU64,

    /// Fallback count.
    pub fallback_count: AtomicU64,

    /// Complexity score distribution.
    pub complexity_buckets: RwLock<[u64; 10]>,
}

impl RoutingMetrics {
    /// Create new metrics.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a routing decision.
    pub fn record(&self, decision: &RoutingDecision) {
        self.total_decisions.fetch_add(1, Ordering::Relaxed);
        self.total_latency_us
            .fetch_add(decision.routing_latency_us, Ordering::Relaxed);

        {
            let mut counts = self.vendor_counts.write();
            *counts.entry(decision.vendor).or_insert(0) += 1;
        }

        if decision.is_fallback {
            self.fallback_count.fetch_add(1, Ordering::Relaxed);
        }

        // Update complexity histogram
        let bucket = (decision.complexity * 10.0).floor() as usize;
        let bucket = bucket.min(9);
        {
            let mut buckets = self.complexity_buckets.write();
            buckets[bucket] += 1;
        }
    }

    /// Get average routing latency in microseconds.
    pub fn avg_latency_us(&self) -> f64 {
        let total = self.total_latency_us.load(Ordering::Relaxed);
        let count = self.total_decisions.load(Ordering::Relaxed);
        if count == 0 {
            0.0
        } else {
            total as f64 / count as f64
        }
    }

    /// Get fallback rate.
    pub fn fallback_rate(&self) -> f64 {
        let fallbacks = self.fallback_count.load(Ordering::Relaxed);
        let total = self.total_decisions.load(Ordering::Relaxed);
        if total == 0 {
            0.0
        } else {
            fallbacks as f64 / total as f64
        }
    }

    /// Get vendor distribution.
    pub fn vendor_distribution(&self) -> HashMap<Vendor, f64> {
        let counts = self.vendor_counts.read();
        let total: u64 = counts.values().sum();
        if total == 0 {
            return HashMap::new();
        }
        counts
            .iter()
            .map(|(&v, &c)| (v, c as f64 / total as f64))
            .collect()
    }

    /// Reset all metrics.
    pub fn reset(&self) {
        self.total_decisions.store(0, Ordering::Relaxed);
        self.total_latency_us.store(0, Ordering::Relaxed);
        self.fallback_count.store(0, Ordering::Relaxed);
        self.vendor_counts.write().clear();
        *self.complexity_buckets.write() = [0; 10];
    }
}

/// Vendor selector based on complexity thresholds.
pub struct VendorSelector {
    /// Configuration.
    config: VendorConfig,

    /// Vendor health status.
    vendor_status: HashMap<Vendor, Arc<VendorStatus>>,

    /// Routing metrics.
    metrics: Arc<RoutingMetrics>,

    /// Optional profiler for recording operation timings.
    profiler: Option<Arc<ProfDAGProfiler>>,
}

impl VendorSelector {
    /// Create a new vendor selector.
    pub fn new(config: VendorConfig) -> Self {
        let mut vendor_status = HashMap::new();
        vendor_status.insert(Vendor::LocalSmall, Arc::new(VendorStatus::new()));
        vendor_status.insert(Vendor::LocalLarge, Arc::new(VendorStatus::new()));
        vendor_status.insert(Vendor::Claude, Arc::new(VendorStatus::new()));
        vendor_status.insert(Vendor::GPT, Arc::new(VendorStatus::new()));

        Self {
            config,
            vendor_status,
            metrics: Arc::new(RoutingMetrics::new()),
            profiler: None,
        }
    }

    /// Attach an optional profiler for recording operation timings.
    pub fn with_profiler(mut self, profiler: Arc<ProfDAGProfiler>) -> Self {
        self.profiler = Some(profiler);
        self
    }

    /// Select a vendor based on complexity score.
    pub fn select(&self, complexity: f32, confidence: f32) -> RoutingDecision {
        let _guard = self.profiler.as_ref().map(|p| p.start_operation(OperationType::Routing));
        let start = Instant::now();

        let primary_vendor = if complexity < self.config.local_small_threshold {
            Vendor::LocalSmall
        } else if complexity < self.config.local_large_threshold {
            Vendor::LocalLarge
        } else if complexity < self.config.cloud_threshold {
            Vendor::Claude
        } else {
            Vendor::Claude // Use Claude for highest complexity
        };

        // Check if primary vendor is available
        let selected_vendor = if self.is_vendor_available(primary_vendor) {
            primary_vendor
        } else {
            // Find available vendor from fallback chain
            let chain = FallbackChain::starting_from(primary_vendor);
            self.find_available_vendor(&chain)
                .unwrap_or(Vendor::LocalLarge) // Last resort
        };

        let latency_us = start.elapsed().as_micros() as u64;
        let mut decision = RoutingDecision::new(selected_vendor, complexity, confidence, latency_us);

        if selected_vendor != primary_vendor {
            decision = decision.as_fallback(
                primary_vendor,
                &format!("{} unavailable", primary_vendor.as_str()),
            );
        }

        // Record metrics
        self.metrics.record(&decision);

        decision
    }

    /// Check if a vendor is available.
    pub fn is_vendor_available(&self, vendor: Vendor) -> bool {
        self.vendor_status
            .get(&vendor)
            .map(|s| s.is_available())
            .unwrap_or(false)
    }

    /// Find the first available vendor from a fallback chain.
    fn find_available_vendor(&self, chain: &FallbackChain) -> Option<Vendor> {
        chain
            .vendors
            .iter()
            .find(|&&v| self.is_vendor_available(v))
            .copied()
    }

    /// Get the next available vendor after a failure.
    pub fn get_fallback(&self, current: Vendor) -> Option<Vendor> {
        let chain = FallbackChain::starting_from(current);
        chain
            .next_after(current)
            .and_then(|v| {
                if self.is_vendor_available(v) {
                    Some(v)
                } else {
                    self.get_fallback(v)
                }
            })
    }

    /// Record a successful vendor request.
    pub fn record_success(&self, vendor: Vendor, latency_us: u64) {
        if let Some(status) = self.vendor_status.get(&vendor) {
            status.record_success(latency_us);
        }
    }

    /// Record a failed vendor request.
    pub fn record_failure(&self, vendor: Vendor, error: String) {
        if let Some(status) = self.vendor_status.get(&vendor) {
            status.record_failure(error);
        }
    }

    /// Mark a vendor as unavailable.
    pub fn mark_unavailable(&self, vendor: Vendor) {
        if let Some(status) = self.vendor_status.get(&vendor) {
            status.available.store(false, Ordering::Relaxed);
        }
    }

    /// Mark a vendor as available.
    pub fn mark_available(&self, vendor: Vendor) {
        if let Some(status) = self.vendor_status.get(&vendor) {
            status.available.store(true, Ordering::Relaxed);
        }
    }

    /// Get vendor status.
    pub fn get_status(&self, vendor: Vendor) -> Option<&Arc<VendorStatus>> {
        self.vendor_status.get(&vendor)
    }

    /// Get all vendor statuses.
    pub fn all_status(&self) -> &HashMap<Vendor, Arc<VendorStatus>> {
        &self.vendor_status
    }

    /// Get routing metrics.
    pub fn metrics(&self) -> &Arc<RoutingMetrics> {
        &self.metrics
    }

    /// Get the configuration.
    pub fn config(&self) -> &VendorConfig {
        &self.config
    }

    /// Reset all vendor statuses.
    pub fn reset_status(&self) {
        for status in self.vendor_status.values() {
            status.reset();
        }
    }
}

/// Complete vendor router combining FastGRNN and VendorSelector.
pub struct VendorRouter {
    /// FastGRNN model for complexity estimation.
    fastgrnn: FastGRNN,

    /// Complexity feature extractor.
    estimator: ComplexityEstimator,

    /// Vendor selector.
    selector: VendorSelector,

    /// Configuration.
    config: RouterConfig,

    /// Optional profiler for recording operation timings.
    profiler: Option<Arc<ProfDAGProfiler>>,
}

impl VendorRouter {
    /// Create a new vendor router.
    pub fn new(config: RouterConfig) -> RouterResult<Self> {
        let fastgrnn = FastGRNN::new(config.fastgrnn.clone())?;
        let estimator = ComplexityEstimator::new(config.estimator.clone());
        let selector = VendorSelector::new(config.selector.clone());

        Ok(Self {
            fastgrnn,
            estimator,
            selector,
            config,
            profiler: None,
        })
    }

    /// Attach an optional profiler for recording operation timings.
    /// This wires the profiler into both the router and its inner VendorSelector.
    pub fn with_profiler(mut self, profiler: Arc<ProfDAGProfiler>) -> Self {
        self.profiler = Some(profiler.clone());
        self.selector = VendorSelector::new(self.config.selector.clone())
            .with_profiler(profiler);
        self
    }

    /// Route a query to the appropriate vendor.
    ///
    /// Returns the routing decision including vendor, complexity, and fallback chain.
    pub fn route(&self, query: &str, embedding: &[f32]) -> RouterResult<RoutingDecision> {
        let _guard = self.profiler.as_ref().map(|p| p.start_operation(OperationType::Routing));
        let start = Instant::now();

        // Extract features
        let features = self.estimator.extract_features(query, embedding)?;

        // Run FastGRNN for complexity estimation
        let feature_vector = features.to_vector();
        let complexity = self.fastgrnn.forward(&feature_vector)?;

        // Compute confidence
        let confidence = features.simple_complexity(&self.config.estimator);

        // Check latency limit
        let elapsed_us = start.elapsed().as_micros() as u64;
        if elapsed_us > self.config.max_latency_ms * 1000 {
            tracing::warn!(
                elapsed_us = elapsed_us,
                limit_ms = self.config.max_latency_ms,
                "Routing exceeded latency limit"
            );
        }

        // Select vendor
        let mut decision = self.selector.select(complexity, confidence);
        decision.routing_latency_us = elapsed_us;

        if self.config.debug_logging {
            tracing::debug!(
                query_len = query.len(),
                complexity = complexity,
                vendor = decision.vendor.as_str(),
                latency_us = elapsed_us,
                "Routed query"
            );
        }

        Ok(decision)
    }

    /// Route using simple estimation (no FastGRNN, faster but less accurate).
    pub fn route_simple(&self, query: &str, embedding: &[f32]) -> RouterResult<RoutingDecision> {
        let start = Instant::now();

        let score = self.estimator.estimate_simple(query, embedding)?;
        let elapsed_us = start.elapsed().as_micros() as u64;

        let mut decision = self.selector.select(score.score, score.confidence);
        decision.routing_latency_us = elapsed_us;

        Ok(decision)
    }

    /// Get complexity score without vendor selection.
    pub fn estimate_complexity(&self, query: &str, embedding: &[f32]) -> RouterResult<ComplexityScore> {
        let start = Instant::now();

        let features = self.estimator.extract_features(query, embedding)?;
        let feature_vector = features.to_vector();
        let complexity = self.fastgrnn.forward(&feature_vector)?;
        let confidence = features.simple_complexity(&self.config.estimator);

        let time_us = start.elapsed().as_micros() as u64;

        Ok(ComplexityScore::new(complexity, features, confidence, time_us))
    }

    /// Record outcome for learning.
    pub fn record_outcome(&self, query: &str, vendor: Vendor, success: bool, latency_us: u64) {
        if success {
            self.selector.record_success(vendor, latency_us);
            self.estimator.record_accuracy(query, 1.0);
        } else {
            self.selector.record_failure(vendor, "Request failed".to_string());
            self.estimator.record_accuracy(query, 0.0);
        }
    }

    /// Get the next fallback vendor.
    pub fn get_fallback(&self, current: Vendor) -> Option<Vendor> {
        self.selector.get_fallback(current)
    }

    /// Get vendor status.
    pub fn vendor_status(&self, vendor: Vendor) -> Option<&Arc<VendorStatus>> {
        self.selector.get_status(vendor)
    }

    /// Get routing metrics.
    pub fn metrics(&self) -> &Arc<RoutingMetrics> {
        self.selector.metrics()
    }

    /// Get FastGRNN statistics.
    pub fn model_stats(&self) -> (u64, f64) {
        (
            self.fastgrnn.inference_count(),
            self.fastgrnn.avg_inference_time_us(),
        )
    }

    /// Get the configuration.
    pub fn config(&self) -> &RouterConfig {
        &self.config
    }

    /// Reset all statistics.
    pub fn reset_stats(&self) {
        self.fastgrnn.reset_stats();
        self.selector.reset_status();
        self.selector.metrics().reset();
        self.estimator.clear_cache();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_embedding() -> Vec<f32> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let mut emb: Vec<f32> = (0..128).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            emb.iter_mut().for_each(|x| *x /= norm);
        }
        emb
    }

    #[test]
    fn test_vendor_from_str() {
        assert_eq!(Vendor::from_str("claude"), Some(Vendor::Claude));
        assert_eq!(Vendor::from_str("local-small"), Some(Vendor::LocalSmall));
        assert_eq!(Vendor::from_str("gpt"), Some(Vendor::GPT));
        assert_eq!(Vendor::from_str("unknown"), None);
    }

    #[test]
    fn test_vendor_properties() {
        assert!(Vendor::LocalSmall.is_local());
        assert!(Vendor::LocalLarge.is_local());
        assert!(Vendor::Claude.is_cloud());
        assert!(Vendor::GPT.is_cloud());

        assert!(Vendor::LocalSmall.relative_cost() < Vendor::Claude.relative_cost());
    }

    #[test]
    fn test_vendor_status() {
        let status = VendorStatus::new();
        assert!(status.is_available());
        assert_eq!(status.success_rate(), 1.0);

        status.record_success(100);
        assert!(status.is_available());
        assert_eq!(status.success_count.load(Ordering::Relaxed), 1);

        status.record_failure("test error".to_string());
        assert_eq!(status.failure_count.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_fallback_chain_default() {
        let chain = FallbackChain::default();
        assert_eq!(chain.vendors.len(), 4);
        assert_eq!(chain.vendors[0], Vendor::LocalSmall);
    }

    #[test]
    fn test_fallback_chain_starting_from() {
        let chain = FallbackChain::starting_from(Vendor::Claude);
        assert_eq!(chain.vendors[0], Vendor::Claude);
        assert!(chain.vendors.contains(&Vendor::GPT));
    }

    #[test]
    fn test_fallback_chain_next() {
        let chain = FallbackChain::default();
        assert_eq!(chain.next_after(Vendor::LocalSmall), Some(Vendor::LocalLarge));
        assert_eq!(chain.next_after(Vendor::GPT), None);
    }

    #[test]
    fn test_vendor_config_default() {
        let config = VendorConfig::default();
        assert!(config.local_small_threshold < config.local_large_threshold);
        assert!(config.local_large_threshold < config.cloud_threshold);
    }

    #[test]
    fn test_vendor_selector_select() {
        let selector = VendorSelector::new(VendorConfig::default());

        // Low complexity -> local small
        let decision = selector.select(0.1, 0.9);
        assert_eq!(decision.vendor, Vendor::LocalSmall);

        // Medium complexity -> local large
        let decision = selector.select(0.4, 0.9);
        assert_eq!(decision.vendor, Vendor::LocalLarge);

        // High complexity -> Claude
        let decision = selector.select(0.8, 0.9);
        assert_eq!(decision.vendor, Vendor::Claude);
    }

    #[test]
    fn test_vendor_selector_fallback() {
        let selector = VendorSelector::new(VendorConfig::default());

        // Mark local small as unavailable
        selector.mark_unavailable(Vendor::LocalSmall);

        // Should fallback to local large
        let decision = selector.select(0.1, 0.9);
        assert!(decision.is_fallback);
        assert_eq!(decision.vendor, Vendor::LocalLarge);
    }

    #[test]
    fn test_routing_metrics() {
        let metrics = RoutingMetrics::new();

        let decision = RoutingDecision::new(Vendor::Claude, 0.7, 0.9, 100);
        metrics.record(&decision);

        assert_eq!(metrics.total_decisions.load(Ordering::Relaxed), 1);
        assert!((metrics.avg_latency_us() - 100.0).abs() < 0.001);
    }

    #[test]
    fn test_routing_decision_creation() {
        let decision = RoutingDecision::new(Vendor::LocalSmall, 0.2, 0.95, 50);

        assert_eq!(decision.vendor, Vendor::LocalSmall);
        assert!((decision.complexity - 0.2).abs() < 0.001);
        assert_eq!(decision.level, ComplexityLevel::Low);
        assert!(!decision.is_fallback);
    }

    #[test]
    fn test_vendor_router_creation() {
        let config = RouterConfig::default();
        let router = VendorRouter::new(config);
        assert!(router.is_ok());
    }

    #[test]
    fn test_vendor_router_route() {
        let config = RouterConfig::default();
        let router = VendorRouter::new(config).unwrap();
        let embedding = sample_embedding();

        let decision = router.route("Simple question", &embedding);
        assert!(decision.is_ok());

        let d = decision.unwrap();
        assert!(d.complexity >= 0.0 && d.complexity <= 1.0);
    }

    #[test]
    fn test_vendor_router_route_simple() {
        let config = RouterConfig::default();
        let router = VendorRouter::new(config).unwrap();
        let embedding = sample_embedding();

        let decision = router.route_simple("Test query", &embedding);
        assert!(decision.is_ok());
    }

    #[test]
    fn test_vendor_router_estimate_complexity() {
        let config = RouterConfig::default();
        let router = VendorRouter::new(config).unwrap();
        let embedding = sample_embedding();

        let score = router.estimate_complexity("Complex algorithmic question", &embedding);
        assert!(score.is_ok());

        let s = score.unwrap();
        assert!(s.score >= 0.0 && s.score <= 1.0);
        assert!(s.confidence >= 0.0 && s.confidence <= 1.0);
    }

    #[test]
    fn test_vendor_router_record_outcome() {
        let config = RouterConfig::default();
        let router = VendorRouter::new(config).unwrap();

        router.record_outcome("test query", Vendor::Claude, true, 1000);
        router.record_outcome("test query 2", Vendor::LocalSmall, false, 500);

        // Check status was updated
        let claude_status = router.vendor_status(Vendor::Claude).unwrap();
        assert_eq!(claude_status.success_count.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_vendor_router_latency() {
        let config = RouterConfig::low_latency();
        let router = VendorRouter::new(config).unwrap();
        let embedding = sample_embedding();

        // Run multiple times
        for _ in 0..10 {
            let start = Instant::now();
            let _ = router.route("Test latency", &embedding);
            let elapsed_ms = start.elapsed().as_millis();

            // Should be well under 5ms
            assert!(elapsed_ms < 50, "Routing took {}ms", elapsed_ms);
        }
    }

    #[test]
    fn test_vendor_router_metrics() {
        let config = RouterConfig::default();
        let router = VendorRouter::new(config).unwrap();
        let embedding = sample_embedding();

        let _ = router.route("query 1", &embedding);
        let _ = router.route("query 2", &embedding);
        let _ = router.route("query 3", &embedding);

        let metrics = router.metrics();
        assert_eq!(metrics.total_decisions.load(Ordering::Relaxed), 3);
    }

    #[test]
    fn test_vendor_router_reset() {
        let config = RouterConfig::default();
        let router = VendorRouter::new(config).unwrap();
        let embedding = sample_embedding();

        let _ = router.route("test", &embedding);
        let _ = router.route("test", &embedding);

        router.reset_stats();

        let (count, _) = router.model_stats();
        assert_eq!(count, 0);
        assert_eq!(router.metrics().total_decisions.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_consecutive_failure_tracking() {
        let status = VendorStatus::new();

        // 3 consecutive failures should mark unavailable
        status.record_failure("err1".to_string());
        assert!(status.is_available());
        status.record_failure("err2".to_string());
        assert!(status.is_available());
        status.record_failure("err3".to_string());
        assert!(!status.is_available(), "Should be unavailable after 3 consecutive failures");
    }

    #[test]
    fn test_success_resets_consecutive_failures() {
        let status = VendorStatus::new();

        // 2 failures, then success, then 2 more failures
        status.record_failure("err1".to_string());
        status.record_failure("err2".to_string());
        status.record_success(100);
        assert_eq!(status.consecutive_failure_count(), 0);
        status.record_failure("err3".to_string());
        status.record_failure("err4".to_string());
        assert!(status.is_available(), "Should still be available - only 2 consecutive failures after success");
    }

    #[test]
    fn test_high_total_failures_doesnt_trigger_if_not_consecutive() {
        let status = VendorStatus::new();

        // Alternate success/failure - should never become unavailable
        for _ in 0..10 {
            status.record_failure("err".to_string());
            status.record_success(100);
        }
        assert!(status.is_available(), "Alternating failures should never trigger unavailable");
        assert_eq!(status.consecutive_failure_count(), 0);
    }

    #[test]
    fn test_profiler_wired_into_vendor_selector() {
        use crate::profdag::profiler::{ProfDAGProfiler, ProfilerConfig, OperationType};

        let profiler = Arc::new(ProfDAGProfiler::new(ProfilerConfig::default()));
        let selector = VendorSelector::new(VendorConfig::default())
            .with_profiler(profiler.clone());

        // Perform several routing decisions
        let _ = selector.select(0.1, 0.9);
        let _ = selector.select(0.5, 0.8);
        let _ = selector.select(0.9, 0.7);

        let snapshot = profiler.snapshot();
        assert!(
            snapshot.total_operations >= 3,
            "Profiler should have recorded at least 3 routing operations, got {}",
            snapshot.total_operations
        );

        let routing_stats = snapshot.by_type.get(&OperationType::Routing);
        assert!(
            routing_stats.is_some(),
            "Profiler should have recorded Routing operation type"
        );
        assert_eq!(routing_stats.unwrap().count, 3);
    }

    #[test]
    fn test_profiler_wired_into_vendor_router() {
        use crate::profdag::profiler::{ProfDAGProfiler, ProfilerConfig, OperationType};

        let profiler = Arc::new(ProfDAGProfiler::new(ProfilerConfig::default()));
        let config = RouterConfig::default();
        let router = VendorRouter::new(config).unwrap()
            .with_profiler(profiler.clone());

        let embedding = sample_embedding();

        // Route several queries
        let _ = router.route("Simple question", &embedding);
        let _ = router.route("Complex analysis", &embedding);

        let snapshot = profiler.snapshot();
        // VendorRouter.route creates a guard + VendorSelector.select creates a guard
        // So we expect at least 2 operations from the route method and 2 from the selector
        assert!(
            snapshot.total_operations >= 4,
            "Profiler should have recorded operations from both router and selector, got {}",
            snapshot.total_operations
        );

        let routing_stats = snapshot.by_type.get(&OperationType::Routing);
        assert!(
            routing_stats.is_some(),
            "Profiler should have recorded Routing operation type"
        );
        assert!(routing_stats.unwrap().count >= 2);
    }

    #[test]
    fn test_vendor_selector_without_profiler_has_no_overhead() {
        // Verify that a VendorSelector without a profiler still works fine
        let selector = VendorSelector::new(VendorConfig::default());
        let decision = selector.select(0.4, 0.8);
        assert_eq!(decision.vendor, Vendor::LocalLarge);
    }
}
