//! Model Hooks - Inference Hook System for Open-Weight Models
//!
//! This module provides a hook-based interface for intercepting transformer
//! inference steps and injecting E_nagual biases. Hooks are registered in a
//! [`HookRegistry`] and executed at well-defined points during inference:
//!
//! - **Pre-attention**: Before softmax, allowing modification of QKV and
//!   attention scores.
//! - **Post-attention**: After the attention output, allowing modification
//!   of the layer output.
//! - **Token generated**: After each token is sampled, for monitoring and
//!   metrics collection.
//!
//! # Architecture
//!
//! ```text
//! Inference Loop:
//!   for each layer:
//!     QKV = compute_qkv(hidden_state)
//!     --> HookRegistry::execute_pre_attention(layer, &mut QKV)
//!     attention_output = attention(QKV)
//!     --> HookRegistry::execute_post_attention(layer, &mut output)
//!   logits = lm_head(hidden_state)
//!   token = sample(logits)
//!   --> HookRegistry::execute_on_token(token, &logits)
//! ```
//!
//! # Example
//!
//! ```ignore
//! use nagual::injection::model_hooks::*;
//! use nagual::injection::attention_surgery::*;
//!
//! let surgery = AttentionSurgery::new(AttentionSurgeryConfig::default());
//! let biases = surgery.prepare_biases(&e_nagual, &model_config);
//!
//! let hook = ENagualHook::new(surgery, biases);
//! let mut registry = HookRegistry::new();
//! registry.register(Box::new(hook));
//!
//! // During inference, call at each layer:
//! let mut state = AttentionState::new(queries, keys, values, layer, head);
//! registry.execute_pre_attention(layer, &mut state);
//! ```

use std::time::Instant;

use super::attention_surgery::{AttentionBias, AttentionSurgery};

// ---------------------------------------------------------------------------
// Attention State
// ---------------------------------------------------------------------------

/// Mutable state passed to hooks during the pre-attention phase.
///
/// Contains the Query, Key, and Value tensors for the current layer,
/// along with optional pre-computed attention scores.
#[derive(Debug, Clone)]
pub struct AttentionState {
    /// Query tensor (flattened: head_dim elements per head).
    pub queries: Vec<f32>,

    /// Key tensor (flattened).
    pub keys: Vec<f32>,

    /// Value tensor (flattened).
    pub values: Vec<f32>,

    /// Pre-computed attention scores (num_heads * seq_len), if available.
    /// Hooks may modify these directly instead of recomputing from Q/K.
    pub attention_scores: Option<Vec<f32>>,

    /// Current transformer layer index (0-indexed).
    pub layer: usize,

    /// Current attention head index.
    pub head: usize,

    /// Sequence length for the current forward pass.
    pub seq_len: usize,

    /// Number of attention heads.
    pub num_heads: usize,
}

impl AttentionState {
    /// Create a new attention state.
    pub fn new(
        queries: Vec<f32>,
        keys: Vec<f32>,
        values: Vec<f32>,
        layer: usize,
        head: usize,
    ) -> Self {
        Self {
            queries,
            keys,
            values,
            attention_scores: None,
            layer,
            head,
            seq_len: 0,
            num_heads: 1,
        }
    }

    /// Set the attention scores.
    pub fn with_attention_scores(mut self, scores: Vec<f32>) -> Self {
        self.attention_scores = Some(scores);
        self
    }

    /// Set the sequence length.
    pub fn with_seq_len(mut self, seq_len: usize) -> Self {
        self.seq_len = seq_len;
        self
    }

    /// Set the number of attention heads.
    pub fn with_num_heads(mut self, num_heads: usize) -> Self {
        self.num_heads = num_heads;
        self
    }

    /// Get the attention scores as a 2D array (num_heads x seq_len).
    ///
    /// Returns `None` if attention scores have not been set or if the
    /// dimensions are inconsistent.
    pub fn attention_scores_2d(&self) -> Option<Vec<Vec<f32>>> {
        let scores = self.attention_scores.as_ref()?;
        if self.num_heads == 0 || self.seq_len == 0 {
            return None;
        }

        let expected = self.num_heads * self.seq_len;
        if scores.len() != expected {
            return None;
        }

        let mut result = Vec::with_capacity(self.num_heads);
        for h in 0..self.num_heads {
            let start = h * self.seq_len;
            let end = start + self.seq_len;
            result.push(scores[start..end].to_vec());
        }
        Some(result)
    }

    /// Set the attention scores from a 2D array (num_heads x seq_len).
    pub fn set_attention_scores_2d(&mut self, scores_2d: &[Vec<f32>]) {
        let flat: Vec<f32> = scores_2d.iter().flat_map(|row| row.iter().copied()).collect();
        self.num_heads = scores_2d.len();
        if let Some(first) = scores_2d.first() {
            self.seq_len = first.len();
        }
        self.attention_scores = Some(flat);
    }
}

// ---------------------------------------------------------------------------
// Model Hook Trait
// ---------------------------------------------------------------------------

/// Trait for inference hooks that can intercept and modify model behavior.
///
/// Implement this trait to create custom hooks that modify attention
/// patterns, collect metrics, or inject knowledge during inference.
pub trait ModelHook: Send + Sync {
    /// Called before the attention computation for a given layer.
    ///
    /// The hook may modify the Q/K/V tensors or the pre-computed attention
    /// scores in `state`.
    fn on_pre_attention(
        &self,
        layer: usize,
        state: &mut AttentionState,
    ) -> Result<(), String>;

    /// Called after the attention computation for a given layer.
    ///
    /// The hook may modify the layer output.
    fn on_post_attention(
        &self,
        layer: usize,
        output: &mut Vec<f32>,
    ) -> Result<(), String>;

    /// Called after a token is generated.
    ///
    /// This is a monitoring/metrics hook; it should not modify state.
    fn on_token_generated(&self, token_id: u32, logits: &[f32]);

    /// Return the name of this hook (for registration and debugging).
    fn name(&self) -> &str;
}

// ---------------------------------------------------------------------------
// Hook Metrics
// ---------------------------------------------------------------------------

/// Metrics collected during hook execution.
#[derive(Debug, Clone, Default)]
pub struct HookMetrics {
    /// Total number of times any hook was invoked.
    pub total_invocations: u64,

    /// Cumulative bias magnitude applied across all invocations.
    pub total_bias_applied: f64,

    /// Average hook execution latency in microseconds.
    pub avg_latency_us: f64,

    /// Total number of tokens processed (via `on_token_generated`).
    pub tokens_processed: u64,

    /// Sum of latencies (for computing average).
    latency_sum_us: f64,
}

impl HookMetrics {
    /// Create new empty metrics.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a hook invocation with the given latency and bias magnitude.
    pub fn record_invocation(&mut self, latency_us: f64, bias_magnitude: f64) {
        self.total_invocations += 1;
        self.total_bias_applied += bias_magnitude;
        self.latency_sum_us += latency_us;
        self.avg_latency_us = self.latency_sum_us / self.total_invocations as f64;
    }

    /// Record a token generation event.
    pub fn record_token(&mut self) {
        self.tokens_processed += 1;
    }

    /// Reset all metrics to zero.
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

// ---------------------------------------------------------------------------
// ENagual Hook
// ---------------------------------------------------------------------------

/// A [`ModelHook`] implementation that applies E_nagual attention biases.
///
/// This hook takes precomputed [`AttentionBias`] matrices (from
/// [`AttentionSurgery::prepare_biases`]) and applies them during the
/// pre-attention phase of inference.
pub struct ENagualHook {
    /// The surgery engine used for applying biases.
    surgery: AttentionSurgery,

    /// Pre-computed per-layer biases.
    biases: Vec<AttentionBias>,

    /// Collected metrics.
    metrics: parking_lot::Mutex<HookMetrics>,

    /// Number of tokens generated so far (for warmup tracking).
    token_count: std::sync::atomic::AtomicU64,
}

impl ENagualHook {
    /// Create a new E_nagual hook with the given surgery engine and biases.
    pub fn new(surgery: AttentionSurgery, biases: Vec<AttentionBias>) -> Self {
        Self {
            surgery,
            biases,
            metrics: parking_lot::Mutex::new(HookMetrics::new()),
            token_count: std::sync::atomic::AtomicU64::new(0),
        }
    }

    /// Get a snapshot of the current metrics.
    pub fn metrics(&self) -> HookMetrics {
        self.metrics.lock().clone()
    }

    /// Reset the metrics.
    pub fn reset_metrics(&self) {
        self.metrics.lock().reset();
    }

    /// Get the number of target layers.
    pub fn num_target_layers(&self) -> usize {
        self.biases.len()
    }

    /// Check whether this hook targets the given layer.
    pub fn targets_layer(&self, layer: usize) -> bool {
        self.biases.iter().any(|b| b.target_layer == layer)
    }

    /// Check whether enough tokens have been processed to start applying bias.
    fn past_warmup(&self) -> bool {
        let count = self.token_count.load(std::sync::atomic::Ordering::Relaxed);
        count >= self.surgery.config().warmup_tokens as u64
    }
}

impl ModelHook for ENagualHook {
    fn on_pre_attention(
        &self,
        layer: usize,
        state: &mut AttentionState,
    ) -> Result<(), String> {
        // Skip if still in warmup phase.
        if !self.past_warmup() {
            return Ok(());
        }

        // Check if we have a bias for this layer.
        if !self.targets_layer(layer) {
            return Ok(());
        }

        let start = Instant::now();

        // If attention scores are available, apply bias directly.
        if let Some(mut scores_2d) = state.attention_scores_2d() {
            let applied = self.surgery.apply_to_layer(layer, &mut scores_2d, &self.biases);

            if applied {
                state.set_attention_scores_2d(&scores_2d);

                // Record metrics.
                let elapsed_us = start.elapsed().as_micros() as f64;
                let bias_mag = self
                    .biases
                    .iter()
                    .find(|b| b.target_layer == layer)
                    .map(|b| b.norm() as f64)
                    .unwrap_or(0.0);

                self.metrics.lock().record_invocation(elapsed_us, bias_mag);
            }
        }

        Ok(())
    }

    fn on_post_attention(
        &self,
        _layer: usize,
        _output: &mut Vec<f32>,
    ) -> Result<(), String> {
        // ENagualHook only modifies pre-attention scores.
        Ok(())
    }

    fn on_token_generated(&self, _token_id: u32, _logits: &[f32]) {
        self.token_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.metrics.lock().record_token();
    }

    fn name(&self) -> &str {
        "e_nagual_hook"
    }
}

// ---------------------------------------------------------------------------
// Hook Registry
// ---------------------------------------------------------------------------

/// Registry for managing and executing inference hooks.
///
/// Hooks are executed in registration order. The registry provides
/// convenience methods for running all hooks at each inference phase.
pub struct HookRegistry {
    hooks: Vec<Box<dyn ModelHook>>,
}

impl HookRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self { hooks: Vec::new() }
    }

    /// Register a new hook.
    pub fn register(&mut self, hook: Box<dyn ModelHook>) {
        self.hooks.push(hook);
    }

    /// Unregister a hook by name. Returns `true` if a hook was removed.
    pub fn unregister(&mut self, name: &str) -> bool {
        let len_before = self.hooks.len();
        self.hooks.retain(|h| h.name() != name);
        self.hooks.len() < len_before
    }

    /// Get the list of registered hook names.
    pub fn hooks(&self) -> Vec<&str> {
        self.hooks.iter().map(|h| h.name()).collect()
    }

    /// Get the number of registered hooks.
    pub fn len(&self) -> usize {
        self.hooks.len()
    }

    /// Check if the registry has no hooks.
    pub fn is_empty(&self) -> bool {
        self.hooks.is_empty()
    }

    /// Execute all registered hooks' `on_pre_attention` in order.
    ///
    /// If any hook returns an error, execution stops and the error is returned.
    pub fn execute_pre_attention(
        &self,
        layer: usize,
        state: &mut AttentionState,
    ) -> Result<(), String> {
        for hook in &self.hooks {
            hook.on_pre_attention(layer, state)?;
        }
        Ok(())
    }

    /// Execute all registered hooks' `on_post_attention` in order.
    pub fn execute_post_attention(
        &self,
        layer: usize,
        output: &mut Vec<f32>,
    ) -> Result<(), String> {
        for hook in &self.hooks {
            hook.on_post_attention(layer, output)?;
        }
        Ok(())
    }

    /// Notify all hooks that a token was generated.
    pub fn execute_on_token(&self, token_id: u32, logits: &[f32]) {
        for hook in &self.hooks {
            hook.on_token_generated(token_id, logits);
        }
    }
}

impl Default for HookRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// A simple logging hook for testing/debugging
// ---------------------------------------------------------------------------

/// A minimal hook that logs invocations for debugging purposes.
///
/// This does not modify any state; it only records which layers were seen.
pub struct LoggingHook {
    hook_name: String,
    seen_layers: parking_lot::Mutex<Vec<usize>>,
    seen_tokens: std::sync::atomic::AtomicU64,
}

impl LoggingHook {
    /// Create a new logging hook with the given name.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            hook_name: name.into(),
            seen_layers: parking_lot::Mutex::new(Vec::new()),
            seen_tokens: std::sync::atomic::AtomicU64::new(0),
        }
    }

    /// Get the list of layers that were observed.
    pub fn seen_layers(&self) -> Vec<usize> {
        self.seen_layers.lock().clone()
    }

    /// Get the number of tokens observed.
    pub fn seen_tokens(&self) -> u64 {
        self.seen_tokens.load(std::sync::atomic::Ordering::Relaxed)
    }
}

impl ModelHook for LoggingHook {
    fn on_pre_attention(
        &self,
        layer: usize,
        _state: &mut AttentionState,
    ) -> Result<(), String> {
        self.seen_layers.lock().push(layer);
        Ok(())
    }

    fn on_post_attention(
        &self,
        _layer: usize,
        _output: &mut Vec<f32>,
    ) -> Result<(), String> {
        Ok(())
    }

    fn on_token_generated(&self, _token_id: u32, _logits: &[f32]) {
        self.seen_tokens
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    fn name(&self) -> &str {
        &self.hook_name
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::injection::attention_surgery::{
        AttentionBias, AttentionSurgery, AttentionSurgeryConfig,
    };

    // ---- AttentionState tests ---------------------------------------------

    #[test]
    fn test_attention_state_basic() {
        let state = AttentionState::new(
            vec![1.0, 2.0],
            vec![3.0, 4.0],
            vec![5.0, 6.0],
            5,
            0,
        );

        assert_eq!(state.layer, 5);
        assert_eq!(state.head, 0);
        assert!(state.attention_scores.is_none());
    }

    #[test]
    fn test_attention_state_2d_roundtrip() {
        let scores_2d = vec![vec![0.1, 0.2, 0.3], vec![0.4, 0.5, 0.6]];

        let mut state = AttentionState::new(vec![], vec![], vec![], 0, 0)
            .with_num_heads(2)
            .with_seq_len(3);

        state.set_attention_scores_2d(&scores_2d);

        let recovered = state.attention_scores_2d().unwrap();
        assert_eq!(recovered.len(), 2);
        assert_eq!(recovered[0].len(), 3);
        assert!((recovered[0][0] - 0.1).abs() < 1e-6);
        assert!((recovered[1][2] - 0.6).abs() < 1e-6);
    }

    #[test]
    fn test_attention_state_2d_none_when_unset() {
        let state = AttentionState::new(vec![], vec![], vec![], 0, 0)
            .with_num_heads(2)
            .with_seq_len(3);

        assert!(state.attention_scores_2d().is_none());
    }

    // ---- HookMetrics tests ------------------------------------------------

    #[test]
    fn test_hook_metrics_default() {
        let metrics = HookMetrics::new();
        assert_eq!(metrics.total_invocations, 0);
        assert!((metrics.total_bias_applied - 0.0).abs() < f64::EPSILON);
        assert_eq!(metrics.tokens_processed, 0);
    }

    #[test]
    fn test_hook_metrics_record() {
        let mut metrics = HookMetrics::new();
        metrics.record_invocation(100.0, 0.5);
        metrics.record_invocation(200.0, 1.0);

        assert_eq!(metrics.total_invocations, 2);
        assert!((metrics.total_bias_applied - 1.5).abs() < 1e-10);
        assert!((metrics.avg_latency_us - 150.0).abs() < 1e-10);
    }

    #[test]
    fn test_hook_metrics_record_token() {
        let mut metrics = HookMetrics::new();
        metrics.record_token();
        metrics.record_token();
        metrics.record_token();

        assert_eq!(metrics.tokens_processed, 3);
    }

    #[test]
    fn test_hook_metrics_reset() {
        let mut metrics = HookMetrics::new();
        metrics.record_invocation(100.0, 1.0);
        metrics.record_token();

        metrics.reset();

        assert_eq!(metrics.total_invocations, 0);
        assert_eq!(metrics.tokens_processed, 0);
    }

    // ---- ENagualHook tests ------------------------------------------------

    #[test]
    fn test_e_nagual_hook_name() {
        let config = AttentionSurgeryConfig::default();
        let surgery = AttentionSurgery::new(config);
        let hook = ENagualHook::new(surgery, vec![]);

        assert_eq!(hook.name(), "e_nagual_hook");
    }

    #[test]
    fn test_e_nagual_hook_warmup_skipping() {
        let mut config = AttentionSurgeryConfig::default();
        config.warmup_tokens = 5;
        let surgery = AttentionSurgery::new(config);

        let bias = AttentionBias::new(vec![vec![1.0, 1.0]; 2], 0);
        let hook = ENagualHook::new(surgery, vec![bias]);

        // Create state with attention scores.
        let mut state = AttentionState::new(vec![], vec![], vec![], 0, 0)
            .with_num_heads(2)
            .with_seq_len(2)
            .with_attention_scores(vec![0.0, 0.0, 0.0, 0.0]);

        // Before warmup, scores should not change.
        hook.on_pre_attention(0, &mut state).unwrap();
        let scores_before = state.attention_scores.clone().unwrap();
        assert!(
            scores_before.iter().all(|&v| v.abs() < 1e-6),
            "Scores should be unchanged during warmup"
        );

        // Simulate generating enough tokens to pass warmup.
        for i in 0..5 {
            hook.on_token_generated(i, &[]);
        }

        // After warmup, bias should be applied.
        hook.on_pre_attention(0, &mut state).unwrap();
        let scores_after = state.attention_scores.clone().unwrap();
        assert!(
            scores_after.iter().any(|&v| v.abs() > 1e-6),
            "Scores should be modified after warmup"
        );
    }

    #[test]
    fn test_e_nagual_hook_targets_layer() {
        let config = AttentionSurgeryConfig::default();
        let surgery = AttentionSurgery::new(config);
        let biases = vec![
            AttentionBias::new(vec![vec![0.1]], 10),
            AttentionBias::new(vec![vec![0.1]], 20),
        ];
        let hook = ENagualHook::new(surgery, biases);

        assert!(hook.targets_layer(10));
        assert!(hook.targets_layer(20));
        assert!(!hook.targets_layer(15));
    }

    #[test]
    fn test_e_nagual_hook_metrics_tracking() {
        let mut config = AttentionSurgeryConfig::default();
        config.warmup_tokens = 0; // No warmup for this test.
        let surgery = AttentionSurgery::new(config);
        let bias = AttentionBias::new(vec![vec![0.5; 4]; 2], 0);
        let hook = ENagualHook::new(surgery, vec![bias]);

        let mut state = AttentionState::new(vec![], vec![], vec![], 0, 0)
            .with_num_heads(2)
            .with_seq_len(4)
            .with_attention_scores(vec![0.0; 8]);

        hook.on_pre_attention(0, &mut state).unwrap();
        hook.on_token_generated(42, &[1.0, 2.0]);

        let metrics = hook.metrics();
        assert_eq!(metrics.total_invocations, 1);
        assert!(metrics.total_bias_applied > 0.0);
        assert_eq!(metrics.tokens_processed, 1);
    }

    // ---- HookRegistry tests -----------------------------------------------

    #[test]
    fn test_registry_empty() {
        let registry = HookRegistry::new();
        assert!(registry.is_empty());
        assert_eq!(registry.len(), 0);
        assert!(registry.hooks().is_empty());
    }

    #[test]
    fn test_registry_register_and_list() {
        let mut registry = HookRegistry::new();
        registry.register(Box::new(LoggingHook::new("hook_a")));
        registry.register(Box::new(LoggingHook::new("hook_b")));

        assert_eq!(registry.len(), 2);
        let names = registry.hooks();
        assert_eq!(names, vec!["hook_a", "hook_b"]);
    }

    #[test]
    fn test_registry_unregister() {
        let mut registry = HookRegistry::new();
        registry.register(Box::new(LoggingHook::new("to_remove")));
        registry.register(Box::new(LoggingHook::new("to_keep")));

        assert_eq!(registry.len(), 2);

        let removed = registry.unregister("to_remove");
        assert!(removed);
        assert_eq!(registry.len(), 1);
        assert_eq!(registry.hooks(), vec!["to_keep"]);

        let removed_again = registry.unregister("nonexistent");
        assert!(!removed_again);
    }

    #[test]
    fn test_registry_execute_pre_attention() {
        let mut registry = HookRegistry::new();
        let hook = LoggingHook::new("logger");
        registry.register(Box::new(hook));

        let mut state = AttentionState::new(vec![], vec![], vec![], 5, 0);
        registry.execute_pre_attention(5, &mut state).unwrap();

        // We cannot inspect the LoggingHook directly through the registry,
        // but we can verify it did not error.
    }

    #[test]
    fn test_registry_execute_on_token() {
        let mut registry = HookRegistry::new();
        registry.register(Box::new(LoggingHook::new("token_logger")));

        // Should not panic.
        registry.execute_on_token(42, &[0.1, 0.2, 0.3]);
    }

    // ---- LoggingHook tests ------------------------------------------------

    #[test]
    fn test_logging_hook_records_layers() {
        let hook = LoggingHook::new("test_logger");

        let mut state = AttentionState::new(vec![], vec![], vec![], 0, 0);
        hook.on_pre_attention(3, &mut state).unwrap();
        hook.on_pre_attention(7, &mut state).unwrap();
        hook.on_pre_attention(3, &mut state).unwrap();

        let seen = hook.seen_layers();
        assert_eq!(seen, vec![3, 7, 3]);
    }

    #[test]
    fn test_logging_hook_records_tokens() {
        let hook = LoggingHook::new("test_logger");

        hook.on_token_generated(1, &[]);
        hook.on_token_generated(2, &[]);

        assert_eq!(hook.seen_tokens(), 2);
    }

    #[test]
    fn test_logging_hook_name() {
        let hook = LoggingHook::new("my_hook");
        assert_eq!(hook.name(), "my_hook");
    }

    // ---- Integration: ENagualHook + Registry ------------------------------

    #[test]
    fn test_registry_with_e_nagual_hook() {
        let mut config = AttentionSurgeryConfig::default();
        config.warmup_tokens = 0;
        let surgery = AttentionSurgery::new(config);

        let biases = vec![AttentionBias::new(vec![vec![0.1; 4]; 2], 5)];
        let hook = ENagualHook::new(surgery, biases);

        let mut registry = HookRegistry::new();
        registry.register(Box::new(hook));

        assert_eq!(registry.hooks(), vec!["e_nagual_hook"]);

        // Execute on the target layer.
        let mut state = AttentionState::new(vec![], vec![], vec![], 5, 0)
            .with_num_heads(2)
            .with_seq_len(4)
            .with_attention_scores(vec![0.0; 8]);

        let result = registry.execute_pre_attention(5, &mut state);
        assert!(result.is_ok());

        // Verify bias was applied (scores should be non-zero now).
        let scores = state.attention_scores.unwrap();
        assert!(
            scores.iter().any(|&v| v.abs() > 1e-6),
            "Bias should have been applied to attention scores"
        );
    }
}
