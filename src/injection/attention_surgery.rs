//! Attention Surgery - Direct Attention Weight Modification for Open-Weight Models
//!
//! This module provides direct attention bias injection for open-weight models
//! (GGUF via llama.cpp, candle inference), complementing the prompt-based
//! E_nagual injection used for closed-source vendor models.
//!
//! # Overview
//!
//! While [`ENagual`] works by injecting formatted context into prompts,
//! `AttentionSurgery` modifies the actual attention scores during inference:
//!
//! ```text
//! attention = softmax(Q * K^T / sqrt(d) + E_nagual_bias) * V
//! ```
//!
//! This enables more precise knowledge injection without consuming context
//! window tokens.
//!
//! # Bias Methods
//!
//! Three methods are supported for combining E_nagual bias with attention:
//!
//! - **Additive**: `softmax(QK^T/sqrt(d) + bias) * V` - Direct bias addition
//! - **Multiplicative**: `softmax((QK^T/sqrt(d)) * (1 + bias)) * V` - Scaling
//! - **Gated**: `softmax(QK^T/sqrt(d) + gate * bias) * V` - Learned gate
//!
//! # Example
//!
//! ```ignore
//! use nagual::injection::attention_surgery::*;
//!
//! let config = AttentionSurgeryConfig::builder()
//!     .with_target_layers(vec![28, 29, 30, 31])
//!     .with_bias_scale(0.1)
//!     .with_bias_method(BiasMethod::Additive)
//!     .build();
//!
//! let surgery = AttentionSurgery::new(config);
//! let model_config = ModelConfig::llama_7b();
//!
//! let e_nagual = ENagual::new("query").with_patterns(patterns);
//! let biases = surgery.prepare_biases(&e_nagual, &model_config);
//! let impact = surgery.estimate_impact(&biases);
//!
//! println!("Risk level: {:?}", impact.risk_level);
//! ```

use std::fmt;

use serde::{Deserialize, Serialize};

use super::e_nagual::ENagual;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Method for combining E_nagual bias with raw attention scores.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BiasMethod {
    /// `softmax(QK^T/sqrt(d) + bias) * V` - Direct additive bias.
    Additive,
    /// `softmax((QK^T/sqrt(d)) * (1 + bias)) * V` - Multiplicative scaling.
    Multiplicative,
    /// `softmax(QK^T/sqrt(d) + gate * bias) * V` - Gated with learned weight.
    Gated,
}

impl fmt::Display for BiasMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BiasMethod::Additive => write!(f, "additive"),
            BiasMethod::Multiplicative => write!(f, "multiplicative"),
            BiasMethod::Gated => write!(f, "gated"),
        }
    }
}

/// Configuration for attention surgery operations.
///
/// Controls which transformer layers to modify, how the bias is applied,
/// and safety guardrails to prevent model destabilization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionSurgeryConfig {
    /// Which transformer layers to modify (0-indexed). Default: last 4 layers.
    pub target_layers: Vec<usize>,

    /// Scale factor applied to the E_nagual bias before injection.
    /// Smaller values are safer. Default: 0.1.
    pub bias_scale: f32,

    /// Method for combining bias with attention scores.
    pub bias_method: BiasMethod,

    /// Maximum L2 norm for the bias matrix. Bias is clamped to this norm
    /// to prevent destabilization. Default: 2.0.
    pub max_bias_norm: f32,

    /// Number of tokens to process before bias kicks in.
    /// Allows the model to establish context first. Default: 10.
    pub warmup_tokens: usize,

    /// Exponential decay rate applied per layer. Deeper layers (closer to
    /// output) receive stronger bias. Default: 0.9.
    pub decay_rate: f32,

    /// Gate weight for [`BiasMethod::Gated`]. Ignored for other methods.
    /// Default: 0.5.
    pub gate_weight: f32,
}

impl Default for AttentionSurgeryConfig {
    fn default() -> Self {
        Self {
            target_layers: Vec::new(), // filled at prepare time from ModelConfig
            bias_scale: 0.1,
            bias_method: BiasMethod::Additive,
            max_bias_norm: 2.0,
            warmup_tokens: 10,
            decay_rate: 0.9,
            gate_weight: 0.5,
        }
    }
}

impl AttentionSurgeryConfig {
    /// Return a builder for constructing the configuration.
    pub fn builder() -> AttentionSurgeryConfigBuilder {
        AttentionSurgeryConfigBuilder::new()
    }

    /// Create a conservative configuration with small bias scale.
    pub fn conservative() -> Self {
        Self {
            bias_scale: 0.05,
            max_bias_norm: 1.0,
            warmup_tokens: 20,
            decay_rate: 0.85,
            ..Default::default()
        }
    }

    /// Create an aggressive configuration with larger bias scale.
    pub fn aggressive() -> Self {
        Self {
            bias_scale: 0.25,
            max_bias_norm: 4.0,
            warmup_tokens: 5,
            decay_rate: 0.95,
            ..Default::default()
        }
    }
}

/// Builder for [`AttentionSurgeryConfig`].
pub struct AttentionSurgeryConfigBuilder {
    config: AttentionSurgeryConfig,
}

impl AttentionSurgeryConfigBuilder {
    /// Create a new builder with default values.
    pub fn new() -> Self {
        Self {
            config: AttentionSurgeryConfig::default(),
        }
    }

    /// Set the target layers.
    pub fn with_target_layers(mut self, layers: Vec<usize>) -> Self {
        self.config.target_layers = layers;
        self
    }

    /// Set the bias scale factor.
    pub fn with_bias_scale(mut self, scale: f32) -> Self {
        self.config.bias_scale = scale.clamp(0.0, 1.0);
        self
    }

    /// Set the bias method.
    pub fn with_bias_method(mut self, method: BiasMethod) -> Self {
        self.config.bias_method = method;
        self
    }

    /// Set the maximum bias norm.
    pub fn with_max_bias_norm(mut self, norm: f32) -> Self {
        self.config.max_bias_norm = norm.max(0.01);
        self
    }

    /// Set the warmup tokens.
    pub fn with_warmup_tokens(mut self, tokens: usize) -> Self {
        self.config.warmup_tokens = tokens;
        self
    }

    /// Set the decay rate.
    pub fn with_decay_rate(mut self, rate: f32) -> Self {
        self.config.decay_rate = rate.clamp(0.0, 1.0);
        self
    }

    /// Set the gate weight (only relevant for Gated bias method).
    pub fn with_gate_weight(mut self, weight: f32) -> Self {
        self.config.gate_weight = weight.clamp(0.0, 1.0);
        self
    }

    /// Build the configuration.
    pub fn build(self) -> AttentionSurgeryConfig {
        self.config
    }
}

impl Default for AttentionSurgeryConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Model Configuration
// ---------------------------------------------------------------------------

/// Supported model architecture types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelType {
    /// LLaMA family (LLaMA 2/3, CodeLlama, etc.)
    Llama,
    /// Mistral family (Mistral 7B, Mixtral, etc.)
    Mistral,
    /// Qwen family (Qwen 1.5/2, etc.)
    Qwen,
    /// Phi family (Phi-2, Phi-3, etc.)
    Phi,
    /// Generic transformer architecture.
    Generic,
}

impl fmt::Display for ModelType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ModelType::Llama => write!(f, "llama"),
            ModelType::Mistral => write!(f, "mistral"),
            ModelType::Qwen => write!(f, "qwen"),
            ModelType::Phi => write!(f, "phi"),
            ModelType::Generic => write!(f, "generic"),
        }
    }
}

/// Minimal model configuration describing the transformer architecture.
///
/// This provides enough information to compute correct bias dimensions
/// and target the right layers/heads.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Total number of transformer layers.
    pub num_layers: usize,

    /// Number of attention heads per layer.
    pub num_heads: usize,

    /// Dimension of each attention head (d_model / num_heads).
    pub head_dim: usize,

    /// Model architecture type.
    pub model_type: ModelType,
}

impl ModelConfig {
    /// Create a new model configuration.
    pub fn new(num_layers: usize, num_heads: usize, head_dim: usize, model_type: ModelType) -> Self {
        Self {
            num_layers,
            num_heads,
            head_dim,
            model_type,
        }
    }

    /// Configuration for LLaMA 7B (32 layers, 32 heads, head_dim 128).
    pub fn llama_7b() -> Self {
        Self::new(32, 32, 128, ModelType::Llama)
    }

    /// Configuration for LLaMA 13B (40 layers, 40 heads, head_dim 128).
    pub fn llama_13b() -> Self {
        Self::new(40, 40, 128, ModelType::Llama)
    }

    /// Configuration for Mistral 7B (32 layers, 32 heads, head_dim 128).
    pub fn mistral_7b() -> Self {
        Self::new(32, 32, 128, ModelType::Mistral)
    }

    /// Configuration for Phi-3 Mini (32 layers, 32 heads, head_dim 96).
    pub fn phi_3() -> Self {
        Self::new(32, 32, 96, ModelType::Phi)
    }

    /// Total model dimension (num_heads * head_dim).
    pub fn d_model(&self) -> usize {
        self.num_heads * self.head_dim
    }

    /// Return the default last-N target layers for this model.
    pub fn default_target_layers(&self, n: usize) -> Vec<usize> {
        let n = n.min(self.num_layers);
        ((self.num_layers - n)..self.num_layers).collect()
    }
}

// ---------------------------------------------------------------------------
// Attention Bias
// ---------------------------------------------------------------------------

/// A computed attention bias matrix for a single transformer layer.
///
/// The bias is added to (or combined with) the raw attention scores
/// `QK^T / sqrt(d)` before the softmax operation.
#[derive(Debug, Clone)]
pub struct AttentionBias {
    /// The bias values. Rows are positions in the sequence (or heads
    /// when flattened), columns are the attention targets.
    /// Shape: [num_heads, seq_len] for per-head biases, or
    /// [1, seq_len] for broadcast bias.
    pub bias_matrix: Vec<Vec<f32>>,

    /// Which transformer layer this bias targets.
    pub target_layer: usize,

    /// Which attention heads to modify. `None` means all heads.
    pub head_indices: Option<Vec<usize>>,

    /// The bias method that should be used to apply this bias.
    bias_method: BiasMethod,

    /// Pre-computed L2 norm of the bias matrix.
    norm: f32,
}

impl AttentionBias {
    /// Create a new attention bias for a given layer.
    pub fn new(bias_matrix: Vec<Vec<f32>>, target_layer: usize) -> Self {
        let norm = compute_matrix_norm(&bias_matrix);
        Self {
            bias_matrix,
            target_layer,
            head_indices: None,
            bias_method: BiasMethod::Additive,
            norm,
        }
    }

    /// Set the head indices to target specific attention heads.
    pub fn with_head_indices(mut self, indices: Vec<usize>) -> Self {
        self.head_indices = Some(indices);
        self
    }

    /// Set the bias method.
    pub fn with_bias_method(mut self, method: BiasMethod) -> Self {
        self.bias_method = method;
        self
    }

    /// Compute an attention bias from E_nagual pattern information.
    ///
    /// Converts the pattern relevance scores and confidence values from
    /// E_nagual into a bias matrix suitable for attention injection.
    pub fn from_e_nagual(
        e_nagual: &ENagual,
        config: &AttentionSurgeryConfig,
        target_layer: usize,
        num_heads: usize,
        seq_len: usize,
    ) -> Self {
        // Compute a base bias signal from pattern confidences and similarities.
        // Each pattern contributes a bias proportional to its similarity * confidence.
        let mut base_signal = vec![0.0f32; seq_len];

        let total_patterns = e_nagual.relevant_patterns.len();
        if total_patterns > 0 {
            for (i, scored) in e_nagual.relevant_patterns.iter().enumerate() {
                let pattern = &scored.pattern;
                let weight = scored.similarity * pattern.confidence * config.bias_scale;

                // Distribute the bias across positions using a smooth window.
                // Earlier patterns bias earlier positions (simulating retrieval order).
                let center = if seq_len > 1 {
                    (i as f32 / total_patterns as f32 * (seq_len - 1) as f32) as usize
                } else {
                    0
                };

                let window_half = (seq_len / 8).max(1);
                for pos in 0..seq_len {
                    let dist = (pos as f32 - center as f32).abs();
                    let sigma = window_half as f32;
                    // Gaussian window: exp(-dist^2 / (2 * sigma^2))
                    let contribution = weight * (-dist * dist / (2.0 * sigma * sigma)).exp();
                    base_signal[pos] += contribution;
                }
            }
        }

        // Add trajectory hint signal - boost attention at the end of sequence
        // (where the model decides) based on trajectory confidence.
        if !e_nagual.trajectory_hints.is_empty() {
            let avg_hint_confidence: f32 = e_nagual
                .trajectory_hints
                .iter()
                .map(|h| h.confidence)
                .sum::<f32>()
                / e_nagual.trajectory_hints.len() as f32;

            let hint_bias = avg_hint_confidence * config.bias_scale * 0.5;
            // Boost the last quarter of positions (decision region)
            let decision_start = seq_len * 3 / 4;
            for pos in decision_start..seq_len {
                let ramp = (pos - decision_start) as f32 / (seq_len - decision_start).max(1) as f32;
                base_signal[pos] += hint_bias * ramp;
            }
        }

        // Broadcast the base signal across all target heads.
        let bias_matrix: Vec<Vec<f32>> = (0..num_heads)
            .map(|_| base_signal.clone())
            .collect();

        let mut bias = Self::new(bias_matrix, target_layer);
        bias.bias_method = config.bias_method;

        // Normalize to respect max_bias_norm.
        bias.normalize(config.max_bias_norm);

        bias
    }

    /// Apply additive bias to attention scores.
    ///
    /// Computes: `output = attention_scores + bias`
    ///
    /// This is applied before softmax:
    /// `softmax(QK^T/sqrt(d) + E_nagual_bias) * V`
    pub fn apply_additive(&self, attention_scores: &mut [Vec<f32>]) {
        let num_rows = attention_scores.len().min(self.bias_matrix.len());
        for row in 0..num_rows {
            if self.should_modify_head(row) {
                let score_len = attention_scores[row].len();
                let bias_len = self.bias_matrix[row].len();
                let len = score_len.min(bias_len);
                for col in 0..len {
                    attention_scores[row][col] += self.bias_matrix[row][col];
                }
            }
        }
    }

    /// Apply multiplicative bias to attention scores.
    ///
    /// Computes: `output = attention_scores * (1 + bias)`
    pub fn apply_multiplicative(&self, attention_scores: &mut [Vec<f32>]) {
        let num_rows = attention_scores.len().min(self.bias_matrix.len());
        for row in 0..num_rows {
            if self.should_modify_head(row) {
                let score_len = attention_scores[row].len();
                let bias_len = self.bias_matrix[row].len();
                let len = score_len.min(bias_len);
                for col in 0..len {
                    attention_scores[row][col] *= 1.0 + self.bias_matrix[row][col];
                }
            }
        }
    }

    /// Apply gated bias to attention scores.
    ///
    /// Computes: `output = attention_scores + gate_weight * bias`
    pub fn apply_gated(&self, attention_scores: &mut [Vec<f32>], gate_weight: f32) {
        let num_rows = attention_scores.len().min(self.bias_matrix.len());
        for row in 0..num_rows {
            if self.should_modify_head(row) {
                let score_len = attention_scores[row].len();
                let bias_len = self.bias_matrix[row].len();
                let len = score_len.min(bias_len);
                for col in 0..len {
                    attention_scores[row][col] += gate_weight * self.bias_matrix[row][col];
                }
            }
        }
    }

    /// Normalize the bias matrix so its L2 norm does not exceed `max_norm`.
    ///
    /// This is critical for numerical stability - unbounded bias values
    /// can cause softmax to saturate and produce degenerate attention.
    pub fn normalize(&mut self, max_norm: f32) {
        if self.norm <= f32::EPSILON || self.norm <= max_norm {
            return;
        }

        let scale = max_norm / self.norm;
        for row in &mut self.bias_matrix {
            for val in row.iter_mut() {
                *val *= scale;
            }
        }
        self.norm = max_norm;
    }

    /// Get the current L2 norm of the bias matrix.
    pub fn norm(&self) -> f32 {
        self.norm
    }

    /// Get the number of heads represented in the bias matrix.
    pub fn num_heads(&self) -> usize {
        self.bias_matrix.len()
    }

    /// Get the sequence length of the bias matrix.
    pub fn seq_len(&self) -> usize {
        self.bias_matrix.first().map_or(0, |r| r.len())
    }

    /// Check whether a given head index should be modified.
    fn should_modify_head(&self, head_idx: usize) -> bool {
        match &self.head_indices {
            Some(indices) => indices.contains(&head_idx),
            None => true,
        }
    }
}

// ---------------------------------------------------------------------------
// Impact Estimation
// ---------------------------------------------------------------------------

/// Risk level for a surgery operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum RiskLevel {
    /// Minimal impact, safe for production.
    Low,
    /// Moderate impact, should be tested.
    Medium,
    /// Significant impact, careful evaluation needed.
    High,
    /// Extreme impact, likely to destabilize output.
    Critical,
}

impl fmt::Display for RiskLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RiskLevel::Low => write!(f, "low"),
            RiskLevel::Medium => write!(f, "medium"),
            RiskLevel::High => write!(f, "high"),
            RiskLevel::Critical => write!(f, "critical"),
        }
    }
}

/// Estimated impact of an attention surgery operation.
///
/// This struct provides a safety assessment before applying biases,
/// helping operators decide whether the modification is safe.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SurgeryImpact {
    /// Total L2 norm of all bias matrices combined.
    pub total_bias_norm: f32,

    /// Number of transformer layers affected.
    pub affected_layers: usize,

    /// Number of attention heads affected across all layers.
    pub affected_heads: usize,

    /// Estimated KL divergence from unmodified attention distribution.
    /// This is an approximation based on bias magnitude.
    pub estimated_kl_divergence: f32,

    /// Overall risk level based on the combined metrics.
    pub risk_level: RiskLevel,
}

impl SurgeryImpact {
    /// Create a zero-impact assessment (no surgery applied).
    pub fn zero() -> Self {
        Self {
            total_bias_norm: 0.0,
            affected_layers: 0,
            affected_heads: 0,
            estimated_kl_divergence: 0.0,
            risk_level: RiskLevel::Low,
        }
    }
}

// ---------------------------------------------------------------------------
// Attention Surgery Engine
// ---------------------------------------------------------------------------

/// The core attention surgery engine.
///
/// Computes per-layer bias matrices from E_nagual and applies them
/// to attention scores during open-weight model inference.
pub struct AttentionSurgery {
    /// Configuration controlling surgery behavior.
    config: AttentionSurgeryConfig,
}

impl AttentionSurgery {
    /// Create a new surgery engine with the given configuration.
    pub fn new(config: AttentionSurgeryConfig) -> Self {
        Self { config }
    }

    /// Get a reference to the current configuration.
    pub fn config(&self) -> &AttentionSurgeryConfig {
        &self.config
    }

    /// Prepare per-layer biases from E_nagual for the given model.
    ///
    /// Returns one [`AttentionBias`] per target layer. The biases are
    /// scaled by the configured `bias_scale` and exponentially decayed
    /// so that layers closer to the output receive stronger bias.
    pub fn prepare_biases(
        &self,
        e_nagual: &ENagual,
        model_config: &ModelConfig,
    ) -> Vec<AttentionBias> {
        self.prepare_biases_for_seq_len(e_nagual, model_config, 128)
    }

    /// Prepare biases with an explicit sequence length.
    pub fn prepare_biases_for_seq_len(
        &self,
        e_nagual: &ENagual,
        model_config: &ModelConfig,
        seq_len: usize,
    ) -> Vec<AttentionBias> {
        // Determine target layers: use config if specified, otherwise last 4.
        let target_layers = if self.config.target_layers.is_empty() {
            model_config.default_target_layers(4)
        } else {
            self.config
                .target_layers
                .iter()
                .copied()
                .filter(|&l| l < model_config.num_layers)
                .collect()
        };

        if !e_nagual.has_content() || target_layers.is_empty() {
            return Vec::new();
        }

        let num_target = target_layers.len();

        target_layers
            .iter()
            .enumerate()
            .map(|(rank, &layer_idx)| {
                // Apply exponential decay: deeper layers (higher rank) get stronger bias.
                // rank 0 is the shallowest targeted layer, rank N-1 is deepest.
                let depth_factor = self.config.decay_rate.powi((num_target - 1 - rank) as i32);

                // Create a layer-local config with the depth-adjusted scale.
                let mut layer_config = self.config.clone();
                layer_config.bias_scale *= depth_factor;

                AttentionBias::from_e_nagual(
                    e_nagual,
                    &layer_config,
                    layer_idx,
                    model_config.num_heads,
                    seq_len,
                )
            })
            .collect()
    }

    /// Apply bias to a specific layer's attention scores.
    ///
    /// The `attention_scores` array has shape [num_heads, seq_len].
    /// The correct bias is looked up from the provided biases vector.
    pub fn apply_to_layer(
        &self,
        layer_idx: usize,
        attention_scores: &mut [Vec<f32>],
        biases: &[AttentionBias],
    ) -> bool {
        let bias = match biases.iter().find(|b| b.target_layer == layer_idx) {
            Some(b) => b,
            None => return false,
        };

        match self.config.bias_method {
            BiasMethod::Additive => bias.apply_additive(attention_scores),
            BiasMethod::Multiplicative => bias.apply_multiplicative(attention_scores),
            BiasMethod::Gated => bias.apply_gated(attention_scores, self.config.gate_weight),
        }

        true
    }

    /// Compute pattern-based attention scores for a set of patterns
    /// against a query embedding.
    ///
    /// Returns a vector of attention weights (one per pattern), computed
    /// as softmax over cosine similarities.
    pub fn compute_pattern_attention(
        &self,
        pattern_embeddings: &[Vec<f32>],
        query_embedding: &[f32],
    ) -> Vec<f32> {
        if pattern_embeddings.is_empty() || query_embedding.is_empty() {
            return Vec::new();
        }

        // Compute cosine similarities.
        let similarities: Vec<f32> = pattern_embeddings
            .iter()
            .map(|pe| cosine_similarity(pe, query_embedding))
            .collect();

        // Apply temperature-scaled softmax for attention weights.
        let temperature = 1.0 / (query_embedding.len() as f32).sqrt();
        softmax_with_temperature(&similarities, temperature)
    }

    /// Estimate the impact of applying the given biases.
    ///
    /// This provides a safety check before actually modifying attention,
    /// estimating how much the output distribution will change.
    pub fn estimate_impact(&self, biases: &[AttentionBias]) -> SurgeryImpact {
        if biases.is_empty() {
            return SurgeryImpact::zero();
        }

        let total_bias_norm: f32 = biases.iter().map(|b| b.norm()).sum();
        let affected_layers = biases.len();
        let affected_heads: usize = biases
            .iter()
            .map(|b| match &b.head_indices {
                Some(indices) => indices.len(),
                None => b.num_heads(),
            })
            .sum();

        // Estimate KL divergence from bias magnitude.
        // For small additive biases, KL ~ 0.5 * ||bias||^2 (second-order Taylor).
        let avg_norm = total_bias_norm / affected_layers as f32;
        let estimated_kl_divergence = 0.5 * avg_norm * avg_norm;

        // Determine risk level based on KL divergence and layer coverage.
        let layer_fraction = affected_layers as f32
            / biases
                .iter()
                .map(|b| b.target_layer)
                .max()
                .unwrap_or(1) as f32;

        let risk_score = estimated_kl_divergence * (1.0 + layer_fraction);
        let risk_level = if risk_score < 0.1 {
            RiskLevel::Low
        } else if risk_score < 0.5 {
            RiskLevel::Medium
        } else if risk_score < 1.5 {
            RiskLevel::High
        } else {
            RiskLevel::Critical
        };

        SurgeryImpact {
            total_bias_norm,
            affected_layers,
            affected_heads,
            estimated_kl_divergence,
            risk_level,
        }
    }
}

// ---------------------------------------------------------------------------
// Numerically stable helper functions
// ---------------------------------------------------------------------------

/// Compute the Frobenius (L2) norm of a 2D matrix represented as Vec<Vec<f32>>.
fn compute_matrix_norm(matrix: &[Vec<f32>]) -> f32 {
    let sum_sq: f32 = matrix
        .iter()
        .flat_map(|row| row.iter())
        .map(|&v| v * v)
        .sum();
    sum_sq.sqrt()
}

/// Compute cosine similarity between two vectors.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a < f32::EPSILON || norm_b < f32::EPSILON {
        return 0.0;
    }

    (dot / (norm_a * norm_b)).clamp(-1.0, 1.0)
}

/// Numerically stable softmax with temperature scaling.
///
/// Computes: `softmax(logits / temperature)` using the log-sum-exp trick
/// for numerical stability.
fn softmax_with_temperature(logits: &[f32], temperature: f32) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }

    let temp = temperature.max(f32::EPSILON);
    let scaled: Vec<f32> = logits.iter().map(|&l| l / temp).collect();

    // Log-sum-exp trick: subtract max for numerical stability.
    let max_val = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = scaled.iter().map(|&s| (s - max_val).exp()).collect();
    let sum: f32 = exps.iter().sum();

    if sum < f32::EPSILON {
        // Uniform fallback.
        let uniform = 1.0 / logits.len() as f32;
        return vec![uniform; logits.len()];
    }

    exps.iter().map(|&e| e / sum).collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::injection::e_nagual::ENagual;
    use crate::reasoning_bank::{FactorScores, Pattern, ScoredPattern};

    // ---- helpers ---------------------------------------------------------

    fn create_test_e_nagual() -> ENagual {
        let pattern = Pattern::new(
            "How to optimize queries?",
            "Use prepared statements and indexes",
            "database.performance",
        )
        .with_context("PostgreSQL")
        .with_confidence(0.9)
        .with_reward(0.85);

        let scored = ScoredPattern {
            pattern,
            similarity: 0.92,
            final_score: 0.88,
            factor_scores: FactorScores::default(),
        };

        ENagual::new("optimize database queries").with_patterns(vec![scored])
    }

    fn create_multi_pattern_e_nagual() -> ENagual {
        let patterns: Vec<ScoredPattern> = (0..3)
            .map(|i| {
                let pattern = Pattern::new(
                    format!("Problem {}", i),
                    format!("Solution {}", i),
                    "test.domain",
                )
                .with_confidence(0.7 + i as f32 * 0.1)
                .with_reward(0.8);

                ScoredPattern {
                    pattern,
                    similarity: 0.8 + i as f32 * 0.05,
                    final_score: 0.85,
                    factor_scores: FactorScores::default(),
                }
            })
            .collect();

        ENagual::new("multi-pattern query").with_patterns(patterns)
    }

    fn make_zero_scores(num_heads: usize, seq_len: usize) -> Vec<Vec<f32>> {
        vec![vec![0.0; seq_len]; num_heads]
    }

    fn make_uniform_scores(num_heads: usize, seq_len: usize, val: f32) -> Vec<Vec<f32>> {
        vec![vec![val; seq_len]; num_heads]
    }

    // ---- config tests ----------------------------------------------------

    #[test]
    fn test_config_builder_defaults() {
        let config = AttentionSurgeryConfig::builder().build();
        assert_eq!(config.bias_scale, 0.1);
        assert_eq!(config.bias_method, BiasMethod::Additive);
        assert_eq!(config.max_bias_norm, 2.0);
        assert_eq!(config.warmup_tokens, 10);
        assert!((config.decay_rate - 0.9).abs() < f32::EPSILON);
    }

    #[test]
    fn test_config_builder_custom() {
        let config = AttentionSurgeryConfig::builder()
            .with_target_layers(vec![28, 29, 30, 31])
            .with_bias_scale(0.2)
            .with_bias_method(BiasMethod::Gated)
            .with_max_bias_norm(3.0)
            .with_warmup_tokens(20)
            .with_decay_rate(0.95)
            .with_gate_weight(0.7)
            .build();

        assert_eq!(config.target_layers, vec![28, 29, 30, 31]);
        assert!((config.bias_scale - 0.2).abs() < f32::EPSILON);
        assert_eq!(config.bias_method, BiasMethod::Gated);
        assert!((config.max_bias_norm - 3.0).abs() < f32::EPSILON);
        assert_eq!(config.warmup_tokens, 20);
        assert!((config.decay_rate - 0.95).abs() < f32::EPSILON);
        assert!((config.gate_weight - 0.7).abs() < f32::EPSILON);
    }

    #[test]
    fn test_config_conservative_and_aggressive() {
        let conservative = AttentionSurgeryConfig::conservative();
        let aggressive = AttentionSurgeryConfig::aggressive();

        assert!(conservative.bias_scale < aggressive.bias_scale);
        assert!(conservative.max_bias_norm < aggressive.max_bias_norm);
        assert!(conservative.warmup_tokens > aggressive.warmup_tokens);
    }

    // ---- model config tests -----------------------------------------------

    #[test]
    fn test_model_config_factory_methods() {
        let llama7b = ModelConfig::llama_7b();
        assert_eq!(llama7b.num_layers, 32);
        assert_eq!(llama7b.num_heads, 32);
        assert_eq!(llama7b.head_dim, 128);
        assert_eq!(llama7b.d_model(), 4096);

        let llama13b = ModelConfig::llama_13b();
        assert_eq!(llama13b.num_layers, 40);
        assert_eq!(llama13b.d_model(), 5120);

        let mistral = ModelConfig::mistral_7b();
        assert_eq!(mistral.model_type, ModelType::Mistral);

        let phi = ModelConfig::phi_3();
        assert_eq!(phi.head_dim, 96);
        assert_eq!(phi.d_model(), 3072);
    }

    #[test]
    fn test_default_target_layers() {
        let model = ModelConfig::llama_7b();
        let layers = model.default_target_layers(4);
        assert_eq!(layers, vec![28, 29, 30, 31]);

        let layers_all = model.default_target_layers(100);
        assert_eq!(layers_all.len(), 32);
    }

    // ---- bias computation tests -------------------------------------------

    #[test]
    fn test_bias_from_e_nagual_produces_correct_shape() {
        let e_nagual = create_test_e_nagual();
        let config = AttentionSurgeryConfig::default();
        let num_heads = 32;
        let seq_len = 64;

        let bias = AttentionBias::from_e_nagual(&e_nagual, &config, 30, num_heads, seq_len);

        assert_eq!(bias.target_layer, 30);
        assert_eq!(bias.bias_matrix.len(), num_heads);
        assert_eq!(bias.bias_matrix[0].len(), seq_len);
    }

    #[test]
    fn test_bias_from_e_nagual_nonzero_values() {
        let e_nagual = create_test_e_nagual();
        let config = AttentionSurgeryConfig::default();

        let bias = AttentionBias::from_e_nagual(&e_nagual, &config, 0, 4, 16);

        // Bias should be non-zero since we have patterns.
        let has_nonzero = bias
            .bias_matrix
            .iter()
            .flat_map(|r| r.iter())
            .any(|&v| v.abs() > f32::EPSILON);
        assert!(has_nonzero, "Bias matrix should have non-zero values");
    }

    #[test]
    fn test_bias_from_empty_e_nagual() {
        let e_nagual = ENagual::new("empty");
        let config = AttentionSurgeryConfig::default();

        let bias = AttentionBias::from_e_nagual(&e_nagual, &config, 0, 4, 16);

        // All values should be zero for an empty E_nagual.
        let all_zero = bias
            .bias_matrix
            .iter()
            .flat_map(|r| r.iter())
            .all(|&v| v.abs() < f32::EPSILON);
        assert!(all_zero, "Empty E_nagual should produce zero bias");
    }

    #[test]
    fn test_bias_multi_pattern() {
        let e_nagual = create_multi_pattern_e_nagual();
        let config = AttentionSurgeryConfig::default();

        let bias = AttentionBias::from_e_nagual(&e_nagual, &config, 0, 4, 32);

        // With 3 patterns, the bias should have a richer distribution.
        let sum: f32 = bias.bias_matrix[0].iter().sum();
        assert!(sum > 0.0, "Multi-pattern bias should have positive sum");
    }

    // ---- application tests ------------------------------------------------

    #[test]
    fn test_apply_additive() {
        let bias_values = vec![vec![0.1, 0.2, 0.3], vec![0.4, 0.5, 0.6]];
        let bias = AttentionBias::new(bias_values, 0);

        let mut scores = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        bias.apply_additive(&mut scores);

        assert!((scores[0][0] - 1.1).abs() < 1e-5);
        assert!((scores[0][1] - 2.2).abs() < 1e-5);
        assert!((scores[0][2] - 3.3).abs() < 1e-5);
        assert!((scores[1][0] - 4.4).abs() < 1e-5);
        assert!((scores[1][1] - 5.5).abs() < 1e-5);
        assert!((scores[1][2] - 6.6).abs() < 1e-5);
    }

    #[test]
    fn test_apply_multiplicative() {
        let bias_values = vec![vec![0.1, 0.2]];
        let bias = AttentionBias::new(bias_values, 0);

        let mut scores = vec![vec![2.0, 3.0]];
        bias.apply_multiplicative(&mut scores);

        // 2.0 * (1 + 0.1) = 2.2, 3.0 * (1 + 0.2) = 3.6
        assert!((scores[0][0] - 2.2).abs() < 1e-5);
        assert!((scores[0][1] - 3.6).abs() < 1e-5);
    }

    #[test]
    fn test_apply_gated() {
        let bias_values = vec![vec![1.0, 2.0]];
        let bias = AttentionBias::new(bias_values, 0);

        let mut scores = vec![vec![0.0, 0.0]];
        bias.apply_gated(&mut scores, 0.5);

        // 0.0 + 0.5 * 1.0 = 0.5, 0.0 + 0.5 * 2.0 = 1.0
        assert!((scores[0][0] - 0.5).abs() < 1e-5);
        assert!((scores[0][1] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_apply_with_head_indices() {
        let bias_values = vec![vec![1.0], vec![1.0], vec![1.0], vec![1.0]];
        let bias = AttentionBias::new(bias_values, 0).with_head_indices(vec![0, 2]);

        let mut scores = vec![vec![0.0], vec![0.0], vec![0.0], vec![0.0]];
        bias.apply_additive(&mut scores);

        // Only heads 0 and 2 should be modified.
        assert!((scores[0][0] - 1.0).abs() < 1e-5);
        assert!((scores[1][0] - 0.0).abs() < 1e-5); // untouched
        assert!((scores[2][0] - 1.0).abs() < 1e-5);
        assert!((scores[3][0] - 0.0).abs() < 1e-5); // untouched
    }

    // ---- normalization tests ----------------------------------------------

    #[test]
    fn test_normalization_clamps_large_bias() {
        let large_values = vec![vec![10.0, 20.0, 30.0]];
        let mut bias = AttentionBias::new(large_values, 0);
        let original_norm = bias.norm();

        assert!(original_norm > 2.0, "Norm should start large");

        bias.normalize(2.0);

        assert!(
            (bias.norm() - 2.0).abs() < 0.01,
            "Norm should be clamped to 2.0, got {}",
            bias.norm()
        );
    }

    #[test]
    fn test_normalization_preserves_small_bias() {
        let small_values = vec![vec![0.01, 0.02]];
        let mut bias = AttentionBias::new(small_values.clone(), 0);

        bias.normalize(2.0);

        // Should be unchanged.
        assert!((bias.bias_matrix[0][0] - 0.01).abs() < 1e-6);
        assert!((bias.bias_matrix[0][1] - 0.02).abs() < 1e-6);
    }

    // ---- surgery engine tests ---------------------------------------------

    #[test]
    fn test_prepare_biases_default_layers() {
        let e_nagual = create_test_e_nagual();
        let config = AttentionSurgeryConfig::default();
        let surgery = AttentionSurgery::new(config);
        let model = ModelConfig::llama_7b();

        let biases = surgery.prepare_biases(&e_nagual, &model);

        // Default: last 4 layers of 32-layer model = layers 28, 29, 30, 31.
        assert_eq!(biases.len(), 4);
        assert_eq!(biases[0].target_layer, 28);
        assert_eq!(biases[1].target_layer, 29);
        assert_eq!(biases[2].target_layer, 30);
        assert_eq!(biases[3].target_layer, 31);
    }

    #[test]
    fn test_prepare_biases_custom_layers() {
        let e_nagual = create_test_e_nagual();
        let config = AttentionSurgeryConfig::builder()
            .with_target_layers(vec![10, 20])
            .build();
        let surgery = AttentionSurgery::new(config);
        let model = ModelConfig::llama_7b();

        let biases = surgery.prepare_biases(&e_nagual, &model);

        assert_eq!(biases.len(), 2);
        assert_eq!(biases[0].target_layer, 10);
        assert_eq!(biases[1].target_layer, 20);
    }

    #[test]
    fn test_prepare_biases_empty_e_nagual() {
        let e_nagual = ENagual::new("empty");
        let config = AttentionSurgeryConfig::default();
        let surgery = AttentionSurgery::new(config);
        let model = ModelConfig::llama_7b();

        let biases = surgery.prepare_biases(&e_nagual, &model);
        assert!(biases.is_empty(), "Empty E_nagual should produce no biases");
    }

    #[test]
    fn test_prepare_biases_decay_ordering() {
        let e_nagual = create_test_e_nagual();
        let config = AttentionSurgeryConfig::builder()
            .with_decay_rate(0.5) // strong decay to make difference visible
            .build();
        let surgery = AttentionSurgery::new(config);
        let model = ModelConfig::llama_7b();

        let biases = surgery.prepare_biases_for_seq_len(&e_nagual, &model, 32);

        // Deeper layers (higher index) should have stronger bias (higher norm)
        // due to decay: shallowest layer gets decay^(N-1), deepest gets decay^0 = 1.
        assert!(biases.len() >= 2);
        let first_norm = biases.first().unwrap().norm();
        let last_norm = biases.last().unwrap().norm();
        assert!(
            last_norm >= first_norm,
            "Deepest layer should have stronger bias: {} >= {}",
            last_norm,
            first_norm
        );
    }

    #[test]
    fn test_apply_to_layer() {
        let e_nagual = create_test_e_nagual();
        let config = AttentionSurgeryConfig::default();
        let surgery = AttentionSurgery::new(config);
        let model = ModelConfig::llama_7b();

        let biases = surgery.prepare_biases_for_seq_len(&e_nagual, &model, 16);

        // Layer 28 should be in the biases.
        let mut scores = make_zero_scores(32, 16);
        let applied = surgery.apply_to_layer(28, &mut scores, &biases);
        assert!(applied, "Layer 28 should be a target");

        // Layer 0 should NOT be in the biases.
        let mut scores2 = make_zero_scores(32, 16);
        let applied2 = surgery.apply_to_layer(0, &mut scores2, &biases);
        assert!(!applied2, "Layer 0 should not be a target");
    }

    // ---- pattern attention tests ------------------------------------------

    #[test]
    fn test_compute_pattern_attention() {
        let surgery = AttentionSurgery::new(AttentionSurgeryConfig::default());

        let patterns = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let query = vec![1.0, 0.0, 0.0];

        let attention = surgery.compute_pattern_attention(&patterns, &query);

        assert_eq!(attention.len(), 3);
        // First pattern is identical to query, should have highest attention.
        assert!(
            attention[0] > attention[1],
            "Most similar pattern should have highest attention"
        );
        assert!(
            attention[0] > attention[2],
            "Most similar pattern should have highest attention"
        );

        // Sum should be ~1.0 (softmax property).
        let sum: f32 = attention.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "Attention should sum to 1.0");
    }

    #[test]
    fn test_compute_pattern_attention_empty() {
        let surgery = AttentionSurgery::new(AttentionSurgeryConfig::default());

        assert!(surgery.compute_pattern_attention(&[], &[1.0, 0.0]).is_empty());
        assert!(surgery.compute_pattern_attention(&[vec![1.0]], &[]).is_empty());
    }

    // ---- impact estimation tests ------------------------------------------

    #[test]
    fn test_estimate_impact_zero() {
        let surgery = AttentionSurgery::new(AttentionSurgeryConfig::default());
        let impact = surgery.estimate_impact(&[]);
        assert_eq!(impact.risk_level, RiskLevel::Low);
        assert_eq!(impact.affected_layers, 0);
        assert!((impact.total_bias_norm - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_estimate_impact_small_bias() {
        let surgery = AttentionSurgery::new(AttentionSurgeryConfig::default());
        let biases = vec![AttentionBias::new(vec![vec![0.01, 0.02]], 31)];

        let impact = surgery.estimate_impact(&biases);

        assert_eq!(impact.affected_layers, 1);
        assert_eq!(impact.affected_heads, 1);
        assert_eq!(impact.risk_level, RiskLevel::Low);
        assert!(impact.estimated_kl_divergence < 0.1);
    }

    #[test]
    fn test_estimate_impact_large_bias() {
        let surgery = AttentionSurgery::new(AttentionSurgeryConfig::default());
        // Create a bias with very large values.
        let biases = vec![AttentionBias::new(vec![vec![10.0; 128]; 32], 31)];

        let impact = surgery.estimate_impact(&biases);

        assert!(impact.risk_level >= RiskLevel::High);
        assert!(impact.estimated_kl_divergence > 1.0);
    }

    #[test]
    fn test_risk_level_ordering() {
        assert!(RiskLevel::Low < RiskLevel::Medium);
        assert!(RiskLevel::Medium < RiskLevel::High);
        assert!(RiskLevel::High < RiskLevel::Critical);
    }

    // ---- helper function tests --------------------------------------------

    #[test]
    fn test_cosine_similarity_identical() {
        let a = vec![1.0, 0.0, 0.0];
        let sim = cosine_similarity(&a, &a);
        assert!((sim - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-5);
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim - (-1.0)).abs() < 1e-5);
    }

    #[test]
    fn test_softmax_with_temperature_sums_to_one() {
        let logits = vec![1.0, 2.0, 3.0, 4.0];
        let probs = softmax_with_temperature(&logits, 1.0);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_softmax_with_temperature_ordering() {
        let logits = vec![1.0, 3.0, 2.0];
        let probs = softmax_with_temperature(&logits, 1.0);
        assert!(probs[1] > probs[2]);
        assert!(probs[2] > probs[0]);
    }

    #[test]
    fn test_softmax_numerical_stability() {
        // Large values that would overflow naive exp().
        let logits = vec![1000.0, 1001.0, 1002.0];
        let probs = softmax_with_temperature(&logits, 1.0);
        let sum: f32 = probs.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "Softmax should be stable with large inputs"
        );
        assert!(probs.iter().all(|&p| p.is_finite()));
    }

    #[test]
    fn test_compute_matrix_norm() {
        let matrix = vec![vec![3.0, 4.0]]; // norm = 5
        let norm = compute_matrix_norm(&matrix);
        assert!((norm - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_bias_method_display() {
        assert_eq!(format!("{}", BiasMethod::Additive), "additive");
        assert_eq!(format!("{}", BiasMethod::Multiplicative), "multiplicative");
        assert_eq!(format!("{}", BiasMethod::Gated), "gated");
    }

    #[test]
    fn test_model_type_display() {
        assert_eq!(format!("{}", ModelType::Llama), "llama");
        assert_eq!(format!("{}", ModelType::Mistral), "mistral");
        assert_eq!(format!("{}", ModelType::Qwen), "qwen");
        assert_eq!(format!("{}", ModelType::Phi), "phi");
        assert_eq!(format!("{}", ModelType::Generic), "generic");
    }
}
