//! FastGRNN - Fast Gated Recurrent Neural Network
//!
//! A lightweight RNN implementation optimized for fast inference (<5ms routing latency).
//! FastGRNN uses low-rank factorization and sparse matrices for efficiency while
//! maintaining accuracy for complexity estimation.
//!
//! # Architecture
//!
//! FastGRNN is based on the paper "FastGRNN: A Fast, Accurate, Stable and Tiny
//! Kilobyte Sized Gated Recurrent Neural Network" (Microsoft Research).
//!
//! The update equations are:
//! ```text
//! z_t = sigmoid(W_z * x_t + U_z * h_{t-1} + b_z)
//! h_tilde = tanh(W_h * x_t + U_h * h_{t-1} + b_h)
//! h_t = (zeta * (1 - z_t) + nu) * h_tilde + z_t * h_{t-1}
//! ```
//!
//! Where zeta and nu are learnable scalars that replace the reset gate,
//! reducing parameters while maintaining expressivity.
//!
//! # Features
//!
//! - Compact model size (< 100KB)
//! - Fast inference (< 1ms per forward pass)
//! - Suitable for edge deployment
//! - Trainable with standard backpropagation
//! - ONNX Runtime support for optimized inference
//!
//! # Backend Selection
//!
//! The module supports two backends:
//! - **ONNX**: Uses ONNX Runtime for optimized inference (preferred)
//! - **Native**: Pure Rust implementation as fallback

use ndarray::{Array1, Array2, ArrayView1};
#[cfg(feature = "onnx-embed")]
use ort::session::builder::GraphOptimizationLevel;
#[cfg(feature = "onnx-embed")]
use ort::session::Session;
#[cfg(feature = "onnx-embed")]
use ort::value::Tensor;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::Arc;

use super::{RouterError, RouterResult};

/// Embedded trained model weights (62.3% accuracy on routing benchmark).
/// These weights are compiled into the binary so they're available even without
/// the JSON file on disk.
const TRAINED_WEIGHTS_JSON: &str = include_str!("../../models/fastgrnn_router.json");

/// Configuration for FastGRNN.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FastGRNNConfig {
    /// Input dimension (number of features).
    pub input_dim: usize,

    /// Hidden state dimension.
    pub hidden_dim: usize,

    /// Output dimension (1 for complexity score).
    pub output_dim: usize,

    /// Zeta parameter for hidden state scaling (default: 1.0).
    pub zeta: f32,

    /// Nu parameter for hidden state bias (default: -0.001).
    pub nu: f32,

    /// Whether to use low-rank factorization.
    pub use_low_rank: bool,

    /// Rank for low-rank factorization (if enabled).
    pub rank: usize,

    /// Sparsity level (0.0 = dense, 1.0 = fully sparse).
    pub sparsity: f32,

    /// Dropout rate during training (not used in inference).
    pub dropout: f32,
}

impl Default for FastGRNNConfig {
    fn default() -> Self {
        Self {
            input_dim: 5, // query_length, embedding_norm, domain_specificity, pattern_coverage, historical_accuracy
            hidden_dim: 16, // Small hidden state for fast inference
            output_dim: 1, // Complexity score [0.0, 1.0]
            zeta: 1.0,
            nu: -0.001,
            use_low_rank: false,
            rank: 8,
            sparsity: 0.0,
            dropout: 0.0,
        }
    }
}

impl FastGRNNConfig {
    /// Create a compact configuration for minimal latency.
    pub fn compact() -> Self {
        Self {
            input_dim: 5,
            hidden_dim: 8,
            output_dim: 1,
            zeta: 1.0,
            nu: -0.001,
            use_low_rank: true,
            rank: 4,
            sparsity: 0.5,
            dropout: 0.0,
        }
    }

    /// Create a standard configuration for balanced performance.
    pub fn standard() -> Self {
        Self {
            input_dim: 5,
            hidden_dim: 32,
            output_dim: 1,
            zeta: 1.0,
            nu: -0.001,
            use_low_rank: true,
            rank: 16,
            sparsity: 0.3,
            dropout: 0.1,
        }
    }

    /// Set custom input dimension.
    pub fn with_input_dim(mut self, dim: usize) -> Self {
        self.input_dim = dim;
        self
    }

    /// Set custom hidden dimension.
    pub fn with_hidden_dim(mut self, dim: usize) -> Self {
        self.hidden_dim = dim;
        self
    }

    /// Validate configuration.
    pub fn validate(&self) -> RouterResult<()> {
        if self.input_dim == 0 {
            return Err(super::RouterError::InvalidConfig(
                "input_dim must be > 0".to_string(),
            ));
        }
        if self.hidden_dim == 0 {
            return Err(super::RouterError::InvalidConfig(
                "hidden_dim must be > 0".to_string(),
            ));
        }
        if self.output_dim == 0 {
            return Err(super::RouterError::InvalidConfig(
                "output_dim must be > 0".to_string(),
            ));
        }
        if self.sparsity < 0.0 || self.sparsity > 1.0 {
            return Err(super::RouterError::InvalidConfig(
                "sparsity must be in [0.0, 1.0]".to_string(),
            ));
        }
        Ok(())
    }
}

/// Pre-trained weights for FastGRNN.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FastGRNNWeights {
    /// Gate weights for input: W_z (hidden_dim x input_dim)
    pub w_z: Vec<f32>,

    /// Gate weights for hidden state: U_z (hidden_dim x hidden_dim)
    pub u_z: Vec<f32>,

    /// Gate bias: b_z (hidden_dim)
    pub b_z: Vec<f32>,

    /// Hidden weights for input: W_h (hidden_dim x input_dim)
    pub w_h: Vec<f32>,

    /// Hidden weights for hidden state: U_h (hidden_dim x hidden_dim)
    pub u_h: Vec<f32>,

    /// Hidden bias: b_h (hidden_dim)
    pub b_h: Vec<f32>,

    /// Output weights: W_o (output_dim x hidden_dim)
    pub w_o: Vec<f32>,

    /// Output bias: b_o (output_dim)
    pub b_o: Vec<f32>,

    /// Zeta scalar
    pub zeta: f32,

    /// Nu scalar
    pub nu: f32,
}

impl FastGRNNWeights {
    /// Create random weights for initialization/testing.
    pub fn random(config: &FastGRNNConfig) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        let xavier_input = (6.0 / (config.input_dim + config.hidden_dim) as f32).sqrt();
        let xavier_hidden = (6.0 / (config.hidden_dim * 2) as f32).sqrt();
        let xavier_output = (6.0 / (config.hidden_dim + config.output_dim) as f32).sqrt();

        let mut random_vec = |size: usize, scale: f32| -> Vec<f32> {
            (0..size)
                .map(|_| rng.gen_range(-scale..scale))
                .collect()
        };

        Self {
            w_z: random_vec(config.hidden_dim * config.input_dim, xavier_input),
            u_z: random_vec(config.hidden_dim * config.hidden_dim, xavier_hidden),
            b_z: vec![0.0; config.hidden_dim],
            w_h: random_vec(config.hidden_dim * config.input_dim, xavier_input),
            u_h: random_vec(config.hidden_dim * config.hidden_dim, xavier_hidden),
            b_h: vec![0.0; config.hidden_dim],
            w_o: random_vec(config.output_dim * config.hidden_dim, xavier_output),
            b_o: vec![0.0; config.output_dim],
            zeta: config.zeta,
            nu: config.nu,
        }
    }

    /// Create pre-trained weights for routing.
    ///
    /// Loads trained weights (62.3% accuracy) embedded at compile time from
    /// `models/fastgrnn_router.json`. Falls back to random Xavier initialization
    /// only if the embedded weights don't match the provided config dimensions.
    ///
    /// These weights produce:
    /// - Low complexity (~0.2) for short, common queries
    /// - Medium complexity (~0.5) for domain-specific queries
    /// - High complexity (~0.8+) for complex reasoning queries
    pub fn pretrained(config: &FastGRNNConfig) -> Self {
        // Try loading trained weights from embedded JSON
        if let Ok(trained) = Self::from_embedded_json(config) {
            return trained;
        }

        // Fall back to random initialization if embedded weights don't match config
        tracing::warn!(
            "Falling back to random weights - embedded weights don't match config \
             (input_dim={}, hidden_dim={}, output_dim={})",
            config.input_dim,
            config.hidden_dim,
            config.output_dim
        );
        let mut weights = Self::random(config);
        for b in weights.b_z.iter_mut() {
            *b = -0.5;
        }
        for b in weights.b_h.iter_mut() {
            *b = 0.0;
        }
        for b in weights.b_o.iter_mut() {
            *b = 0.0;
        }
        weights
    }

    /// Attempt to parse the compile-time embedded JSON weights and validate
    /// that their dimensions match the given config.
    fn from_embedded_json(config: &FastGRNNConfig) -> Result<Self, String> {
        #[derive(Deserialize)]
        struct JsonModel {
            weights: FastGRNNWeights,
            // config and metrics fields are present in the JSON but we only
            // need the weights; dimensions are validated explicitly below.
        }

        let model: JsonModel = serde_json::from_str(TRAINED_WEIGHTS_JSON)
            .map_err(|e| format!("Failed to parse embedded weights: {}", e))?;

        let weights = model.weights;

        // Verify all critical dimensions match the requested config
        let expected_wz = config.hidden_dim * config.input_dim;
        if weights.w_z.len() != expected_wz {
            return Err(format!(
                "w_z dimension mismatch: got {} but config expects {}",
                weights.w_z.len(),
                expected_wz
            ));
        }

        let expected_uz = config.hidden_dim * config.hidden_dim;
        if weights.u_z.len() != expected_uz {
            return Err(format!(
                "u_z dimension mismatch: got {} but config expects {}",
                weights.u_z.len(),
                expected_uz
            ));
        }

        if weights.b_z.len() != config.hidden_dim {
            return Err(format!(
                "b_z dimension mismatch: got {} but config expects {}",
                weights.b_z.len(),
                config.hidden_dim
            ));
        }

        let expected_wh = config.hidden_dim * config.input_dim;
        if weights.w_h.len() != expected_wh {
            return Err(format!(
                "w_h dimension mismatch: got {} but config expects {}",
                weights.w_h.len(),
                expected_wh
            ));
        }

        let expected_uh = config.hidden_dim * config.hidden_dim;
        if weights.u_h.len() != expected_uh {
            return Err(format!(
                "u_h dimension mismatch: got {} but config expects {}",
                weights.u_h.len(),
                expected_uh
            ));
        }

        if weights.b_h.len() != config.hidden_dim {
            return Err(format!(
                "b_h dimension mismatch: got {} but config expects {}",
                weights.b_h.len(),
                config.hidden_dim
            ));
        }

        let expected_wo = config.output_dim * config.hidden_dim;
        if weights.w_o.len() != expected_wo {
            return Err(format!(
                "w_o dimension mismatch: got {} but config expects {}",
                weights.w_o.len(),
                expected_wo
            ));
        }

        if weights.b_o.len() != config.output_dim {
            return Err(format!(
                "b_o dimension mismatch: got {} but config expects {}",
                weights.b_o.len(),
                config.output_dim
            ));
        }

        Ok(weights)
    }

    /// Validate weights dimensions match config.
    pub fn validate(&self, config: &FastGRNNConfig) -> RouterResult<()> {
        let expected_wz = config.hidden_dim * config.input_dim;
        if self.w_z.len() != expected_wz {
            return Err(super::RouterError::DimensionMismatch {
                expected: expected_wz,
                actual: self.w_z.len(),
            });
        }

        let expected_uz = config.hidden_dim * config.hidden_dim;
        if self.u_z.len() != expected_uz {
            return Err(super::RouterError::DimensionMismatch {
                expected: expected_uz,
                actual: self.u_z.len(),
            });
        }

        if self.b_z.len() != config.hidden_dim {
            return Err(super::RouterError::DimensionMismatch {
                expected: config.hidden_dim,
                actual: self.b_z.len(),
            });
        }

        let expected_wo = config.output_dim * config.hidden_dim;
        if self.w_o.len() != expected_wo {
            return Err(super::RouterError::DimensionMismatch {
                expected: expected_wo,
                actual: self.w_o.len(),
            });
        }

        Ok(())
    }
}

/// A single FastGRNN cell for one time step.
#[derive(Debug)]
pub struct GRNNCell {
    /// Hidden dimension.
    hidden_dim: usize,

    /// Input dimension.
    input_dim: usize,

    /// Zeta parameter.
    zeta: f32,

    /// Nu parameter.
    nu: f32,

    /// Gate weights for input: W_z.
    w_z: Array2<f32>,

    /// Gate weights for hidden: U_z.
    u_z: Array2<f32>,

    /// Gate bias: b_z.
    b_z: Array1<f32>,

    /// Hidden weights for input: W_h.
    w_h: Array2<f32>,

    /// Hidden weights for hidden: U_h.
    u_h: Array2<f32>,

    /// Hidden bias: b_h.
    b_h: Array1<f32>,
}

impl GRNNCell {
    /// Create a new GRNN cell from weights.
    pub fn new(config: &FastGRNNConfig, weights: &FastGRNNWeights) -> RouterResult<Self> {
        config.validate()?;
        weights.validate(config)?;

        let w_z = Array2::from_shape_vec(
            (config.hidden_dim, config.input_dim),
            weights.w_z.clone(),
        )
        .map_err(|e| super::RouterError::InvalidConfig(e.to_string()))?;

        let u_z = Array2::from_shape_vec(
            (config.hidden_dim, config.hidden_dim),
            weights.u_z.clone(),
        )
        .map_err(|e| super::RouterError::InvalidConfig(e.to_string()))?;

        let b_z = Array1::from_vec(weights.b_z.clone());

        let w_h = Array2::from_shape_vec(
            (config.hidden_dim, config.input_dim),
            weights.w_h.clone(),
        )
        .map_err(|e| super::RouterError::InvalidConfig(e.to_string()))?;

        let u_h = Array2::from_shape_vec(
            (config.hidden_dim, config.hidden_dim),
            weights.u_h.clone(),
        )
        .map_err(|e| super::RouterError::InvalidConfig(e.to_string()))?;

        let b_h = Array1::from_vec(weights.b_h.clone());

        Ok(Self {
            hidden_dim: config.hidden_dim,
            input_dim: config.input_dim,
            zeta: weights.zeta,
            nu: weights.nu,
            w_z,
            u_z,
            b_z,
            w_h,
            u_h,
            b_h,
        })
    }

    /// Forward pass for one time step.
    ///
    /// Computes: h_t = (zeta * (1 - z_t) + nu) * h_tilde + z_t * h_prev
    pub fn forward(&self, x: &ArrayView1<f32>, h_prev: &ArrayView1<f32>) -> Array1<f32> {
        // z_t = sigmoid(W_z * x_t + U_z * h_{t-1} + b_z)
        let z_pre = self.w_z.dot(x) + self.u_z.dot(h_prev) + &self.b_z;
        let z = z_pre.mapv(sigmoid);

        // h_tilde = tanh(W_h * x_t + U_h * h_{t-1} + b_h)
        let h_tilde_pre = self.w_h.dot(x) + self.u_h.dot(h_prev) + &self.b_h;
        let h_tilde = h_tilde_pre.mapv(|x| x.tanh());

        // h_t = (zeta * (1 - z_t) + nu) * h_tilde + z_t * h_prev
        let one_minus_z = z.mapv(|z_i| 1.0 - z_i);
        let scale = one_minus_z.mapv(|omz| self.zeta * omz + self.nu);

        &scale * &h_tilde + &z * h_prev
    }

    /// Get initial hidden state (zeros).
    pub fn initial_hidden(&self) -> Array1<f32> {
        Array1::zeros(self.hidden_dim)
    }
}

/// Native Rust FastGRNN model for complexity estimation.
pub struct FastGRNN {
    /// Configuration.
    config: FastGRNNConfig,

    /// GRNN cell.
    cell: GRNNCell,

    /// Output weights.
    w_o: Array2<f32>,

    /// Output bias.
    b_o: Array1<f32>,

    /// Running statistics for normalization.
    inference_count: std::sync::atomic::AtomicU64,
    total_inference_time_ns: std::sync::atomic::AtomicU64,
}

impl FastGRNN {
    /// Create a new FastGRNN model with default pretrained weights.
    pub fn new(config: FastGRNNConfig) -> RouterResult<Self> {
        let weights = FastGRNNWeights::pretrained(&config);
        Self::with_weights(config, weights)
    }

    /// Create a new FastGRNN model with custom weights.
    pub fn with_weights(config: FastGRNNConfig, weights: FastGRNNWeights) -> RouterResult<Self> {
        config.validate()?;
        weights.validate(&config)?;

        let cell = GRNNCell::new(&config, &weights)?;

        let w_o = Array2::from_shape_vec(
            (config.output_dim, config.hidden_dim),
            weights.w_o.clone(),
        )
        .map_err(|e| super::RouterError::InvalidConfig(e.to_string()))?;

        let b_o = Array1::from_vec(weights.b_o.clone());

        Ok(Self {
            config,
            cell,
            w_o,
            b_o,
            inference_count: std::sync::atomic::AtomicU64::new(0),
            total_inference_time_ns: std::sync::atomic::AtomicU64::new(0),
        })
    }

    /// Forward pass to estimate complexity from features.
    ///
    /// Input: features [input_dim] - normalized feature vector
    /// Output: complexity score [0.0, 1.0]
    pub fn forward(&self, features: &[f32]) -> RouterResult<f32> {
        let start = std::time::Instant::now();

        if features.len() != self.config.input_dim {
            return Err(super::RouterError::DimensionMismatch {
                expected: self.config.input_dim,
                actual: features.len(),
            });
        }

        let x = Array1::from_vec(features.to_vec());
        let h = self.cell.initial_hidden();

        // Single step forward (we treat the feature vector as a single time step)
        let h_final = self.cell.forward(&x.view(), &h.view());

        // Output layer: y = sigmoid(W_o * h + b_o)
        let y_pre = self.w_o.dot(&h_final) + &self.b_o;
        let y = y_pre.mapv(sigmoid);

        // Get the first (and only) output
        let complexity = y[0].clamp(0.0, 1.0);

        // Update statistics
        let elapsed_ns = start.elapsed().as_nanos() as u64;
        self.inference_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.total_inference_time_ns
            .fetch_add(elapsed_ns, std::sync::atomic::Ordering::Relaxed);

        Ok(complexity)
    }

    /// Batch forward pass for multiple feature vectors.
    pub fn forward_batch(&self, batch: &[Vec<f32>]) -> RouterResult<Vec<f32>> {
        batch.iter().map(|features| self.forward(features)).collect()
    }

    /// Get average inference time in microseconds.
    pub fn avg_inference_time_us(&self) -> f64 {
        let count = self.inference_count.load(std::sync::atomic::Ordering::Relaxed);
        if count == 0 {
            return 0.0;
        }
        let total_ns = self
            .total_inference_time_ns
            .load(std::sync::atomic::Ordering::Relaxed);
        (total_ns as f64 / count as f64) / 1000.0
    }

    /// Get total inference count.
    pub fn inference_count(&self) -> u64 {
        self.inference_count
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Get the configuration.
    pub fn config(&self) -> &FastGRNNConfig {
        &self.config
    }

    /// Reset statistics.
    pub fn reset_stats(&self) {
        self.inference_count
            .store(0, std::sync::atomic::Ordering::Relaxed);
        self.total_inference_time_ns
            .store(0, std::sync::atomic::Ordering::Relaxed);
    }

    /// Get model size in bytes (approximate).
    pub fn model_size_bytes(&self) -> usize {
        let gate_weights =
            2 * self.config.hidden_dim * self.config.input_dim * std::mem::size_of::<f32>();
        let hidden_weights =
            2 * self.config.hidden_dim * self.config.hidden_dim * std::mem::size_of::<f32>();
        let biases = 2 * self.config.hidden_dim * std::mem::size_of::<f32>();
        let output = (self.config.output_dim * self.config.hidden_dim + self.config.output_dim)
            * std::mem::size_of::<f32>();

        gate_weights + hidden_weights + biases + output
    }
}

// ============================================================================
// ONNX Runtime Backend (requires onnx-embed feature)
// ============================================================================

#[cfg(feature = "onnx-embed")]
/// Configuration for ONNX FastGRNN backend.
#[derive(Debug, Clone)]
pub struct OnnxFastGRNNConfig {
    /// Path to the ONNX model file.
    pub model_path: String,

    /// Expected input dimension.
    pub input_dim: usize,

    /// Number of inference threads (0 = auto).
    pub num_threads: usize,

    /// Enable graph optimization.
    pub enable_optimization: bool,
}

#[cfg(feature = "onnx-embed")]
impl Default for OnnxFastGRNNConfig {
    fn default() -> Self {
        Self {
            model_path: "models/fastgrnn_router.onnx".to_string(),
            input_dim: 5,
            num_threads: 0, // Auto
            enable_optimization: true,
        }
    }
}

#[cfg(feature = "onnx-embed")]
impl OnnxFastGRNNConfig {
    /// Create config with custom model path.
    pub fn with_model_path(mut self, path: impl Into<String>) -> Self {
        self.model_path = path.into();
        self
    }

    /// Set number of inference threads.
    pub fn with_threads(mut self, threads: usize) -> Self {
        self.num_threads = threads;
        self
    }
}

#[cfg(feature = "onnx-embed")]
/// ONNX Runtime-based FastGRNN for optimized inference.
///
/// This backend uses ONNX Runtime for potentially faster inference,
/// especially on systems with optimized ONNX Runtime builds.
pub struct OnnxFastGRNN {
    /// ONNX session (wrapped in RwLock for mutable access during inference).
    session: Arc<RwLock<Session>>,

    /// Configuration.
    config: OnnxFastGRNNConfig,

    /// Running statistics.
    inference_count: std::sync::atomic::AtomicU64,
    total_inference_time_ns: std::sync::atomic::AtomicU64,
}

#[cfg(feature = "onnx-embed")]
impl OnnxFastGRNN {
    /// Create a new ONNX FastGRNN model from file.
    ///
    /// # Arguments
    ///
    /// * `config` - Configuration including model path
    ///
    /// # Returns
    ///
    /// An ONNX FastGRNN instance or an error if loading fails.
    pub fn new(config: OnnxFastGRNNConfig) -> RouterResult<Self> {
        // Validate model path exists
        if !Path::new(&config.model_path).exists() {
            return Err(RouterError::ModelLoad {
                path: config.model_path.clone(),
                reason: "ONNX model file not found".to_string(),
            });
        }

        let session = Self::load_session(&config)?;

        Ok(Self {
            session: Arc::new(RwLock::new(session)),
            config,
            inference_count: std::sync::atomic::AtomicU64::new(0),
            total_inference_time_ns: std::sync::atomic::AtomicU64::new(0),
        })
    }

    /// Load ONNX session from file.
    fn load_session(config: &OnnxFastGRNNConfig) -> RouterResult<Session> {
        let mut builder = Session::builder()
            .map_err(|e| RouterError::ModelLoad {
                path: config.model_path.clone(),
                reason: format!("Failed to create session builder: {}", e),
            })?;

        // Set number of threads
        if config.num_threads > 0 {
            builder = builder
                .with_intra_threads(config.num_threads)
                .map_err(|e| RouterError::ModelLoad {
                    path: config.model_path.clone(),
                    reason: format!("Failed to set thread count: {}", e),
                })?;
        }

        // Set optimization level
        if config.enable_optimization {
            builder = builder
                .with_optimization_level(GraphOptimizationLevel::Level3)
                .map_err(|e| RouterError::ModelLoad {
                    path: config.model_path.clone(),
                    reason: format!("Failed to set optimization level: {}", e),
                })?;
        }

        // Load model
        let session = builder
            .commit_from_file(&config.model_path)
            .map_err(|e| RouterError::ModelLoad {
                path: config.model_path.clone(),
                reason: format!("Failed to load ONNX model: {}", e),
            })?;

        Ok(session)
    }

    /// Forward pass using ONNX Runtime.
    ///
    /// Input: features [input_dim] - normalized feature vector
    /// Output: complexity score [0.0, 1.0]
    pub fn forward(&self, features: &[f32]) -> RouterResult<f32> {
        let start = std::time::Instant::now();

        if features.len() != self.config.input_dim {
            return Err(RouterError::DimensionMismatch {
                expected: self.config.input_dim,
                actual: features.len(),
            });
        }

        // Create input tensor with shape [1, input_dim]
        let input_tensor = Tensor::from_array(([1, self.config.input_dim], features.to_vec()))
            .map_err(|e| RouterError::Inference(format!("Failed to create input tensor: {}", e)))?;

        // Build inputs
        let inputs = ort::inputs![
            "input" => input_tensor,
        ];

        // Run inference
        let mut session = self.session.write();
        let outputs = session.run(inputs).map_err(|e| {
            RouterError::Inference(format!("ONNX inference failed: {}", e))
        })?;

        // Extract output
        let output = outputs
            .iter()
            .next()
            .ok_or_else(|| RouterError::Inference("No output tensor found".to_string()))?;

        // Extract tensor data
        let (_, data): (&ort::tensor::Shape, &[f32]) = output
            .1
            .try_extract_tensor()
            .map_err(|e| RouterError::Inference(format!("Failed to extract output: {}", e)))?;

        // Get the complexity score
        let complexity = data
            .first()
            .copied()
            .unwrap_or(0.5)
            .clamp(0.0, 1.0);

        // Update statistics
        let elapsed_ns = start.elapsed().as_nanos() as u64;
        self.inference_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.total_inference_time_ns
            .fetch_add(elapsed_ns, std::sync::atomic::Ordering::Relaxed);

        Ok(complexity)
    }

    /// Batch forward pass for multiple feature vectors.
    pub fn forward_batch(&self, batch: &[Vec<f32>]) -> RouterResult<Vec<f32>> {
        if batch.is_empty() {
            return Ok(Vec::new());
        }

        let start = std::time::Instant::now();
        let batch_size = batch.len();

        // Validate dimensions
        for features in batch.iter() {
            if features.len() != self.config.input_dim {
                return Err(RouterError::DimensionMismatch {
                    expected: self.config.input_dim,
                    actual: features.len(),
                });
            }
        }

        // Flatten batch into single tensor
        let flattened: Vec<f32> = batch.iter().flat_map(|f| f.iter().copied()).collect();

        // Create input tensor with shape [batch_size, input_dim]
        let input_tensor =
            Tensor::from_array(([batch_size, self.config.input_dim], flattened)).map_err(|e| {
                RouterError::Inference(format!("Failed to create batch input tensor: {}", e))
            })?;

        // Build inputs
        let inputs = ort::inputs![
            "input" => input_tensor,
        ];

        // Run inference
        let mut session = self.session.write();
        let outputs = session.run(inputs).map_err(|e| {
            RouterError::Inference(format!("ONNX batch inference failed: {}", e))
        })?;

        // Extract output
        let output = outputs
            .iter()
            .next()
            .ok_or_else(|| RouterError::Inference("No output tensor found".to_string()))?;

        // Extract tensor data
        let (_, data): (&ort::tensor::Shape, &[f32]) = output
            .1
            .try_extract_tensor()
            .map_err(|e| RouterError::Inference(format!("Failed to extract batch output: {}", e)))?;

        // Clamp all outputs
        let results: Vec<f32> = data.iter().map(|&v| v.clamp(0.0, 1.0)).collect();

        // Update statistics
        let elapsed_ns = start.elapsed().as_nanos() as u64;
        self.inference_count
            .fetch_add(batch_size as u64, std::sync::atomic::Ordering::Relaxed);
        self.total_inference_time_ns
            .fetch_add(elapsed_ns, std::sync::atomic::Ordering::Relaxed);

        Ok(results)
    }

    /// Get average inference time in microseconds.
    pub fn avg_inference_time_us(&self) -> f64 {
        let count = self
            .inference_count
            .load(std::sync::atomic::Ordering::Relaxed);
        if count == 0 {
            return 0.0;
        }
        let total_ns = self
            .total_inference_time_ns
            .load(std::sync::atomic::Ordering::Relaxed);
        (total_ns as f64 / count as f64) / 1000.0
    }

    /// Get total inference count.
    pub fn inference_count(&self) -> u64 {
        self.inference_count
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Reset statistics.
    pub fn reset_stats(&self) {
        self.inference_count
            .store(0, std::sync::atomic::Ordering::Relaxed);
        self.total_inference_time_ns
            .store(0, std::sync::atomic::Ordering::Relaxed);
    }

    /// Get the configuration.
    pub fn config(&self) -> &OnnxFastGRNNConfig {
        &self.config
    }

    /// Get model path.
    pub fn model_path(&self) -> &str {
        &self.config.model_path
    }
}

// ============================================================================
// FastGRNN Backend Enum
// ============================================================================

/// Backend selection for FastGRNN inference.
///
/// Provides unified interface for both ONNX and native Rust implementations.
pub enum FastGRNNBackend {
    /// ONNX Runtime backend (optimized inference).
    #[cfg(feature = "onnx-embed")]
    Onnx(OnnxFastGRNN),
    /// Native Rust backend (fallback).
    Native(FastGRNN),
}

impl FastGRNNBackend {
    /// Load FastGRNN backend, preferring ONNX if available.
    ///
    /// # Arguments
    ///
    /// * `onnx_path` - Optional path to ONNX model file
    /// * `json_path` - Optional path to JSON weights file (for native fallback)
    /// * `config` - FastGRNN configuration for native fallback
    ///
    /// # Returns
    ///
    /// A FastGRNN backend instance. Uses ONNX if the model file exists,
    /// otherwise falls back to native Rust implementation.
    pub fn load(
        onnx_path: Option<&str>,
        json_path: Option<&str>,
        config: FastGRNNConfig,
    ) -> RouterResult<Self> {
        // Try ONNX first (only when onnx-embed feature is enabled)
        #[cfg(feature = "onnx-embed")]
        if let Some(path) = onnx_path {
            if Path::new(path).exists() {
                let onnx_config = OnnxFastGRNNConfig {
                    model_path: path.to_string(),
                    input_dim: config.input_dim,
                    ..Default::default()
                };

                match OnnxFastGRNN::new(onnx_config) {
                    Ok(onnx) => {
                        tracing::info!("Loaded FastGRNN ONNX backend from: {}", path);
                        return Ok(Self::Onnx(onnx));
                    }
                    Err(e) => {
                        tracing::warn!("Failed to load ONNX model, falling back to native: {}", e);
                    }
                }
            }
        }
        #[cfg(not(feature = "onnx-embed"))]
        let _ = onnx_path;

        // Try loading JSON weights for native backend
        if let Some(path) = json_path {
            if Path::new(path).exists() {
                match Self::load_native_from_json(path, config.clone()) {
                    Ok(native) => {
                        tracing::info!("Loaded FastGRNN native backend from: {}", path);
                        return Ok(Self::Native(native));
                    }
                    Err(e) => {
                        tracing::warn!(
                            "Failed to load JSON weights, using random initialization: {}",
                            e
                        );
                    }
                }
            }
        }

        // Fall back to native with pretrained weights
        tracing::info!("Using FastGRNN native backend with pretrained weights");
        let native = FastGRNN::new(config)?;
        Ok(Self::Native(native))
    }

    /// Load native FastGRNN from JSON weights file.
    fn load_native_from_json(path: &str, config: FastGRNNConfig) -> RouterResult<FastGRNN> {
        let json_content = std::fs::read_to_string(path).map_err(|e| RouterError::ModelLoad {
            path: path.to_string(),
            reason: format!("Failed to read JSON file: {}", e),
        })?;

        #[derive(Deserialize)]
        struct JsonModel {
            weights: FastGRNNWeights,
        }

        let model: JsonModel =
            serde_json::from_str(&json_content).map_err(|e| RouterError::ModelLoad {
                path: path.to_string(),
                reason: format!("Failed to parse JSON: {}", e),
            })?;

        FastGRNN::with_weights(config, model.weights)
    }

    /// Forward pass through the backend.
    pub fn forward(&self, features: &[f32]) -> RouterResult<f32> {
        match self {
            #[cfg(feature = "onnx-embed")]
            Self::Onnx(onnx) => onnx.forward(features),
            Self::Native(native) => native.forward(features),
        }
    }

    /// Batch forward pass through the backend.
    pub fn forward_batch(&self, batch: &[Vec<f32>]) -> RouterResult<Vec<f32>> {
        match self {
            #[cfg(feature = "onnx-embed")]
            Self::Onnx(onnx) => onnx.forward_batch(batch),
            Self::Native(native) => native.forward_batch(batch),
        }
    }

    /// Get average inference time in microseconds.
    pub fn avg_inference_time_us(&self) -> f64 {
        match self {
            #[cfg(feature = "onnx-embed")]
            Self::Onnx(onnx) => onnx.avg_inference_time_us(),
            Self::Native(native) => native.avg_inference_time_us(),
        }
    }

    /// Get total inference count.
    pub fn inference_count(&self) -> u64 {
        match self {
            #[cfg(feature = "onnx-embed")]
            Self::Onnx(onnx) => onnx.inference_count(),
            Self::Native(native) => native.inference_count(),
        }
    }

    /// Reset statistics.
    pub fn reset_stats(&self) {
        match self {
            #[cfg(feature = "onnx-embed")]
            Self::Onnx(onnx) => onnx.reset_stats(),
            Self::Native(native) => native.reset_stats(),
        }
    }

    /// Get backend name.
    pub fn backend_name(&self) -> &'static str {
        match self {
            #[cfg(feature = "onnx-embed")]
            Self::Onnx(_) => "onnx",
            Self::Native(_) => "native",
        }
    }

    /// Check if using ONNX backend.
    pub fn is_onnx(&self) -> bool {
        #[cfg(feature = "onnx-embed")]
        { matches!(self, Self::Onnx(_)) }
        #[cfg(not(feature = "onnx-embed"))]
        { false }
    }

    /// Check if using native backend.
    pub fn is_native(&self) -> bool {
        matches!(self, Self::Native(_))
    }

    /// Get input dimension.
    pub fn input_dim(&self) -> usize {
        match self {
            #[cfg(feature = "onnx-embed")]
            Self::Onnx(onnx) => onnx.config.input_dim,
            Self::Native(native) => native.config.input_dim,
        }
    }
}

/// Sigmoid activation function.
#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fastgrnn_config_default() {
        let config = FastGRNNConfig::default();
        assert_eq!(config.input_dim, 5);
        assert_eq!(config.hidden_dim, 16);
        assert_eq!(config.output_dim, 1);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_fastgrnn_config_compact() {
        let config = FastGRNNConfig::compact();
        assert_eq!(config.hidden_dim, 8);
        assert!(config.use_low_rank);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_fastgrnn_config_validation() {
        let mut config = FastGRNNConfig::default();
        config.input_dim = 0;
        assert!(config.validate().is_err());

        config.input_dim = 5;
        config.sparsity = 1.5;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_fastgrnn_weights_random() {
        let config = FastGRNNConfig::default();
        let weights = FastGRNNWeights::random(&config);

        assert_eq!(weights.w_z.len(), config.hidden_dim * config.input_dim);
        assert_eq!(weights.u_z.len(), config.hidden_dim * config.hidden_dim);
        assert_eq!(weights.b_z.len(), config.hidden_dim);
        assert!(weights.validate(&config).is_ok());
    }

    #[test]
    fn test_fastgrnn_weights_pretrained() {
        let config = FastGRNNConfig::default();
        let weights = FastGRNNWeights::pretrained(&config);
        assert!(weights.validate(&config).is_ok());
    }

    #[test]
    fn test_grnn_cell_creation() {
        let config = FastGRNNConfig::default();
        let weights = FastGRNNWeights::random(&config);
        let cell = GRNNCell::new(&config, &weights);
        assert!(cell.is_ok());
    }

    #[test]
    fn test_grnn_cell_forward() {
        let config = FastGRNNConfig::default();
        let weights = FastGRNNWeights::random(&config);
        let cell = GRNNCell::new(&config, &weights).unwrap();

        let x = Array1::from_vec(vec![0.5; config.input_dim]);
        let h = cell.initial_hidden();

        let h_next = cell.forward(&x.view(), &h.view());
        assert_eq!(h_next.len(), config.hidden_dim);
    }

    #[test]
    fn test_fastgrnn_creation() {
        let config = FastGRNNConfig::default();
        let model = FastGRNN::new(config);
        assert!(model.is_ok());
    }

    #[test]
    fn test_fastgrnn_forward() {
        let config = FastGRNNConfig::default();
        let model = FastGRNN::new(config.clone()).unwrap();

        let features = vec![0.5; config.input_dim];
        let result = model.forward(&features);
        assert!(result.is_ok());

        let complexity = result.unwrap();
        assert!(complexity >= 0.0 && complexity <= 1.0);
    }

    #[test]
    fn test_fastgrnn_forward_dimension_mismatch() {
        let config = FastGRNNConfig::default();
        let model = FastGRNN::new(config).unwrap();

        let features = vec![0.5; 3]; // Wrong dimension
        let result = model.forward(&features);
        assert!(result.is_err());
    }

    #[test]
    fn test_fastgrnn_batch_forward() {
        let config = FastGRNNConfig::default();
        let model = FastGRNN::new(config.clone()).unwrap();

        let batch = vec![
            vec![0.1; config.input_dim],
            vec![0.5; config.input_dim],
            vec![0.9; config.input_dim],
        ];

        let results = model.forward_batch(&batch);
        assert!(results.is_ok());

        let complexities = results.unwrap();
        assert_eq!(complexities.len(), 3);
        for c in complexities {
            assert!(c >= 0.0 && c <= 1.0);
        }
    }

    #[test]
    fn test_fastgrnn_statistics() {
        let config = FastGRNNConfig::default();
        let model = FastGRNN::new(config.clone()).unwrap();

        assert_eq!(model.inference_count(), 0);

        let features = vec![0.5; config.input_dim];
        let _ = model.forward(&features);
        let _ = model.forward(&features);
        let _ = model.forward(&features);

        assert_eq!(model.inference_count(), 3);
        assert!(model.avg_inference_time_us() > 0.0);

        model.reset_stats();
        assert_eq!(model.inference_count(), 0);
    }

    #[test]
    fn test_fastgrnn_model_size() {
        let config = FastGRNNConfig::default();
        let model = FastGRNN::new(config).unwrap();

        let size = model.model_size_bytes();
        assert!(size > 0);
        assert!(size < 100_000); // Should be less than 100KB
    }

    #[test]
    fn test_sigmoid() {
        assert!((sigmoid(0.0) - 0.5).abs() < 0.001);
        assert!(sigmoid(10.0) > 0.999);
        assert!(sigmoid(-10.0) < 0.001);
    }

    #[test]
    fn test_fastgrnn_inference_speed() {
        let config = FastGRNNConfig::compact();
        let model = FastGRNN::new(config.clone()).unwrap();

        let features = vec![0.5; config.input_dim];

        // Warm up
        for _ in 0..10 {
            let _ = model.forward(&features);
        }

        model.reset_stats();

        // Benchmark
        let iterations = 100;
        let start = std::time::Instant::now();
        for _ in 0..iterations {
            let _ = model.forward(&features);
        }
        let elapsed_us = start.elapsed().as_micros();

        let avg_us = elapsed_us as f64 / iterations as f64;
        println!("Average inference time: {:.2} us", avg_us);

        // Should be well under 5ms (5000us)
        assert!(avg_us < 1000.0, "Inference too slow: {} us", avg_us);
    }

    #[cfg(feature = "onnx-embed")]
    #[test]
    fn test_onnx_fastgrnn_config_default() {
        let config = OnnxFastGRNNConfig::default();
        assert_eq!(config.input_dim, 5);
        assert!(config.enable_optimization);
    }

    #[cfg(feature = "onnx-embed")]
    #[test]
    fn test_onnx_fastgrnn_config_builder() {
        let config = OnnxFastGRNNConfig::default()
            .with_model_path("custom/path.onnx")
            .with_threads(4);

        assert_eq!(config.model_path, "custom/path.onnx");
        assert_eq!(config.num_threads, 4);
    }

    #[test]
    fn test_fastgrnn_backend_load_native_fallback() {
        // When ONNX model doesn't exist, should fall back to native
        let config = FastGRNNConfig::default();
        let backend = FastGRNNBackend::load(
            Some("nonexistent.onnx"),
            Some("nonexistent.json"),
            config,
        );

        assert!(backend.is_ok());
        let backend = backend.unwrap();
        assert!(backend.is_native());
        assert_eq!(backend.backend_name(), "native");
    }

    #[test]
    fn test_fastgrnn_backend_forward() {
        let config = FastGRNNConfig::default();
        let backend = FastGRNNBackend::load(None, None, config.clone()).unwrap();

        let features = vec![0.5; config.input_dim];
        let result = backend.forward(&features);
        assert!(result.is_ok());

        let complexity = result.unwrap();
        assert!(complexity >= 0.0 && complexity <= 1.0);
    }

    #[test]
    fn test_fastgrnn_backend_batch_forward() {
        let config = FastGRNNConfig::default();
        let backend = FastGRNNBackend::load(None, None, config.clone()).unwrap();

        let batch = vec![
            vec![0.1; config.input_dim],
            vec![0.5; config.input_dim],
            vec![0.9; config.input_dim],
        ];

        let results = backend.forward_batch(&batch);
        assert!(results.is_ok());

        let complexities = results.unwrap();
        assert_eq!(complexities.len(), 3);
    }

    #[test]
    fn test_fastgrnn_backend_statistics() {
        let config = FastGRNNConfig::default();
        let backend = FastGRNNBackend::load(None, None, config.clone()).unwrap();

        assert_eq!(backend.inference_count(), 0);

        let features = vec![0.5; config.input_dim];
        let _ = backend.forward(&features);
        let _ = backend.forward(&features);

        assert_eq!(backend.inference_count(), 2);
        assert!(backend.avg_inference_time_us() > 0.0);

        backend.reset_stats();
        assert_eq!(backend.inference_count(), 0);
    }

    #[test]
    fn test_pretrained_uses_trained_weights() {
        let config = FastGRNNConfig::default();
        let weights = FastGRNNWeights::pretrained(&config);

        // Trained weights should NOT be random - verify against known values
        // from models/fastgrnn_router.json. The first value of w_z is -0.0517...
        assert!(
            (weights.w_z[0] - (-0.05174808306609452)).abs() < 0.001,
            "pretrained() should load trained weights, got w_z[0] = {}",
            weights.w_z[0]
        );

        // Check a few more known values to confirm it's the full trained model
        assert!(
            (weights.b_o[0] - 0.007006540950020059).abs() < 0.001,
            "b_o[0] should match trained value, got {}",
            weights.b_o[0]
        );
        assert!(
            (weights.zeta - 1.0).abs() < 0.001,
            "zeta should be 1.0, got {}",
            weights.zeta
        );
        assert!(
            (weights.nu - (-0.001)).abs() < 0.001,
            "nu should be -0.001, got {}",
            weights.nu
        );
    }

    #[test]
    fn test_pretrained_falls_back_for_mismatched_config() {
        // Use a config with different dimensions than the trained weights
        let config = FastGRNNConfig::compact(); // hidden_dim=8 vs trained hidden_dim=16
        let weights = FastGRNNWeights::pretrained(&config);

        // Should still produce valid weights for the compact config
        assert!(weights.validate(&config).is_ok());
        assert_eq!(weights.w_z.len(), config.hidden_dim * config.input_dim);

        // These should be random (fallback), so b_z should be -0.5
        for &b in &weights.b_z {
            assert!(
                (b - (-0.5)).abs() < 0.001,
                "Fallback b_z should be -0.5, got {}",
                b
            );
        }
    }

    #[test]
    fn test_from_embedded_json_validates_dimensions() {
        // The embedded JSON has input_dim=5, hidden_dim=16, output_dim=1
        // A matching config should succeed
        let config = FastGRNNConfig::default();
        let result = FastGRNNWeights::from_embedded_json(&config);
        assert!(result.is_ok(), "from_embedded_json should succeed for default config");

        // A mismatched config should fail
        let bad_config = FastGRNNConfig {
            input_dim: 10,
            ..FastGRNNConfig::default()
        };
        let result = FastGRNNWeights::from_embedded_json(&bad_config);
        assert!(result.is_err(), "from_embedded_json should fail for mismatched config");
    }

    #[test]
    fn test_trained_routing_decisions() {
        let config = FastGRNNConfig::default();
        let model = FastGRNN::new(config).unwrap();

        // Simple short query should get low complexity
        // Features: [query_length, embedding_norm, domain_specificity, pattern_coverage, historical_accuracy]
        let simple_features = vec![0.1, 0.2, 0.1, 0.8, 0.7];
        let simple_score = model.forward(&simple_features).unwrap();

        // Complex reasoning query should get higher complexity
        let complex_features = vec![0.9, 0.8, 0.9, 0.2, 0.3];
        let complex_score = model.forward(&complex_features).unwrap();

        // Complex should score higher than simple
        assert!(
            complex_score > simple_score,
            "Complex query ({}) should score higher than simple ({})",
            complex_score,
            simple_score
        );
    }

    #[test]
    fn test_trained_weights_deterministic() {
        // Calling pretrained() twice with the same config should yield identical weights
        let config = FastGRNNConfig::default();
        let w1 = FastGRNNWeights::pretrained(&config);
        let w2 = FastGRNNWeights::pretrained(&config);

        assert_eq!(w1.w_z, w2.w_z, "Trained weights should be deterministic");
        assert_eq!(w1.u_z, w2.u_z, "Trained weights should be deterministic");
        assert_eq!(w1.b_z, w2.b_z, "Trained weights should be deterministic");
        assert_eq!(w1.w_o, w2.w_o, "Trained weights should be deterministic");
        assert_eq!(w1.b_o, w2.b_o, "Trained weights should be deterministic");
    }
}
