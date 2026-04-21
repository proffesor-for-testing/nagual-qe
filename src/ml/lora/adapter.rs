//! LoRA adapter for domain-specific embedding transformation.
//!
//! Implements Low-Rank Adaptation (LoRA) that modifies base embeddings
//! to improve retrieval accuracy for specific domains.
//!
//! The transformation is: `output = input + alpha * B @ A @ input`
//! where A is the down-projection (rank x base_dim) and B is the
//! up-projection (base_dim x rank).

use ndarray::{Array1, Array2, ArrayView1};
use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::ml::{MlError, MlResult};

/// LoRA configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoraConfig {
    /// Base embedding dimension (default: 128).
    pub base_dim: usize,
    /// LoRA rank (default: 4). Lower rank = smaller adapter, less expressiveness.
    pub rank: usize,
    /// Learning rate for training (default: 0.001).
    pub learning_rate: f32,
    /// Scaling factor alpha (default: 1.0).
    pub alpha: f32,
}

impl Default for LoraConfig {
    fn default() -> Self {
        Self {
            base_dim: 128,
            rank: 4,
            learning_rate: 0.001,
            alpha: 1.0,
        }
    }
}

/// A trained LoRA adapter.
///
/// Applies a low-rank transformation: `output = input + alpha * (B @ A @ input)`
/// where A is (rank x base_dim) and B is (base_dim x rank).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoraAdapter {
    /// Domain this adapter was trained for.
    pub domain: String,
    /// Configuration.
    pub config: LoraConfig,
    /// Down-projection matrix A (rank x base_dim), stored as flat Vec<f32>.
    pub matrix_a: Vec<f32>,
    /// Up-projection matrix B (base_dim x rank), stored as flat Vec<f32>.
    pub matrix_b: Vec<f32>,
    /// Number of training iterations completed.
    pub iterations: u32,
    /// Final training loss.
    pub final_loss: f32,
    /// Timestamp of training completion.
    pub trained_at: String,
}

impl LoraAdapter {
    /// Create a new untrained adapter with small random initialization.
    ///
    /// Matrix A is initialized with small random values (Kaiming-style),
    /// and matrix B is initialized to zeros. This ensures the initial
    /// transformation is near-identity (output ~= input).
    pub fn new(domain: impl Into<String>, config: LoraConfig) -> Self {
        let mut rng = rand::thread_rng();
        let a_size = config.rank * config.base_dim;
        let b_size = config.base_dim * config.rank;

        // Initialize A with small random values (He initialization scaled down)
        let std_dev = (2.0 / config.base_dim as f32).sqrt() * 0.01;
        let matrix_a: Vec<f32> = (0..a_size)
            .map(|_| rng.gen_range(-std_dev..std_dev))
            .collect();

        // Initialize B to zeros so initial output = input (no perturbation)
        let matrix_b: Vec<f32> = vec![0.0; b_size];

        Self {
            domain: domain.into(),
            config,
            matrix_a,
            matrix_b,
            iterations: 0,
            final_loss: f32::MAX,
            trained_at: String::new(),
        }
    }

    /// Apply the LoRA transformation to an embedding.
    ///
    /// Computes: `output = input + alpha * B @ A @ input`
    ///
    /// The result is then L2-normalized to maintain unit-length embeddings.
    pub fn transform(&self, embedding: &ArrayView1<f32>) -> MlResult<Array1<f32>> {
        if embedding.len() != self.config.base_dim {
            return Err(MlError::DimensionMismatch {
                expected: self.config.base_dim,
                actual: embedding.len(),
            });
        }

        let a = self.matrix_a_2d();
        let b = self.matrix_b_2d();

        // Step 1: down-project: z = A @ input (rank-dim vector)
        let z = a.dot(embedding);

        // Step 2: up-project: delta = B @ z (base_dim vector)
        let delta = b.dot(&z);

        // Step 3: residual connection with scaling
        let output = embedding.to_owned() + &(delta * self.config.alpha);

        // L2-normalize the output
        let norm = output.dot(&output).sqrt();
        if norm > f32::EPSILON {
            Ok(output.mapv(|x| x / norm))
        } else {
            Ok(output)
        }
    }

    /// Get matrix A as a 2D array (rank x base_dim).
    fn matrix_a_2d(&self) -> Array2<f32> {
        Array2::from_shape_vec(
            (self.config.rank, self.config.base_dim),
            self.matrix_a.clone(),
        )
        .expect("matrix_a should have correct dimensions")
    }

    /// Get matrix B as a 2D array (base_dim x rank).
    fn matrix_b_2d(&self) -> Array2<f32> {
        Array2::from_shape_vec(
            (self.config.base_dim, self.config.rank),
            self.matrix_b.clone(),
        )
        .expect("matrix_b should have correct dimensions")
    }

    /// Get the adapter size in bytes (just the matrix data).
    pub fn size_bytes(&self) -> usize {
        (self.config.rank * self.config.base_dim * 2) * std::mem::size_of::<f32>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn test_lora_config_default() {
        let config = LoraConfig::default();
        assert_eq!(config.base_dim, 128);
        assert_eq!(config.rank, 4);
        assert!((config.learning_rate - 0.001).abs() < f32::EPSILON);
        assert!((config.alpha - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_lora_adapter_new() {
        let config = LoraConfig::default();
        let adapter = LoraAdapter::new("rust", config.clone());

        assert_eq!(adapter.domain, "rust");
        assert_eq!(adapter.matrix_a.len(), config.rank * config.base_dim);
        assert_eq!(adapter.matrix_b.len(), config.base_dim * config.rank);
        assert_eq!(adapter.iterations, 0);
        assert_eq!(adapter.final_loss, f32::MAX);

        // B should be all zeros (so initial transform is near-identity)
        assert!(adapter.matrix_b.iter().all(|&x| x == 0.0));

        // A should have small non-zero values (random init)
        let a_nonzero = adapter.matrix_a.iter().any(|&x| x != 0.0);
        assert!(a_nonzero, "matrix_a should have non-zero random values");
    }

    #[test]
    fn test_lora_adapter_transform_dimensions() {
        let config = LoraConfig {
            base_dim: 8,
            rank: 2,
            ..Default::default()
        };
        let adapter = LoraAdapter::new("test", config);

        let input = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let output = adapter.transform(&input.view()).unwrap();

        assert_eq!(output.len(), 8, "output dimension should match base_dim");
    }

    #[test]
    fn test_lora_adapter_transform_preserves_base() {
        // With B initialized to zeros, the transform should produce
        // output very close to the normalized input.
        let config = LoraConfig {
            base_dim: 8,
            rank: 2,
            ..Default::default()
        };
        let adapter = LoraAdapter::new("test", config);

        let input = Array1::from_vec(vec![0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0]);
        let output = adapter.transform(&input.view()).unwrap();

        // Normalize input for comparison
        let norm = input.dot(&input).sqrt();
        let normalized_input = input.mapv(|x| x / norm);

        // Output should be very close to normalized input since B=0
        let diff: f32 = output
            .iter()
            .zip(normalized_input.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(
            diff < 0.01,
            "fresh adapter should approximately preserve input (diff={})",
            diff
        );
    }

    #[test]
    fn test_lora_adapter_transform_dimension_mismatch() {
        let config = LoraConfig {
            base_dim: 8,
            rank: 2,
            ..Default::default()
        };
        let adapter = LoraAdapter::new("test", config);

        let wrong_input = Array1::from_vec(vec![1.0, 0.0, 0.0]); // 3-dim, expected 8
        let result = adapter.transform(&wrong_input.view());
        assert!(result.is_err());
    }

    #[test]
    fn test_lora_adapter_size_bytes() {
        let config = LoraConfig {
            base_dim: 128,
            rank: 4,
            ..Default::default()
        };
        let adapter = LoraAdapter::new("test", config);

        // 4 * 128 * 2 * 4 bytes = 4096 bytes
        assert_eq!(adapter.size_bytes(), 4096);
    }

    #[test]
    fn test_lora_adapter_serialization() {
        let config = LoraConfig {
            base_dim: 8,
            rank: 2,
            ..Default::default()
        };
        let adapter = LoraAdapter::new("test_domain", config);

        let json = serde_json::to_string(&adapter).unwrap();
        let deserialized: LoraAdapter = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.domain, "test_domain");
        assert_eq!(deserialized.matrix_a.len(), adapter.matrix_a.len());
        assert_eq!(deserialized.matrix_b.len(), adapter.matrix_b.len());
    }
}
