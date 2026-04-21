//! Machine Learning module for Nagual.
//!
//! Provides embedding generation, batch processing, and migration utilities.
//!
//! # Features
//!
//! - **HashEmbedder**: Always-available hash-based embedding (no model files needed)
//! - **Embedder**: ONNX-based embedding generation with all-MiniLM support (requires `onnx-embed` feature)
//! - **Batch Processing**: Memory-efficient batch embedding with progress callbacks (requires `onnx-embed`)
//! - **Migration**: 384->128 dimension migration with checkpointing (requires `onnx-embed`)
//! - **Quality Gate**: Recall@10 and precision validation with rollback triggers

// Always available modules
mod hash_embedder;
pub mod hyperbolic;
pub mod lora;
pub mod poincare;
mod quality;

// Always available re-exports
pub use hash_embedder::HashEmbedder;
pub use hyperbolic::{
    euclidean_to_poincare, exponential_map, logarithmic_map, poincare_distance,
    poincare_distance_f32, project_to_ball, HyperbolicConfig, HyperbolicEmbedder, HyperbolicPoint,
};
pub use poincare::{PoincareBall, PoincareKNN, PoincareModel};
pub use quality::{
    QualityConfig, QualityGate, QualityMetrics, QualityResult, ValidationSample,
};
pub use lora::{
    LoraAdapter, LoraConfig, LoraStorage, LoraTrainer, StoredAdapter, TrainingConfig,
    TrainingPair, TrainingResult,
};

// ONNX-specific modules (optional, behind onnx-embed feature)
#[cfg(feature = "onnx-embed")]
mod batch;
#[cfg(feature = "onnx-embed")]
mod embedder;
#[cfg(feature = "onnx-embed")]
mod migration;

#[cfg(feature = "onnx-embed")]
pub use batch::{BatchEmbedder, BatchProgress, BatchResult};
#[cfg(feature = "onnx-embed")]
pub use embedder::{
    CacheConfig, CacheStats, CacheStatsSnapshot, CachedEmbedder, Embedder, EmbedderConfig,
};
#[cfg(feature = "onnx-embed")]
pub use migration::{
    EmbeddingMigration, MigrationCheckpoint, MigrationConfig, MigrationProgress, MigrationResult,
};

use ndarray::{Array1, ArrayView1};
use thiserror::Error;

/// Result from embedding a single text.
///
/// Shared between ONNX embedder and hash embedder so it is always available.
#[derive(Debug, Clone)]
pub struct EmbeddingResult {
    /// The embedding vector.
    pub embedding: Vec<f32>,

    /// Whether the embedding was normalized.
    pub normalized: bool,

    /// Number of tokens in the input.
    pub token_count: usize,

    /// Whether the input was truncated.
    pub truncated: bool,
}

impl EmbeddingResult {
    /// Get the embedding as an ndarray.
    pub fn as_array(&self) -> Array1<f32> {
        Array1::from_vec(self.embedding.clone())
    }
}

/// Errors specific to ML operations.
#[derive(Error, Debug)]
pub enum MlError {
    /// ONNX Runtime error
    #[cfg(feature = "onnx-embed")]
    #[error("ONNX Runtime error: {0}")]
    Ort(#[from] ort::Error),

    /// Tokenizer error
    #[error("Tokenizer error: {0}")]
    Tokenizer(String),

    /// Model loading error
    #[error("Failed to load model from '{path}': {reason}")]
    ModelLoad { path: String, reason: String },

    /// Invalid embedding dimension
    #[error("Invalid embedding dimension: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    /// Empty input error
    #[error("Cannot embed empty input")]
    EmptyInput,

    /// Batch processing error
    #[error("Batch processing failed at index {index}: {message}")]
    BatchError { index: usize, message: String },

    /// Migration error
    #[error("Migration error: {0}")]
    Migration(String),

    /// Quality gate failure
    #[error("Quality gate failed: {metric} = {value:.4}, required >= {threshold:.4}")]
    QualityGateFailed {
        metric: String,
        value: f32,
        threshold: f32,
    },

    /// Checkpoint error
    #[error("Checkpoint error: {0}")]
    Checkpoint(String),

    /// Database error during ML operations
    #[error("Database error: {0}")]
    Database(String),

    /// I/O error
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Hyperbolic operation failed
    #[error("Hyperbolic operation failed: {0}")]
    Hyperbolic(String),
}

/// Result type for ML operations.
pub type MlResult<T> = std::result::Result<T, MlError>;

/// Normalize a vector to unit length using L2 normalization.
///
/// # Arguments
///
/// * `vector` - The input vector to normalize
///
/// # Returns
///
/// A new vector with L2 norm of 1.0. If the input is a zero vector,
/// returns the zero vector unchanged to avoid division by zero.
///
/// # Example
///
/// ```
/// use nagual::ml::normalize_l2;
/// use ndarray::Array1;
///
/// let v = Array1::from_vec(vec![3.0, 4.0]);
/// let normalized = normalize_l2(&v.view());
/// // Result is approximately [0.6, 0.8] with norm = 1.0
/// ```
pub fn normalize_l2(vector: &ArrayView1<f32>) -> Array1<f32> {
    let norm = vector.dot(vector).sqrt();
    if norm > f32::EPSILON {
        vector.mapv(|x| x / norm)
    } else {
        vector.to_owned()
    }
}

/// Normalize a vector in-place using L2 normalization.
///
/// # Arguments
///
/// * `vector` - The vector to normalize in-place
pub fn normalize_l2_inplace(vector: &mut Array1<f32>) {
    let norm = vector.dot(vector).sqrt();
    if norm > f32::EPSILON {
        vector.mapv_inplace(|x| x / norm);
    }
}

/// Compute cosine similarity between two vectors.
///
/// For normalized vectors, this is equivalent to the dot product.
/// For unnormalized vectors, this computes the full cosine similarity formula.
///
/// # Arguments
///
/// * `a` - First vector
/// * `b` - Second vector
///
/// # Returns
///
/// Cosine similarity in range [-1.0, 1.0] where:
/// - 1.0 means identical direction
/// - 0.0 means orthogonal
/// - -1.0 means opposite direction
///
/// # Example
///
/// ```
/// use nagual::ml::cosine_similarity;
/// use ndarray::Array1;
///
/// let a = Array1::from_vec(vec![1.0, 0.0]);
/// let b = Array1::from_vec(vec![1.0, 0.0]);
/// let sim = cosine_similarity(&a.view(), &b.view());
/// assert!((sim - 1.0).abs() < 0.0001);
/// ```
pub fn cosine_similarity(a: &ArrayView1<f32>, b: &ArrayView1<f32>) -> f32 {
    let dot = a.dot(b);
    let norm_a = a.dot(a).sqrt();
    let norm_b = b.dot(b).sqrt();

    if norm_a > f32::EPSILON && norm_b > f32::EPSILON {
        dot / (norm_a * norm_b)
    } else {
        0.0
    }
}

/// Compute cosine similarity between two normalized vectors.
///
/// This is an optimized version that assumes both vectors are already
/// L2-normalized, making it a simple dot product.
///
/// # Arguments
///
/// * `a` - First normalized vector
/// * `b` - Second normalized vector
///
/// # Returns
///
/// Cosine similarity (dot product for normalized vectors)
///
/// # Safety
///
/// This function assumes inputs are normalized. Using unnormalized vectors
/// will produce incorrect results.
#[inline]
pub fn cosine_similarity_normalized(a: &ArrayView1<f32>, b: &ArrayView1<f32>) -> f32 {
    a.dot(b)
}

/// Compute L2 (Euclidean) distance between two vectors.
///
/// # Arguments
///
/// * `a` - First vector
/// * `b` - Second vector
///
/// # Returns
///
/// The Euclidean distance between the vectors
pub fn l2_distance(a: &ArrayView1<f32>, b: &ArrayView1<f32>) -> f32 {
    let diff = a - b;
    diff.dot(&diff).sqrt()
}

/// Compute squared L2 distance (avoids sqrt for comparison purposes).
///
/// # Arguments
///
/// * `a` - First vector
/// * `b` - Second vector
///
/// # Returns
///
/// The squared Euclidean distance between the vectors
#[inline]
pub fn l2_distance_squared(a: &ArrayView1<f32>, b: &ArrayView1<f32>) -> f32 {
    let diff = a - b;
    diff.dot(&diff)
}

/// Convert a raw embedding slice to an ndarray Array1.
pub fn to_array1(embedding: &[f32]) -> Array1<f32> {
    Array1::from_vec(embedding.to_vec())
}

/// Validate that an embedding has the expected dimension.
pub fn validate_dimension(embedding: &[f32], expected: usize) -> MlResult<()> {
    if embedding.len() != expected {
        return Err(MlError::DimensionMismatch {
            expected,
            actual: embedding.len(),
        });
    }
    Ok(())
}

/// Check if a vector is approximately normalized (L2 norm close to 1.0).
pub fn is_normalized(vector: &ArrayView1<f32>, tolerance: f32) -> bool {
    let norm = vector.dot(vector).sqrt();
    (norm - 1.0).abs() < tolerance
}

/// Standard embedding dimensions.
pub mod dimensions {
    /// all-MiniLM-L6-v2 dimension (384)
    pub const MINILM_384: usize = 384;

    /// Optimized dimension for Nagual (128)
    pub const NAGUAL_128: usize = 128;

    /// all-MiniLM-L12 dimension (384)
    pub const MINILM_L12_384: usize = 384;

    /// BERT base dimension (768)
    pub const BERT_768: usize = 768;
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn test_normalize_l2() {
        let v = Array1::from_vec(vec![3.0, 4.0]);
        let normalized = normalize_l2(&v.view());

        // Check that the norm is 1.0
        let norm = normalized.dot(&normalized).sqrt();
        assert!((norm - 1.0).abs() < 1e-6);

        // Check values
        assert!((normalized[0] - 0.6).abs() < 1e-6);
        assert!((normalized[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_normalize_l2_zero_vector() {
        let v = Array1::from_vec(vec![0.0, 0.0, 0.0]);
        let normalized = normalize_l2(&v.view());

        // Zero vector should remain zero
        assert!(normalized.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_normalize_l2_inplace() {
        let mut v = Array1::from_vec(vec![3.0, 4.0]);
        normalize_l2_inplace(&mut v);

        let norm = v.dot(&v).sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let a = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let b = Array1::from_vec(vec![1.0, 2.0, 3.0]);

        let sim = cosine_similarity(&a.view(), &b.view());
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = Array1::from_vec(vec![1.0, 0.0]);
        let b = Array1::from_vec(vec![0.0, 1.0]);

        let sim = cosine_similarity(&a.view(), &b.view());
        assert!(sim.abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let a = Array1::from_vec(vec![1.0, 0.0]);
        let b = Array1::from_vec(vec![-1.0, 0.0]);

        let sim = cosine_similarity(&a.view(), &b.view());
        assert!((sim + 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_normalized() {
        let a = normalize_l2(&Array1::from_vec(vec![3.0, 4.0]).view());
        let b = normalize_l2(&Array1::from_vec(vec![6.0, 8.0]).view());

        let sim = cosine_similarity_normalized(&a.view(), &b.view());
        assert!((sim - 1.0).abs() < 1e-6); // Same direction
    }

    #[test]
    fn test_l2_distance() {
        let a = Array1::from_vec(vec![0.0, 0.0]);
        let b = Array1::from_vec(vec![3.0, 4.0]);

        let dist = l2_distance(&a.view(), &b.view());
        assert!((dist - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_l2_distance_squared() {
        let a = Array1::from_vec(vec![0.0, 0.0]);
        let b = Array1::from_vec(vec![3.0, 4.0]);

        let dist_sq = l2_distance_squared(&a.view(), &b.view());
        assert!((dist_sq - 25.0).abs() < 1e-6);
    }

    #[test]
    fn test_validate_dimension() {
        let embedding = vec![0.0; 128];
        assert!(validate_dimension(&embedding, 128).is_ok());
        assert!(validate_dimension(&embedding, 384).is_err());
    }

    #[test]
    fn test_is_normalized() {
        let normalized = normalize_l2(&Array1::from_vec(vec![3.0, 4.0]).view());
        assert!(is_normalized(&normalized.view(), 1e-6));

        let not_normalized = Array1::from_vec(vec![3.0, 4.0]);
        assert!(!is_normalized(&not_normalized.view(), 1e-6));
    }

    #[test]
    fn test_to_array1() {
        let slice = &[1.0, 2.0, 3.0];
        let arr = to_array1(slice);
        assert_eq!(arr.len(), 3);
        assert_eq!(arr[0], 1.0);
    }
}
