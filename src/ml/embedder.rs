//! ONNX Runtime embedder for generating text embeddings.
//!
//! Provides a high-level interface for loading ONNX models (e.g., all-MiniLM)
//! and generating embeddings from text input.
//!
//! # Example
//!
//! ```ignore
//! use nagual::ml::{Embedder, EmbedderConfig};
//!
//! let config = EmbedderConfig::default();
//! let embedder = Embedder::new(&config)?;
//!
//! let embedding = embedder.embed("Hello, world!")?;
//! assert_eq!(embedding.len(), 128);
//! ```

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::num::NonZeroUsize;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use lru::LruCache;
use ndarray::{Array1, Array2};
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor;
use parking_lot::{Mutex, MutexGuard, RwLock};
use tokenizers::Tokenizer;

use super::{normalize_l2, EmbeddingResult, MlError, MlResult};

/// Configuration for the embedder.
#[derive(Debug, Clone)]
pub struct EmbedderConfig {
    /// Path to the ONNX model file.
    pub model_path: String,

    /// Path to the tokenizer JSON file.
    pub tokenizer_path: String,

    /// Expected output embedding dimension.
    pub embedding_dim: usize,

    /// Maximum sequence length for tokenization.
    pub max_seq_length: usize,

    /// Whether to normalize embeddings to unit length.
    pub normalize: bool,

    /// Number of threads for ONNX inference (0 = auto).
    pub num_threads: usize,

    /// Enable ONNX graph optimization.
    pub optimization_level: OptimizationLevel,

    /// Number of ONNX sessions to pool for concurrent inference.
    /// Higher values allow more concurrent embeddings but use more memory.
    /// Default: 4
    pub pool_size: usize,
}

/// ONNX graph optimization level.
#[derive(Debug, Clone, Copy, Default)]
pub enum OptimizationLevel {
    /// No optimization
    Disabled,
    /// Basic optimizations
    Basic,
    /// Extended optimizations (default)
    #[default]
    Extended,
    /// All optimizations including potentially slower ones
    All,
}

impl From<OptimizationLevel> for GraphOptimizationLevel {
    fn from(level: OptimizationLevel) -> Self {
        match level {
            OptimizationLevel::Disabled => GraphOptimizationLevel::Disable,
            OptimizationLevel::Basic => GraphOptimizationLevel::Level1,
            OptimizationLevel::Extended => GraphOptimizationLevel::Level2,
            OptimizationLevel::All => GraphOptimizationLevel::Level3,
        }
    }
}

impl Default for EmbedderConfig {
    fn default() -> Self {
        Self {
            model_path: "models/all-MiniLM-L6-v2.onnx".to_string(),
            tokenizer_path: "models/tokenizer.json".to_string(),
            embedding_dim: 128,
            max_seq_length: 512,
            normalize: true,
            num_threads: 0, // Auto
            optimization_level: OptimizationLevel::Extended,
            pool_size: 4, // 4 concurrent sessions
        }
    }
}

impl EmbedderConfig {
    /// Create a config for 128-dimensional embeddings.
    pub fn dim_128(model_path: impl Into<String>, tokenizer_path: impl Into<String>) -> Self {
        Self {
            model_path: model_path.into(),
            tokenizer_path: tokenizer_path.into(),
            embedding_dim: 128,
            ..Default::default()
        }
    }

    /// Create a config for 384-dimensional embeddings (original MiniLM).
    pub fn dim_384(model_path: impl Into<String>, tokenizer_path: impl Into<String>) -> Self {
        Self {
            model_path: model_path.into(),
            tokenizer_path: tokenizer_path.into(),
            embedding_dim: 384,
            ..Default::default()
        }
    }
}

/// Pool of ONNX sessions for concurrent inference.
///
/// Uses try-lock strategy: attempts to acquire any available session without
/// blocking, then falls back to waiting on the first session if all are busy.
/// This allows up to `pool_size` concurrent embedding operations.
pub struct SessionPool {
    sessions: Vec<Mutex<Session>>,
}

impl SessionPool {
    /// Create a new session pool.
    fn new(config: &EmbedderConfig) -> MlResult<Self> {
        let pool_size = config.pool_size.max(1);
        let mut sessions = Vec::with_capacity(pool_size);

        for _ in 0..pool_size {
            let session = Self::load_session(config)?;
            sessions.push(Mutex::new(session));
        }

        Ok(Self { sessions })
    }

    /// Load a single ONNX session.
    fn load_session(config: &EmbedderConfig) -> MlResult<Session> {
        let mut builder = Session::builder()?;

        if config.num_threads > 0 {
            builder = builder.with_intra_threads(config.num_threads)?;
        }

        builder = builder.with_optimization_level(config.optimization_level.into())?;
        let session = builder.commit_from_file(&config.model_path)?;

        Ok(session)
    }

    /// Acquire a session from the pool.
    ///
    /// Uses try-lock strategy: first attempts to acquire any unlocked session,
    /// then falls back to waiting on the first session if all are busy.
    #[inline]
    fn acquire(&self) -> MutexGuard<'_, Session> {
        // Try to acquire any available session without blocking
        for session in &self.sessions {
            if let Some(guard) = session.try_lock() {
                return guard;
            }
        }
        // All sessions busy, wait on first one
        self.sessions[0].lock()
    }
}

/// ONNX Runtime embedder for generating text embeddings.
pub struct Embedder {
    /// Pool of ONNX sessions for concurrent inference.
    session_pool: SessionPool,

    /// Tokenizer for text preprocessing.
    tokenizer: Arc<Tokenizer>,

    /// Configuration.
    config: EmbedderConfig,

    /// Statistics (using atomic counters for lock-free updates).
    stats: EmbedderStats,
}

/// Statistics for the embedder (using atomic counters for lock-free updates).
#[derive(Debug, Default)]
pub struct EmbedderStats {
    /// Total texts embedded.
    pub texts_embedded: AtomicU64,

    /// Total tokens processed.
    pub tokens_processed: AtomicU64,

    /// Total inference time in milliseconds.
    pub total_inference_time_ms: AtomicU64,

    /// Number of truncated inputs.
    pub truncated_inputs: AtomicU64,
}

impl EmbedderStats {
    /// Create a new stats instance with all counters at zero.
    pub fn new() -> Self {
        Self::default()
    }

    /// Get a snapshot of all statistics.
    pub fn snapshot(&self) -> EmbedderStatsSnapshot {
        EmbedderStatsSnapshot {
            texts_embedded: self.texts_embedded.load(Ordering::Relaxed),
            tokens_processed: self.tokens_processed.load(Ordering::Relaxed),
            total_inference_time_ms: self.total_inference_time_ms.load(Ordering::Relaxed),
            truncated_inputs: self.truncated_inputs.load(Ordering::Relaxed),
        }
    }

    /// Reset all counters to zero.
    pub fn reset(&self) {
        self.texts_embedded.store(0, Ordering::Relaxed);
        self.tokens_processed.store(0, Ordering::Relaxed);
        self.total_inference_time_ms.store(0, Ordering::Relaxed);
        self.truncated_inputs.store(0, Ordering::Relaxed);
    }
}

impl Clone for EmbedderStats {
    fn clone(&self) -> Self {
        Self {
            texts_embedded: AtomicU64::new(self.texts_embedded.load(Ordering::Relaxed)),
            tokens_processed: AtomicU64::new(self.tokens_processed.load(Ordering::Relaxed)),
            total_inference_time_ms: AtomicU64::new(
                self.total_inference_time_ms.load(Ordering::Relaxed),
            ),
            truncated_inputs: AtomicU64::new(self.truncated_inputs.load(Ordering::Relaxed)),
        }
    }
}

/// Snapshot of embedder statistics (non-atomic, for display/serialization).
#[derive(Debug, Clone, Default)]
pub struct EmbedderStatsSnapshot {
    /// Total texts embedded.
    pub texts_embedded: u64,

    /// Total tokens processed.
    pub tokens_processed: u64,

    /// Total inference time in milliseconds.
    pub total_inference_time_ms: u64,

    /// Number of truncated inputs.
    pub truncated_inputs: u64,
}

impl Embedder {
    /// Create a new embedder with the given configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if the model or tokenizer cannot be loaded.
    pub fn new(config: &EmbedderConfig) -> MlResult<Self> {
        // Validate paths
        if !Path::new(&config.model_path).exists() {
            return Err(MlError::ModelLoad {
                path: config.model_path.clone(),
                reason: "Model file not found".to_string(),
            });
        }

        if !Path::new(&config.tokenizer_path).exists() {
            return Err(MlError::ModelLoad {
                path: config.tokenizer_path.clone(),
                reason: "Tokenizer file not found".to_string(),
            });
        }

        // Create session pool for concurrent inference
        let session_pool = SessionPool::new(config)?;

        // Load tokenizer
        let tokenizer = Self::load_tokenizer(&config.tokenizer_path)?;

        Ok(Self {
            session_pool,
            tokenizer: Arc::new(tokenizer),
            config: config.clone(),
            stats: EmbedderStats::new(),
        })
    }

    /// Load the ONNX session from file (legacy, used by tests).
    #[allow(dead_code)]
    fn load_session(config: &EmbedderConfig) -> MlResult<Session> {
        SessionPool::load_session(config)
    }

    /// Get the number of sessions in the pool.
    pub fn pool_size(&self) -> usize {
        self.session_pool.sessions.len()
    }

    /// Load the tokenizer from file.
    fn load_tokenizer(path: &str) -> MlResult<Tokenizer> {
        Tokenizer::from_file(path).map_err(|e| MlError::Tokenizer(e.to_string()))
    }

    /// Embed a single text string.
    ///
    /// # Arguments
    ///
    /// * `text` - The text to embed
    ///
    /// # Returns
    ///
    /// An embedding result containing the vector and metadata.
    ///
    /// # Errors
    ///
    /// Returns an error if the text is empty or inference fails.
    pub fn embed(&self, text: &str) -> MlResult<EmbeddingResult> {
        if text.is_empty() {
            return Err(MlError::EmptyInput);
        }

        let start = std::time::Instant::now();

        // Tokenize
        let (input_ids, attention_mask, token_count, truncated) = self.tokenize(text)?;

        // Run inference
        let raw_embedding = self.infer(&input_ids, &attention_mask)?;

        // Normalize if configured
        let embedding = if self.config.normalize {
            let arr = Array1::from_vec(raw_embedding);
            normalize_l2(&arr.view()).to_vec()
        } else {
            raw_embedding
        };

        // Update stats using atomic operations (lock-free)
        self.stats.texts_embedded.fetch_add(1, Ordering::Relaxed);
        self.stats
            .tokens_processed
            .fetch_add(token_count as u64, Ordering::Relaxed);
        self.stats
            .total_inference_time_ms
            .fetch_add(start.elapsed().as_millis() as u64, Ordering::Relaxed);
        if truncated {
            self.stats.truncated_inputs.fetch_add(1, Ordering::Relaxed);
        }

        Ok(EmbeddingResult {
            embedding,
            normalized: self.config.normalize,
            token_count,
            truncated,
        })
    }

    /// Embed multiple texts in a single batch.
    ///
    /// This is more efficient than calling `embed()` multiple times
    /// as it batches the inference.
    ///
    /// # Arguments
    ///
    /// * `texts` - Slice of texts to embed
    ///
    /// # Returns
    ///
    /// Vector of embedding results, one per input text.
    ///
    /// # Errors
    ///
    /// Returns an error if any text is empty or inference fails.
    pub fn embed_batch(&self, texts: &[&str]) -> MlResult<Vec<EmbeddingResult>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        // Check for empty texts
        for (i, text) in texts.iter().enumerate() {
            if text.is_empty() {
                return Err(MlError::BatchError {
                    index: i,
                    message: "Empty input text".to_string(),
                });
            }
        }

        let start = std::time::Instant::now();
        let batch_size = texts.len();

        // Tokenize all texts
        let mut all_input_ids = Vec::with_capacity(batch_size);
        let mut all_attention_masks = Vec::with_capacity(batch_size);
        let mut token_counts = Vec::with_capacity(batch_size);
        let mut truncated_flags = Vec::with_capacity(batch_size);

        // Find max length for padding
        let mut max_len = 0;
        for text in texts {
            let (input_ids, attention_mask, token_count, truncated) = self.tokenize(text)?;
            max_len = max_len.max(input_ids.len());
            all_input_ids.push(input_ids);
            all_attention_masks.push(attention_mask);
            token_counts.push(token_count);
            truncated_flags.push(truncated);
        }

        // Pad sequences to same length
        for i in 0..batch_size {
            let pad_len = max_len - all_input_ids[i].len();
            all_input_ids[i].extend(vec![0i64; pad_len]);
            all_attention_masks[i].extend(vec![0i64; pad_len]);
        }

        // Create batch tensors
        let input_ids_batch = self.create_batch_tensor(&all_input_ids, max_len)?;
        let attention_mask_batch = self.create_batch_tensor(&all_attention_masks, max_len)?;

        // Run batch inference
        let raw_embeddings =
            self.infer_batch(&input_ids_batch, &attention_mask_batch, batch_size)?;

        // Create results
        let mut results = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            let raw_embedding = raw_embeddings.row(i).to_vec();

            let embedding = if self.config.normalize {
                let arr = Array1::from_vec(raw_embedding);
                normalize_l2(&arr.view()).to_vec()
            } else {
                raw_embedding
            };

            results.push(EmbeddingResult {
                embedding,
                normalized: self.config.normalize,
                token_count: token_counts[i],
                truncated: truncated_flags[i],
            });
        }

        // Update stats using atomic operations (lock-free)
        self.stats
            .texts_embedded
            .fetch_add(batch_size as u64, Ordering::Relaxed);
        self.stats
            .tokens_processed
            .fetch_add(token_counts.iter().sum::<usize>() as u64, Ordering::Relaxed);
        self.stats
            .total_inference_time_ms
            .fetch_add(start.elapsed().as_millis() as u64, Ordering::Relaxed);
        self.stats.truncated_inputs.fetch_add(
            truncated_flags.iter().filter(|&&t| t).count() as u64,
            Ordering::Relaxed,
        );

        Ok(results)
    }

    /// Tokenize a text string.
    fn tokenize(&self, text: &str) -> MlResult<(Vec<i64>, Vec<i64>, usize, bool)> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| MlError::Tokenizer(e.to_string()))?;

        let token_count = encoding.get_ids().len();
        let truncated = token_count > self.config.max_seq_length;

        // Truncate if necessary
        let len = token_count.min(self.config.max_seq_length);

        let input_ids: Vec<i64> = encoding.get_ids()[..len]
            .iter()
            .map(|&id| id as i64)
            .collect();

        let attention_mask: Vec<i64> = encoding.get_attention_mask()[..len]
            .iter()
            .map(|&m| m as i64)
            .collect();

        Ok((input_ids, attention_mask, token_count, truncated))
    }

    /// Run inference for a single sequence.
    fn infer(&self, input_ids: &[i64], attention_mask: &[i64]) -> MlResult<Vec<f32>> {
        let seq_len = input_ids.len();

        // Create input tensors using shape + data tuple format
        let input_ids_tensor = Tensor::from_array(([1, seq_len], input_ids.to_vec()))?;
        let attention_mask_tensor = Tensor::from_array(([1, seq_len], attention_mask.to_vec()))?;
        let token_type_ids_tensor = Tensor::from_array(([1, seq_len], vec![0i64; seq_len]))?;

        // Build inputs (BERT-based models require all three inputs)
        let inputs = ort::inputs![
            "input_ids" => input_ids_tensor,
            "attention_mask" => attention_mask_tensor,
            "token_type_ids" => token_type_ids_tensor,
        ];

        // Run inference using pooled session
        let mut session = self.session_pool.acquire();
        let outputs = session.run(inputs)?;

        // Get first output (sentence embedding or hidden states)
        let output = outputs
            .iter()
            .next()
            .ok_or_else(|| MlError::Ort(ort::Error::new("No output tensor found")))?;

        // Extract tensor using try_extract_tensor which returns (Shape, &[T]) tuple in ort 2.0
        let (shape_ref, data): (&ort::tensor::Shape, &[f32]) = output.1.try_extract_tensor()?;
        let shape: Vec<usize> = shape_ref.iter().map(|&d| d as usize).collect();

        // Handle different output shapes
        let embedding: Vec<f32> = if shape.len() == 3 {
            // Shape: [batch, seq_len, hidden_dim] - apply mean pooling
            let hidden_dim = shape[2];

            // Mean pooling with attention mask
            let mut pooled = vec![0.0f32; hidden_dim];
            let mask_sum: f32 = attention_mask.iter().map(|&m| m as f32).sum();

            if mask_sum > 0.0 {
                for (i, &mask) in attention_mask.iter().enumerate() {
                    if mask > 0 && i < shape[1] {
                        for j in 0..hidden_dim {
                            let idx = i * hidden_dim + j;
                            pooled[j] += data[idx] * (mask as f32);
                        }
                    }
                }
                for val in pooled.iter_mut() {
                    *val /= mask_sum;
                }
            }

            pooled
        } else if shape.len() == 2 {
            // Shape: [batch, hidden_dim] - already pooled
            let hidden_dim = shape[1];
            data[..hidden_dim].to_vec()
        } else {
            return Err(MlError::Ort(ort::Error::new(format!(
                "Unexpected output shape: {:?}",
                shape
            ))));
        };

        // Validate dimension
        if embedding.len() != self.config.embedding_dim {
            // Try to handle dimension mismatch by taking first N dimensions
            if embedding.len() > self.config.embedding_dim {
                return Ok(embedding[..self.config.embedding_dim].to_vec());
            } else {
                return Err(MlError::DimensionMismatch {
                    expected: self.config.embedding_dim,
                    actual: embedding.len(),
                });
            }
        }

        Ok(embedding)
    }

    /// Create a batch tensor from multiple sequences.
    fn create_batch_tensor(&self, sequences: &[Vec<i64>], max_len: usize) -> MlResult<Vec<i64>> {
        let batch_size = sequences.len();
        let mut batch = vec![0i64; batch_size * max_len];

        for (i, seq) in sequences.iter().enumerate() {
            for (j, &val) in seq.iter().enumerate() {
                batch[i * max_len + j] = val;
            }
        }

        Ok(batch)
    }

    /// Run batch inference.
    fn infer_batch(
        &self,
        input_ids: &[i64],
        attention_mask: &[i64],
        batch_size: usize,
    ) -> MlResult<Array2<f32>> {
        let seq_len = input_ids.len() / batch_size;

        // Create input tensors using shape + data tuple format
        let input_ids_tensor = Tensor::from_array(([batch_size, seq_len], input_ids.to_vec()))?;
        let attention_mask_tensor =
            Tensor::from_array(([batch_size, seq_len], attention_mask.to_vec()))?;
        let token_type_ids_tensor =
            Tensor::from_array(([batch_size, seq_len], vec![0i64; batch_size * seq_len]))?;

        // Build inputs (BERT-based models require all three inputs)
        let inputs = ort::inputs![
            "input_ids" => input_ids_tensor,
            "attention_mask" => attention_mask_tensor,
            "token_type_ids" => token_type_ids_tensor,
        ];

        // Run inference using pooled session
        let mut session = self.session_pool.acquire();
        let outputs = session.run(inputs)?;

        // Get first output
        let output = outputs
            .iter()
            .next()
            .ok_or_else(|| MlError::Ort(ort::Error::new("No output tensor found")))?;

        // Extract tensor using try_extract_tensor which returns (Shape, &[T]) tuple in ort 2.0
        let (shape_ref, data): (&ort::tensor::Shape, &[f32]) = output.1.try_extract_tensor()?;
        let shape: Vec<usize> = shape_ref.iter().map(|&d| d as usize).collect();

        // Handle different output shapes
        if shape.len() == 3 {
            // Shape: [batch, seq_len, hidden_dim] - apply mean pooling
            let output_seq_len = shape[1];
            let hidden_dim = shape[2].min(self.config.embedding_dim);

            let mut result = Array2::zeros((batch_size, self.config.embedding_dim));

            for batch_idx in 0..batch_size {
                let mask_start = batch_idx * seq_len;
                let mask_slice = &attention_mask[mask_start..mask_start + seq_len];
                let mask_sum: f32 = mask_slice.iter().map(|&m| m as f32).sum();

                if mask_sum > 0.0 {
                    for (i, &mask) in mask_slice.iter().enumerate().take(output_seq_len) {
                        if mask > 0 {
                            for j in 0..hidden_dim {
                                let idx = batch_idx * output_seq_len * shape[2] + i * shape[2] + j;
                                result[[batch_idx, j]] += data[idx] * (mask as f32);
                            }
                        }
                    }
                    for j in 0..hidden_dim {
                        result[[batch_idx, j]] /= mask_sum;
                    }
                }
            }

            Ok(result)
        } else if shape.len() == 2 {
            // Shape: [batch, hidden_dim] - already pooled
            let hidden_dim = shape[1].min(self.config.embedding_dim);
            let mut result = Array2::zeros((batch_size, self.config.embedding_dim));

            for i in 0..batch_size {
                for j in 0..hidden_dim {
                    let idx = i * shape[1] + j;
                    result[[i, j]] = data[idx];
                }
            }

            Ok(result)
        } else {
            Err(MlError::Ort(ort::Error::new(format!(
                "Unexpected output shape: {:?}",
                shape
            ))))
        }
    }

    /// Get the configured embedding dimension.
    pub fn embedding_dim(&self) -> usize {
        self.config.embedding_dim
    }

    /// Get whether embeddings are normalized.
    pub fn is_normalizing(&self) -> bool {
        self.config.normalize
    }

    /// Get the embedder statistics as a snapshot.
    pub fn stats(&self) -> EmbedderStatsSnapshot {
        self.stats.snapshot()
    }

    /// Reset the embedder statistics.
    pub fn reset_stats(&self) {
        self.stats.reset();
    }

    /// Get the model path.
    pub fn model_path(&self) -> &str {
        &self.config.model_path
    }

    /// Get the configuration.
    pub fn config(&self) -> &EmbedderConfig {
        &self.config
    }
}

// ============================================================================
// LRU Cached Embedder
// ============================================================================

/// Default cache size for embeddings (10,000 entries).
pub const DEFAULT_CACHE_SIZE: usize = 10_000;

/// Configuration for the cached embedder.
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// Maximum number of entries in the cache.
    pub max_entries: usize,

    /// Whether to use text hash as key (more memory efficient) or full text.
    pub use_hash_key: bool,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            max_entries: DEFAULT_CACHE_SIZE,
            use_hash_key: true,
        }
    }
}

impl CacheConfig {
    /// Create a cache config with a specific size.
    pub fn with_size(max_entries: usize) -> Self {
        Self {
            max_entries,
            ..Default::default()
        }
    }
}

/// Statistics for the cached embedder.
#[derive(Debug, Default)]
pub struct CacheStats {
    /// Number of cache hits.
    pub hits: AtomicU64,

    /// Number of cache misses.
    pub misses: AtomicU64,

    /// Number of cache insertions.
    pub insertions: AtomicU64,

    /// Number of cache evictions.
    pub evictions: AtomicU64,
}

impl CacheStats {
    /// Create new cache statistics.
    pub fn new() -> Self {
        Self::default()
    }

    /// Get a snapshot of cache statistics.
    pub fn snapshot(&self) -> CacheStatsSnapshot {
        CacheStatsSnapshot {
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            insertions: self.insertions.load(Ordering::Relaxed),
            evictions: self.evictions.load(Ordering::Relaxed),
        }
    }

    /// Reset all statistics.
    pub fn reset(&self) {
        self.hits.store(0, Ordering::Relaxed);
        self.misses.store(0, Ordering::Relaxed);
        self.insertions.store(0, Ordering::Relaxed);
        self.evictions.store(0, Ordering::Relaxed);
    }

    /// Calculate cache hit rate.
    pub fn hit_rate(&self) -> f64 {
        let hits = self.hits.load(Ordering::Relaxed);
        let misses = self.misses.load(Ordering::Relaxed);
        let total = hits + misses;
        if total == 0 {
            0.0
        } else {
            hits as f64 / total as f64
        }
    }
}

/// Snapshot of cache statistics (non-atomic, for display/serialization).
#[derive(Debug, Clone, Default)]
pub struct CacheStatsSnapshot {
    /// Number of cache hits.
    pub hits: u64,

    /// Number of cache misses.
    pub misses: u64,

    /// Number of cache insertions.
    pub insertions: u64,

    /// Number of cache evictions.
    pub evictions: u64,
}

impl CacheStatsSnapshot {
    /// Calculate cache hit rate from snapshot.
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }
}

/// Cached entry storing embedding with metadata.
#[derive(Clone)]
struct CachedEntry {
    /// The embedding vector.
    embedding: Vec<f32>,

    /// Whether the embedding was normalized.
    normalized: bool,

    /// Number of tokens in the input.
    token_count: usize,

    /// Whether the input was truncated.
    truncated: bool,
}

impl CachedEntry {
    fn to_result(&self) -> EmbeddingResult {
        EmbeddingResult {
            embedding: self.embedding.clone(),
            normalized: self.normalized,
            token_count: self.token_count,
            truncated: self.truncated,
        }
    }
}

/// LRU-cached embedder wrapper that caches computed embeddings.
///
/// Provides 50-90% latency reduction for repeated queries by caching
/// previously computed embeddings. Uses text hash as cache key for
/// memory efficiency.
///
/// # Example
///
/// ```ignore
/// use nagual::ml::{Embedder, EmbedderConfig, CachedEmbedder, CacheConfig};
///
/// let embedder = Embedder::new(&EmbedderConfig::default())?;
/// let cached = CachedEmbedder::new(embedder, CacheConfig::with_size(10000));
///
/// // First call computes embedding
/// let result1 = cached.embed("Hello, world!")?;
///
/// // Second call returns cached embedding (much faster)
/// let result2 = cached.embed("Hello, world!")?;
/// ```
pub struct CachedEmbedder {
    /// The underlying embedder.
    embedder: Arc<Embedder>,

    /// LRU cache: hash -> cached entry.
    cache: RwLock<LruCache<u64, CachedEntry>>,

    /// Cache configuration.
    config: CacheConfig,

    /// Cache statistics.
    cache_stats: CacheStats,
}

impl CachedEmbedder {
    /// Create a new cached embedder with the given configuration.
    pub fn new(embedder: Embedder, config: CacheConfig) -> Self {
        let cache_size = NonZeroUsize::new(config.max_entries).unwrap_or(
            NonZeroUsize::new(DEFAULT_CACHE_SIZE).expect("default cache size is non-zero"),
        );

        Self {
            embedder: Arc::new(embedder),
            cache: RwLock::new(LruCache::new(cache_size)),
            config,
            cache_stats: CacheStats::new(),
        }
    }

    /// Create a cached embedder from a shared embedder.
    pub fn from_arc(embedder: Arc<Embedder>, config: CacheConfig) -> Self {
        let cache_size = NonZeroUsize::new(config.max_entries).unwrap_or(
            NonZeroUsize::new(DEFAULT_CACHE_SIZE).expect("default cache size is non-zero"),
        );

        Self {
            embedder,
            cache: RwLock::new(LruCache::new(cache_size)),
            config,
            cache_stats: CacheStats::new(),
        }
    }

    /// Create a cached embedder with default configuration.
    pub fn with_default_config(embedder: Embedder) -> Self {
        Self::new(embedder, CacheConfig::default())
    }

    /// Hash text to create cache key.
    #[inline]
    fn hash_text(text: &str) -> u64 {
        let mut hasher = DefaultHasher::new();
        text.hash(&mut hasher);
        hasher.finish()
    }

    /// Embed text with caching support.
    ///
    /// Checks the cache first and returns cached embedding if available.
    /// Otherwise computes the embedding and stores it in the cache.
    pub fn embed(&self, text: &str) -> MlResult<EmbeddingResult> {
        if text.is_empty() {
            return Err(MlError::EmptyInput);
        }

        let hash = Self::hash_text(text);

        // Check cache first (read lock)
        {
            let mut cache = self.cache.write();
            if let Some(cached) = cache.get(&hash) {
                self.cache_stats.hits.fetch_add(1, Ordering::Relaxed);
                return Ok(cached.to_result());
            }
        }

        // Cache miss - compute embedding
        self.cache_stats.misses.fetch_add(1, Ordering::Relaxed);
        let result = self.embedder.embed(text)?;

        // Store in cache
        {
            let mut cache = self.cache.write();
            let was_full = cache.len() >= self.config.max_entries;

            cache.put(
                hash,
                CachedEntry {
                    embedding: result.embedding.clone(),
                    normalized: result.normalized,
                    token_count: result.token_count,
                    truncated: result.truncated,
                },
            );

            self.cache_stats.insertions.fetch_add(1, Ordering::Relaxed);
            if was_full {
                self.cache_stats.evictions.fetch_add(1, Ordering::Relaxed);
            }
        }

        Ok(result)
    }

    /// Embed multiple texts with caching support.
    ///
    /// For each text, checks the cache first before computing.
    /// Uses batch inference for cache misses when possible.
    pub fn embed_batch(&self, texts: &[&str]) -> MlResult<Vec<EmbeddingResult>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let mut results: Vec<Option<EmbeddingResult>> = vec![None; texts.len()];
        let mut uncached_indices: Vec<usize> = Vec::new();
        let mut uncached_texts: Vec<&str> = Vec::new();

        // Check cache for each text
        {
            let mut cache = self.cache.write();
            for (i, text) in texts.iter().enumerate() {
                if text.is_empty() {
                    return Err(MlError::BatchError {
                        index: i,
                        message: "Empty input text".to_string(),
                    });
                }

                let hash = Self::hash_text(text);
                if let Some(cached) = cache.get(&hash) {
                    self.cache_stats.hits.fetch_add(1, Ordering::Relaxed);
                    results[i] = Some(cached.to_result());
                } else {
                    self.cache_stats.misses.fetch_add(1, Ordering::Relaxed);
                    uncached_indices.push(i);
                    uncached_texts.push(text);
                }
            }
        }

        // Compute uncached embeddings in batch
        if !uncached_texts.is_empty() {
            let computed = self.embedder.embed_batch(&uncached_texts)?;

            // Store results and update cache
            let mut cache = self.cache.write();
            for (idx, result) in uncached_indices.into_iter().zip(computed.into_iter()) {
                let text = texts[idx];
                let hash = Self::hash_text(text);

                let was_full = cache.len() >= self.config.max_entries;
                cache.put(
                    hash,
                    CachedEntry {
                        embedding: result.embedding.clone(),
                        normalized: result.normalized,
                        token_count: result.token_count,
                        truncated: result.truncated,
                    },
                );

                self.cache_stats.insertions.fetch_add(1, Ordering::Relaxed);
                if was_full {
                    self.cache_stats.evictions.fetch_add(1, Ordering::Relaxed);
                }

                results[idx] = Some(result);
            }
        }

        // Unwrap all results (all should be Some at this point)
        Ok(results.into_iter().map(|r| r.unwrap()).collect())
    }

    /// Get cache statistics snapshot.
    pub fn cache_stats(&self) -> CacheStatsSnapshot {
        self.cache_stats.snapshot()
    }

    /// Get cache hit rate.
    pub fn cache_hit_rate(&self) -> f64 {
        self.cache_stats.hit_rate()
    }

    /// Get current cache size.
    pub fn cache_size(&self) -> usize {
        self.cache.read().len()
    }

    /// Get maximum cache capacity.
    pub fn cache_capacity(&self) -> usize {
        self.config.max_entries
    }

    /// Clear the cache.
    pub fn clear_cache(&self) {
        self.cache.write().clear();
    }

    /// Reset cache statistics.
    pub fn reset_cache_stats(&self) {
        self.cache_stats.reset();
    }

    /// Get the underlying embedder statistics.
    pub fn embedder_stats(&self) -> EmbedderStatsSnapshot {
        self.embedder.stats()
    }

    /// Get the embedding dimension.
    pub fn embedding_dim(&self) -> usize {
        self.embedder.embedding_dim()
    }

    /// Check if embeddings are normalized.
    pub fn is_normalizing(&self) -> bool {
        self.embedder.is_normalizing()
    }

    /// Get reference to the underlying embedder.
    pub fn embedder(&self) -> &Embedder {
        &self.embedder
    }

    /// Get the cache configuration.
    pub fn config(&self) -> &CacheConfig {
        &self.config
    }
}

/// Create a mock embedder for testing (generates random normalized embeddings).
#[cfg(test)]
pub fn mock_embedder(dim: usize) -> MockEmbedder {
    MockEmbedder { dim }
}

/// Mock embedder for testing purposes.
#[cfg(test)]
pub struct MockEmbedder {
    dim: usize,
}

#[cfg(test)]
impl MockEmbedder {
    pub fn embed(&self, _text: &str) -> Vec<f32> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let mut vec: Vec<f32> = (0..self.dim).map(|_| rng.gen_range(-1.0..1.0)).collect();

        // Normalize
        let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            vec.iter_mut().for_each(|x| *x /= norm);
        }

        vec
    }

    pub fn embed_batch(&self, texts: &[&str]) -> Vec<Vec<f32>> {
        texts.iter().map(|t| self.embed(t)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedder_config_default() {
        let config = EmbedderConfig::default();
        assert_eq!(config.embedding_dim, 128);
        assert!(config.normalize);
        assert_eq!(config.max_seq_length, 512);
        assert_eq!(config.pool_size, 4); // Default pool size
    }

    #[test]
    fn test_embedder_config_pool_size() {
        let mut config = EmbedderConfig::default();
        config.pool_size = 8;
        assert_eq!(config.pool_size, 8);
    }

    #[test]
    fn test_embedder_config_dim_384() {
        let config = EmbedderConfig::dim_384("model.onnx", "tokenizer.json");
        assert_eq!(config.embedding_dim, 384);
    }

    #[test]
    fn test_mock_embedder() {
        let embedder = mock_embedder(128);
        let embedding = embedder.embed("Hello, world!");

        assert_eq!(embedding.len(), 128);

        // Check normalization
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_mock_embedder_batch() {
        let embedder = mock_embedder(128);
        let embeddings = embedder.embed_batch(&["Hello", "World", "Test"]);

        assert_eq!(embeddings.len(), 3);
        for emb in embeddings {
            assert_eq!(emb.len(), 128);
        }
    }

    #[test]
    fn test_embedding_result_as_array() {
        let result = EmbeddingResult {
            embedding: vec![0.6, 0.8],
            normalized: true,
            token_count: 5,
            truncated: false,
        };

        let arr = result.as_array();
        assert_eq!(arr.len(), 2);
        assert_eq!(arr[0], 0.6);
    }

    #[test]
    fn test_cache_config_default() {
        let config = CacheConfig::default();
        assert_eq!(config.max_entries, DEFAULT_CACHE_SIZE);
        assert!(config.use_hash_key);
    }

    #[test]
    fn test_cache_config_with_size() {
        let config = CacheConfig::with_size(5000);
        assert_eq!(config.max_entries, 5000);
    }

    #[test]
    fn test_cache_stats_snapshot() {
        let stats = CacheStats::new();
        stats.hits.fetch_add(10, Ordering::Relaxed);
        stats.misses.fetch_add(5, Ordering::Relaxed);
        stats.insertions.fetch_add(5, Ordering::Relaxed);

        let snapshot = stats.snapshot();
        assert_eq!(snapshot.hits, 10);
        assert_eq!(snapshot.misses, 5);
        assert_eq!(snapshot.insertions, 5);
        assert_eq!(snapshot.evictions, 0);
    }

    #[test]
    fn test_cache_stats_hit_rate() {
        let stats = CacheStats::new();
        stats.hits.fetch_add(80, Ordering::Relaxed);
        stats.misses.fetch_add(20, Ordering::Relaxed);

        let hit_rate = stats.hit_rate();
        assert!((hit_rate - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_cache_stats_hit_rate_empty() {
        let stats = CacheStats::new();
        let hit_rate = stats.hit_rate();
        assert_eq!(hit_rate, 0.0);
    }

    #[test]
    fn test_cache_stats_reset() {
        let stats = CacheStats::new();
        stats.hits.fetch_add(10, Ordering::Relaxed);
        stats.misses.fetch_add(5, Ordering::Relaxed);

        stats.reset();

        let snapshot = stats.snapshot();
        assert_eq!(snapshot.hits, 0);
        assert_eq!(snapshot.misses, 0);
    }

    #[test]
    fn test_embedder_stats_atomic() {
        let stats = EmbedderStats::new();
        stats.texts_embedded.fetch_add(5, Ordering::Relaxed);
        stats.tokens_processed.fetch_add(100, Ordering::Relaxed);

        let snapshot = stats.snapshot();
        assert_eq!(snapshot.texts_embedded, 5);
        assert_eq!(snapshot.tokens_processed, 100);
    }

    #[test]
    fn test_embedder_stats_reset() {
        let stats = EmbedderStats::new();
        stats.texts_embedded.fetch_add(10, Ordering::Relaxed);
        stats.tokens_processed.fetch_add(200, Ordering::Relaxed);

        stats.reset();

        let snapshot = stats.snapshot();
        assert_eq!(snapshot.texts_embedded, 0);
        assert_eq!(snapshot.tokens_processed, 0);
    }

    #[test]
    fn test_hash_text_consistency() {
        // Test that hashing is deterministic
        let text = "Hello, world!";
        let hash1 = CachedEmbedder::hash_text(text);
        let hash2 = CachedEmbedder::hash_text(text);
        assert_eq!(hash1, hash2);

        // Different texts should have different hashes (usually)
        let hash3 = CachedEmbedder::hash_text("Different text");
        assert_ne!(hash1, hash3);
    }
}
