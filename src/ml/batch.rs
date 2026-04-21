//! Batch embedding processor for efficient bulk embedding generation.
//!
//! Provides memory-efficient streaming of embeddings with progress callbacks
//! and configurable batch sizes.
//!
//! # Example
//!
//! ```ignore
//! use nagual::ml::{BatchEmbedder, Embedder, EmbedderConfig};
//!
//! let embedder = Embedder::new(&EmbedderConfig::default())?;
//! let batch_embedder = BatchEmbedder::new(embedder).with_batch_size(64);
//!
//! let records = vec![
//!     ("id1", "First text"),
//!     ("id2", "Second text"),
//!     // ... many more
//! ];
//!
//! let results = batch_embedder.process_with_callback(
//!     &records,
//!     |progress| println!("Progress: {:.1}%", progress.percent()),
//! )?;
//! ```
//!
//! # Async Support
//!
//! For async contexts, use the async variants to avoid blocking the executor:
//!
//! ```ignore
//! let results = batch_embedder.process_async(&records).await?;
//! ```

use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::RwLock;

use super::{Embedder, MlError, MlResult};

/// Default batch size for embedding generation.
pub const DEFAULT_BATCH_SIZE: usize = 32;

/// Maximum batch size to prevent memory issues.
pub const MAX_BATCH_SIZE: usize = 512;

/// Configuration for batch embedding.
#[derive(Debug, Clone)]
pub struct BatchConfig {
    /// Number of records to process in each batch.
    pub batch_size: usize,

    /// Delay between batches (for rate limiting).
    pub batch_delay: Option<Duration>,

    /// Whether to continue on individual record errors.
    pub continue_on_error: bool,

    /// Maximum number of errors before aborting.
    pub max_errors: usize,

    /// Callback frequency (every N records).
    pub callback_frequency: usize,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            batch_size: DEFAULT_BATCH_SIZE,
            batch_delay: None,
            continue_on_error: true,
            max_errors: 100,
            callback_frequency: 100,
        }
    }
}

/// Progress information during batch processing.
#[derive(Debug, Clone)]
pub struct BatchProgress {
    /// Total number of records to process.
    pub total: usize,

    /// Number of records processed so far.
    pub processed: usize,

    /// Number of successful embeddings.
    pub successful: usize,

    /// Number of failed embeddings.
    pub failed: usize,

    /// Current batch number.
    pub current_batch: usize,

    /// Total number of batches.
    pub total_batches: usize,

    /// Elapsed time since start.
    pub elapsed: Duration,

    /// Estimated time remaining.
    pub eta: Option<Duration>,

    /// Records per second.
    pub records_per_second: f64,
}

impl BatchProgress {
    /// Get the progress as a percentage (0-100).
    pub fn percent(&self) -> f64 {
        if self.total == 0 {
            100.0
        } else {
            (self.processed as f64 / self.total as f64) * 100.0
        }
    }

    /// Check if processing is complete.
    pub fn is_complete(&self) -> bool {
        self.processed >= self.total
    }
}

/// Result of batch processing.
#[derive(Debug)]
pub struct BatchResult<T> {
    /// Successfully processed items.
    pub items: Vec<T>,

    /// Number of successful items.
    pub successful: usize,

    /// Number of failed items.
    pub failed: usize,

    /// Total processing time.
    pub duration: Duration,

    /// Errors encountered (index, error message).
    pub errors: Vec<(usize, String)>,
}

impl<T> BatchResult<T> {
    /// Check if all items were processed successfully.
    pub fn is_success(&self) -> bool {
        self.failed == 0
    }

    /// Get the success rate as a percentage.
    pub fn success_rate(&self) -> f64 {
        let total = self.successful + self.failed;
        if total == 0 {
            100.0
        } else {
            (self.successful as f64 / total as f64) * 100.0
        }
    }
}

/// Processed record with embedding.
#[derive(Debug, Clone)]
pub struct ProcessedRecord {
    /// Original record ID.
    pub id: String,

    /// Original text.
    pub text: String,

    /// Generated embedding.
    pub embedding: Vec<f32>,

    /// Token count.
    pub token_count: usize,

    /// Whether input was truncated.
    pub truncated: bool,
}

/// Batch embedder for processing many records efficiently.
pub struct BatchEmbedder {
    /// The underlying embedder.
    embedder: Arc<Embedder>,

    /// Configuration.
    config: BatchConfig,

    /// Processing statistics.
    stats: RwLock<BatchStats>,
}

/// Cumulative statistics for batch processing.
#[derive(Debug, Clone, Default)]
pub struct BatchStats {
    /// Total records processed across all batches.
    pub total_records: u64,

    /// Total successful embeddings.
    pub total_successful: u64,

    /// Total failed embeddings.
    pub total_failed: u64,

    /// Total processing time.
    pub total_time_ms: u64,

    /// Total batches processed.
    pub batches_processed: u64,
}

impl BatchEmbedder {
    /// Create a new batch embedder with default configuration.
    pub fn new(embedder: Embedder) -> Self {
        Self {
            embedder: Arc::new(embedder),
            config: BatchConfig::default(),
            stats: RwLock::new(BatchStats::default()),
        }
    }

    /// Create a batch embedder from a shared embedder.
    pub fn from_arc(embedder: Arc<Embedder>) -> Self {
        Self {
            embedder,
            config: BatchConfig::default(),
            stats: RwLock::new(BatchStats::default()),
        }
    }

    /// Set the batch size.
    pub fn with_batch_size(mut self, size: usize) -> Self {
        self.config.batch_size = size.min(MAX_BATCH_SIZE).max(1);
        self
    }

    /// Set the configuration.
    pub fn with_config(mut self, config: BatchConfig) -> Self {
        self.config = config;
        self
    }

    /// Set whether to continue on individual errors.
    pub fn continue_on_error(mut self, continue_on_error: bool) -> Self {
        self.config.continue_on_error = continue_on_error;
        self
    }

    /// Set the batch delay for rate limiting.
    pub fn with_batch_delay(mut self, delay: Duration) -> Self {
        self.config.batch_delay = Some(delay);
        self
    }

    /// Process a list of (id, text) pairs and generate embeddings.
    ///
    /// # Arguments
    ///
    /// * `records` - Slice of (id, text) pairs to process
    ///
    /// # Returns
    ///
    /// A batch result containing processed records and statistics.
    pub fn process(&self, records: &[(&str, &str)]) -> MlResult<BatchResult<ProcessedRecord>> {
        self.process_with_callback(records, |_| {})
    }

    /// Process records with a progress callback.
    ///
    /// # Arguments
    ///
    /// * `records` - Slice of (id, text) pairs to process
    /// * `callback` - Function called with progress updates
    ///
    /// # Returns
    ///
    /// A batch result containing processed records and statistics.
    pub fn process_with_callback<F>(
        &self,
        records: &[(&str, &str)],
        mut callback: F,
    ) -> MlResult<BatchResult<ProcessedRecord>>
    where
        F: FnMut(&BatchProgress),
    {
        let start = Instant::now();
        let total = records.len();
        let batch_size = self.config.batch_size;
        let total_batches = (total + batch_size - 1) / batch_size;

        let mut items = Vec::with_capacity(total);
        let mut successful = 0;
        let mut failed = 0;
        let mut errors = Vec::new();
        let mut error_count = 0;

        // Process in batches
        for (batch_idx, chunk) in records.chunks(batch_size).enumerate() {
            // Rate limiting delay (synchronous - use process_async for async contexts)
            if let Some(delay) = self.config.batch_delay {
                if batch_idx > 0 {
                    std::thread::sleep(delay);
                }
            }

            // Collect texts for batch embedding
            let texts: Vec<&str> = chunk.iter().map(|(_, text)| *text).collect();

            // Try batch embedding
            match self.embedder.embed_batch(&texts) {
                Ok(embeddings) => {
                    for (_i, ((id, text), result)) in chunk.iter().zip(embeddings.iter()).enumerate()
                    {
                        items.push(ProcessedRecord {
                            id: id.to_string(),
                            text: text.to_string(),
                            embedding: result.embedding.clone(),
                            token_count: result.token_count,
                            truncated: result.truncated,
                        });
                        successful += 1;
                    }
                }
                Err(_e) => {
                    // Fall back to individual processing
                    for (i, (id, text)) in chunk.iter().enumerate() {
                        let global_idx = batch_idx * batch_size + i;

                        match self.embedder.embed(text) {
                            Ok(result) => {
                                items.push(ProcessedRecord {
                                    id: id.to_string(),
                                    text: text.to_string(),
                                    embedding: result.embedding,
                                    token_count: result.token_count,
                                    truncated: result.truncated,
                                });
                                successful += 1;
                            }
                            Err(e) => {
                                failed += 1;
                                error_count += 1;
                                errors.push((global_idx, e.to_string()));

                                if !self.config.continue_on_error
                                    || error_count >= self.config.max_errors
                                {
                                    return Err(MlError::BatchError {
                                        index: global_idx,
                                        message: format!(
                                            "Too many errors ({}/{}): {}",
                                            error_count, self.config.max_errors, e
                                        ),
                                    });
                                }
                            }
                        }
                    }
                }
            }

            // Progress callback
            let processed = (batch_idx + 1) * batch_size.min(total - batch_idx * batch_size);
            if processed % self.config.callback_frequency == 0 || batch_idx == total_batches - 1 {
                let elapsed = start.elapsed();
                let records_per_second = if elapsed.as_secs_f64() > 0.0 {
                    processed as f64 / elapsed.as_secs_f64()
                } else {
                    0.0
                };

                let eta = if records_per_second > 0.0 && processed < total {
                    Some(Duration::from_secs_f64(
                        (total - processed) as f64 / records_per_second,
                    ))
                } else {
                    None
                };

                let progress = BatchProgress {
                    total,
                    processed: processed.min(total),
                    successful,
                    failed,
                    current_batch: batch_idx + 1,
                    total_batches,
                    elapsed,
                    eta,
                    records_per_second,
                };

                callback(&progress);
            }
        }

        let duration = start.elapsed();

        // Update cumulative stats
        {
            let mut stats = self.stats.write();
            stats.total_records += total as u64;
            stats.total_successful += successful as u64;
            stats.total_failed += failed as u64;
            stats.total_time_ms += duration.as_millis() as u64;
            stats.batches_processed += total_batches as u64;
        }

        Ok(BatchResult {
            items,
            successful,
            failed,
            duration,
            errors,
        })
    }

    /// Process a streaming iterator of records.
    ///
    /// This is memory-efficient for very large datasets as it doesn't
    /// require loading all records into memory at once.
    pub fn process_stream<'a, I, F>(
        &self,
        records: I,
        total_hint: Option<usize>,
        mut callback: F,
    ) -> MlResult<BatchResult<ProcessedRecord>>
    where
        I: Iterator<Item = (&'a str, &'a str)>,
        F: FnMut(&BatchProgress),
    {
        let start = Instant::now();
        let batch_size = self.config.batch_size;

        let mut items = Vec::new();
        let mut successful = 0;
        let mut failed = 0;
        let mut errors = Vec::new();
        let mut error_count = 0;
        let mut processed = 0;
        let mut batch_idx = 0;

        let mut batch: Vec<(&str, &str)> = Vec::with_capacity(batch_size);

        for (id, text) in records {
            batch.push((id, text));

            if batch.len() >= batch_size {
                // Process batch
                let texts: Vec<&str> = batch.iter().map(|(_, text)| *text).collect();

                match self.embedder.embed_batch(&texts) {
                    Ok(embeddings) => {
                        for ((id, text), result) in batch.iter().zip(embeddings.iter()) {
                            items.push(ProcessedRecord {
                                id: id.to_string(),
                                text: text.to_string(),
                                embedding: result.embedding.clone(),
                                token_count: result.token_count,
                                truncated: result.truncated,
                            });
                            successful += 1;
                        }
                    }
                    Err(_) => {
                        // Fall back to individual processing
                        for (i, (id, text)) in batch.iter().enumerate() {
                            let global_idx = processed + i;

                            match self.embedder.embed(text) {
                                Ok(result) => {
                                    items.push(ProcessedRecord {
                                        id: id.to_string(),
                                        text: text.to_string(),
                                        embedding: result.embedding,
                                        token_count: result.token_count,
                                        truncated: result.truncated,
                                    });
                                    successful += 1;
                                }
                                Err(e) => {
                                    failed += 1;
                                    error_count += 1;
                                    errors.push((global_idx, e.to_string()));

                                    if !self.config.continue_on_error
                                        || error_count >= self.config.max_errors
                                    {
                                        return Err(MlError::BatchError {
                                            index: global_idx,
                                            message: format!("Too many errors: {}", e),
                                        });
                                    }
                                }
                            }
                        }
                    }
                }

                processed += batch.len();
                batch_idx += 1;

                // Rate limiting (synchronous - use process_stream_async for async contexts)
                if let Some(delay) = self.config.batch_delay {
                    std::thread::sleep(delay);
                }

                // Progress callback
                let elapsed = start.elapsed();
                let records_per_second = if elapsed.as_secs_f64() > 0.0 {
                    processed as f64 / elapsed.as_secs_f64()
                } else {
                    0.0
                };

                let (total, total_batches, eta) = if let Some(total) = total_hint {
                    let total_batches = (total + batch_size - 1) / batch_size;
                    let eta = if records_per_second > 0.0 && processed < total {
                        Some(Duration::from_secs_f64(
                            (total - processed) as f64 / records_per_second,
                        ))
                    } else {
                        None
                    };
                    (total, total_batches, eta)
                } else {
                    (processed, batch_idx, None)
                };

                let progress = BatchProgress {
                    total,
                    processed,
                    successful,
                    failed,
                    current_batch: batch_idx,
                    total_batches,
                    elapsed,
                    eta,
                    records_per_second,
                };

                callback(&progress);

                batch.clear();
            }
        }

        // Process remaining items
        if !batch.is_empty() {
            let texts: Vec<&str> = batch.iter().map(|(_, text)| *text).collect();

            match self.embedder.embed_batch(&texts) {
                Ok(embeddings) => {
                    for ((id, text), result) in batch.iter().zip(embeddings.iter()) {
                        items.push(ProcessedRecord {
                            id: id.to_string(),
                            text: text.to_string(),
                            embedding: result.embedding.clone(),
                            token_count: result.token_count,
                            truncated: result.truncated,
                        });
                        successful += 1;
                    }
                }
                Err(_) => {
                    for (i, (id, text)) in batch.iter().enumerate() {
                        let global_idx = processed + i;

                        match self.embedder.embed(text) {
                            Ok(result) => {
                                items.push(ProcessedRecord {
                                    id: id.to_string(),
                                    text: text.to_string(),
                                    embedding: result.embedding,
                                    token_count: result.token_count,
                                    truncated: result.truncated,
                                });
                                successful += 1;
                            }
                            Err(e) => {
                                failed += 1;
                                errors.push((global_idx, e.to_string()));
                            }
                        }
                    }
                }
            }

            processed += batch.len();
        }

        let duration = start.elapsed();

        // Final progress callback
        let progress = BatchProgress {
            total: total_hint.unwrap_or(processed),
            processed,
            successful,
            failed,
            current_batch: batch_idx + 1,
            total_batches: batch_idx + 1,
            elapsed: duration,
            eta: None,
            records_per_second: if duration.as_secs_f64() > 0.0 {
                processed as f64 / duration.as_secs_f64()
            } else {
                0.0
            },
        };
        callback(&progress);

        Ok(BatchResult {
            items,
            successful,
            failed,
            duration,
            errors,
        })
    }

    /// Process records asynchronously (non-blocking).
    ///
    /// This is the async-friendly version that uses `tokio::time::sleep`
    /// instead of `std::thread::sleep` to avoid blocking the executor.
    ///
    /// # Arguments
    ///
    /// * `records` - Slice of (id, text) pairs to process
    ///
    /// # Returns
    ///
    /// A batch result containing processed records and statistics.
    pub async fn process_async(
        &self,
        records: &[(&str, &str)],
    ) -> MlResult<BatchResult<ProcessedRecord>> {
        self.process_with_callback_async(records, |_| {}).await
    }

    /// Process records asynchronously with a progress callback.
    ///
    /// This is the async-friendly version that uses `tokio::time::sleep`
    /// instead of `std::thread::sleep` to avoid blocking the executor.
    ///
    /// # Arguments
    ///
    /// * `records` - Slice of (id, text) pairs to process
    /// * `callback` - Function called with progress updates
    ///
    /// # Returns
    ///
    /// A batch result containing processed records and statistics.
    pub async fn process_with_callback_async<F>(
        &self,
        records: &[(&str, &str)],
        mut callback: F,
    ) -> MlResult<BatchResult<ProcessedRecord>>
    where
        F: FnMut(&BatchProgress),
    {
        let start = Instant::now();
        let total = records.len();
        let batch_size = self.config.batch_size;
        let total_batches = (total + batch_size - 1) / batch_size;

        let mut items = Vec::with_capacity(total);
        let mut successful = 0;
        let mut failed = 0;
        let mut errors = Vec::new();
        let mut error_count = 0;

        // Process in batches
        for (batch_idx, chunk) in records.chunks(batch_size).enumerate() {
            // Rate limiting delay (async-friendly - does not block executor)
            if let Some(delay) = self.config.batch_delay {
                if batch_idx > 0 {
                    tokio::time::sleep(delay).await;
                }
            }

            // Collect texts for batch embedding
            let texts: Vec<&str> = chunk.iter().map(|(_, text)| *text).collect();

            // Try batch embedding
            match self.embedder.embed_batch(&texts) {
                Ok(embeddings) => {
                    for (_i, ((id, text), result)) in chunk.iter().zip(embeddings.iter()).enumerate()
                    {
                        items.push(ProcessedRecord {
                            id: id.to_string(),
                            text: text.to_string(),
                            embedding: result.embedding.clone(),
                            token_count: result.token_count,
                            truncated: result.truncated,
                        });
                        successful += 1;
                    }
                }
                Err(_e) => {
                    // Fall back to individual processing
                    for (i, (id, text)) in chunk.iter().enumerate() {
                        let global_idx = batch_idx * batch_size + i;

                        match self.embedder.embed(text) {
                            Ok(result) => {
                                items.push(ProcessedRecord {
                                    id: id.to_string(),
                                    text: text.to_string(),
                                    embedding: result.embedding,
                                    token_count: result.token_count,
                                    truncated: result.truncated,
                                });
                                successful += 1;
                            }
                            Err(e) => {
                                failed += 1;
                                error_count += 1;
                                errors.push((global_idx, e.to_string()));

                                if !self.config.continue_on_error
                                    || error_count >= self.config.max_errors
                                {
                                    return Err(MlError::BatchError {
                                        index: global_idx,
                                        message: format!(
                                            "Too many errors ({}/{}): {}",
                                            error_count, self.config.max_errors, e
                                        ),
                                    });
                                }
                            }
                        }
                    }
                }
            }

            // Progress callback
            let processed = (batch_idx + 1) * batch_size.min(total - batch_idx * batch_size);
            if processed % self.config.callback_frequency == 0 || batch_idx == total_batches - 1 {
                let elapsed = start.elapsed();
                let records_per_second = if elapsed.as_secs_f64() > 0.0 {
                    processed as f64 / elapsed.as_secs_f64()
                } else {
                    0.0
                };

                let eta = if records_per_second > 0.0 && processed < total {
                    Some(Duration::from_secs_f64(
                        (total - processed) as f64 / records_per_second,
                    ))
                } else {
                    None
                };

                let progress = BatchProgress {
                    total,
                    processed: processed.min(total),
                    successful,
                    failed,
                    current_batch: batch_idx + 1,
                    total_batches,
                    elapsed,
                    eta,
                    records_per_second,
                };

                callback(&progress);
            }
        }

        let duration = start.elapsed();

        // Update cumulative stats
        {
            let mut stats = self.stats.write();
            stats.total_records += total as u64;
            stats.total_successful += successful as u64;
            stats.total_failed += failed as u64;
            stats.total_time_ms += duration.as_millis() as u64;
            stats.batches_processed += total_batches as u64;
        }

        Ok(BatchResult {
            items,
            successful,
            failed,
            duration,
            errors,
        })
    }

    /// Get cumulative statistics.
    pub fn stats(&self) -> BatchStats {
        self.stats.read().clone()
    }

    /// Reset cumulative statistics.
    pub fn reset_stats(&self) {
        *self.stats.write() = BatchStats::default();
    }

    /// Get the batch size.
    pub fn batch_size(&self) -> usize {
        self.config.batch_size
    }

    /// Get the configuration.
    pub fn config(&self) -> &BatchConfig {
        &self.config
    }

    /// Get the underlying embedder.
    pub fn embedder(&self) -> &Embedder {
        &self.embedder
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_config_default() {
        let config = BatchConfig::default();
        assert_eq!(config.batch_size, DEFAULT_BATCH_SIZE);
        assert!(config.continue_on_error);
        assert!(config.batch_delay.is_none());
    }

    #[test]
    fn test_batch_progress_percent() {
        let progress = BatchProgress {
            total: 100,
            processed: 50,
            successful: 50,
            failed: 0,
            current_batch: 5,
            total_batches: 10,
            elapsed: Duration::from_secs(5),
            eta: Some(Duration::from_secs(5)),
            records_per_second: 10.0,
        };

        assert_eq!(progress.percent(), 50.0);
        assert!(!progress.is_complete());
    }

    #[test]
    fn test_batch_progress_complete() {
        let progress = BatchProgress {
            total: 100,
            processed: 100,
            successful: 100,
            failed: 0,
            current_batch: 10,
            total_batches: 10,
            elapsed: Duration::from_secs(10),
            eta: None,
            records_per_second: 10.0,
        };

        assert_eq!(progress.percent(), 100.0);
        assert!(progress.is_complete());
    }

    #[test]
    fn test_batch_result_success_rate() {
        let result: BatchResult<()> = BatchResult {
            items: vec![],
            successful: 90,
            failed: 10,
            duration: Duration::from_secs(1),
            errors: vec![],
        };

        assert_eq!(result.success_rate(), 90.0);
        assert!(!result.is_success());
    }

    #[test]
    fn test_batch_result_all_success() {
        let result: BatchResult<()> = BatchResult {
            items: vec![],
            successful: 100,
            failed: 0,
            duration: Duration::from_secs(1),
            errors: vec![],
        };

        assert_eq!(result.success_rate(), 100.0);
        assert!(result.is_success());
    }
}
