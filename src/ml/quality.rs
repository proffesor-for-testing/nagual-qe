//! Quality gate validation for embedding migrations.
//!
//! Provides sample-based validation of embedding quality using Recall@K
//! and Precision metrics. Triggers rollback if quality thresholds are not met.
//!
//! # Metrics
//!
//! - **Recall@K**: Measures how many of the true K nearest neighbors are found
//! - **Precision**: Measures the accuracy of retrieved neighbors
//!
//! # Example
//!
//! ```ignore
//! use nagual::ml::{QualityGate, QualityConfig};
//!
//! let config = QualityConfig::default();
//! let gate = QualityGate::new(config);
//!
//! // Validate sample of embeddings
//! let result = gate.validate(&samples)?;
//!
//! if result.passed {
//!     println!("Quality gate passed: Recall@10 = {:.2}", result.recall_at_k);
//! } else {
//!     println!("Quality gate FAILED - triggering rollback");
//! }
//! ```

use std::collections::HashSet;
use std::time::{Duration, Instant};

use ndarray::Array1;
use rand::seq::SliceRandom;
use rand::thread_rng;

use super::{cosine_similarity, normalize_l2, MlError, MlResult};

/// Configuration for quality gate validation.
#[derive(Debug, Clone)]
pub struct QualityConfig {
    /// Minimum required Recall@K.
    pub min_recall_at_k: f32,

    /// K value for Recall@K.
    pub k: usize,

    /// Minimum required precision.
    pub min_precision: f32,

    /// Number of samples to validate (None = all).
    pub sample_size: Option<usize>,

    /// Random seed for sampling (None = random).
    pub seed: Option<u64>,

    /// Whether to abort immediately on failure.
    pub fail_fast: bool,

    /// Maximum time for validation (seconds).
    pub timeout_secs: Option<u64>,

    /// Number of reference embeddings to compare against.
    pub reference_size: usize,
}

impl Default for QualityConfig {
    fn default() -> Self {
        Self {
            min_recall_at_k: 0.85,
            k: 10,
            min_precision: 0.80,
            sample_size: Some(1000),
            seed: None,
            fail_fast: false,
            timeout_secs: Some(300), // 5 minutes
            reference_size: 100,
        }
    }
}

impl QualityConfig {
    /// Create a strict quality config.
    pub fn strict() -> Self {
        Self {
            min_recall_at_k: 0.95,
            min_precision: 0.90,
            fail_fast: true,
            ..Default::default()
        }
    }

    /// Create a lenient quality config.
    pub fn lenient() -> Self {
        Self {
            min_recall_at_k: 0.70,
            min_precision: 0.65,
            sample_size: Some(100),
            ..Default::default()
        }
    }
}

/// A sample for validation containing old and new embeddings.
#[derive(Debug, Clone)]
pub struct ValidationSample {
    /// Record ID.
    pub id: String,

    /// Original text.
    pub text: String,

    /// Old embedding (384-dim).
    pub old_embedding: Vec<f32>,

    /// New embedding (128-dim).
    pub new_embedding: Vec<f32>,

    /// Ground truth neighbors (from old embeddings).
    pub ground_truth_neighbors: Option<Vec<String>>,
}

impl ValidationSample {
    /// Create a new validation sample.
    pub fn new(
        id: impl Into<String>,
        text: impl Into<String>,
        old_embedding: Vec<f32>,
        new_embedding: Vec<f32>,
    ) -> Self {
        Self {
            id: id.into(),
            text: text.into(),
            old_embedding,
            new_embedding,
            ground_truth_neighbors: None,
        }
    }

    /// Set ground truth neighbors.
    pub fn with_ground_truth(mut self, neighbors: Vec<String>) -> Self {
        self.ground_truth_neighbors = Some(neighbors);
        self
    }
}

/// Quality metrics from validation.
#[derive(Debug, Clone)]
pub struct QualityMetrics {
    /// Recall@K score (0.0 - 1.0).
    pub recall_at_k: f32,

    /// K value used.
    pub k: usize,

    /// Precision score (0.0 - 1.0).
    pub precision: f32,

    /// Mean Reciprocal Rank (MRR).
    pub mrr: f32,

    /// Number of samples validated.
    pub samples_validated: usize,

    /// Number of samples failed.
    pub samples_failed: usize,

    /// Validation duration.
    pub duration: Duration,
}

impl QualityMetrics {
    /// Get the F1 score (harmonic mean of precision and recall).
    pub fn f1_score(&self) -> f32 {
        if self.precision + self.recall_at_k > 0.0 {
            2.0 * (self.precision * self.recall_at_k) / (self.precision + self.recall_at_k)
        } else {
            0.0
        }
    }
}

/// Result of quality gate validation.
#[derive(Debug, Clone)]
pub struct QualityResult {
    /// Whether the quality gate passed.
    pub passed: bool,

    /// Quality metrics.
    pub metrics: QualityMetrics,

    /// Reason for failure (if any).
    pub failure_reason: Option<String>,

    /// Whether rollback is recommended.
    pub recommend_rollback: bool,

    /// Per-sample results (optional, for debugging).
    pub sample_results: Option<Vec<SampleResult>>,
}

/// Result for a single sample.
#[derive(Debug, Clone)]
pub struct SampleResult {
    /// Sample ID.
    pub id: String,

    /// Recall for this sample.
    pub recall: f32,

    /// Precision for this sample.
    pub precision: f32,

    /// Rank of first correct match.
    pub first_match_rank: Option<usize>,
}

/// Quality gate for validating embedding migrations.
pub struct QualityGate {
    /// Configuration.
    config: QualityConfig,
}

impl QualityGate {
    /// Create a new quality gate with the given configuration.
    pub fn new(config: QualityConfig) -> Self {
        Self { config }
    }

    /// Validate embeddings against quality thresholds.
    ///
    /// This performs sample-based validation, comparing new embeddings
    /// against old embeddings to ensure semantic similarity is preserved.
    pub fn validate(&self, samples: &[ValidationSample]) -> MlResult<QualityResult> {
        let start = Instant::now();

        if samples.is_empty() {
            return Ok(QualityResult {
                passed: true,
                metrics: QualityMetrics {
                    recall_at_k: 1.0,
                    k: self.config.k,
                    precision: 1.0,
                    mrr: 1.0,
                    samples_validated: 0,
                    samples_failed: 0,
                    duration: start.elapsed(),
                },
                failure_reason: None,
                recommend_rollback: false,
                sample_results: None,
            });
        }

        // Select samples for validation
        let validation_samples = self.select_samples(samples);
        let num_samples = validation_samples.len();

        // Build reference index from old embeddings
        let reference_embeddings = self.build_reference_index(samples);

        let mut total_recall = 0.0f32;
        let mut total_precision = 0.0f32;
        let mut total_rr = 0.0f32; // Reciprocal rank
        let mut samples_failed = 0usize;
        let mut sample_results = Vec::with_capacity(num_samples);

        for sample in &validation_samples {
            // Check timeout
            if let Some(timeout) = self.config.timeout_secs {
                if start.elapsed().as_secs() > timeout {
                    return Err(MlError::QualityGateFailed {
                        metric: "timeout".to_string(),
                        value: start.elapsed().as_secs_f32(),
                        threshold: timeout as f32,
                    });
                }
            }

            // Get ground truth neighbors using old embedding
            let ground_truth = self.find_neighbors_old(
                &sample.old_embedding,
                &reference_embeddings,
                self.config.k,
            );

            // Get predicted neighbors using new embedding
            let predicted = self.find_neighbors_new(
                &sample.new_embedding,
                samples,
                self.config.k,
            );

            // Calculate metrics
            let (recall, precision, rr) = self.calculate_sample_metrics(
                &ground_truth,
                &predicted,
            );

            if recall < self.config.min_recall_at_k || precision < self.config.min_precision {
                samples_failed += 1;

                if self.config.fail_fast {
                    let failure_reason = if recall < self.config.min_recall_at_k {
                        format!(
                            "Recall@{} = {:.4} < {:.4} for sample {}",
                            self.config.k, recall, self.config.min_recall_at_k, sample.id
                        )
                    } else {
                        format!(
                            "Precision = {:.4} < {:.4} for sample {}",
                            precision, self.config.min_precision, sample.id
                        )
                    };

                    return Ok(QualityResult {
                        passed: false,
                        metrics: QualityMetrics {
                            recall_at_k: recall,
                            k: self.config.k,
                            precision,
                            mrr: rr,
                            samples_validated: sample_results.len() + 1,
                            samples_failed: 1,
                            duration: start.elapsed(),
                        },
                        failure_reason: Some(failure_reason),
                        recommend_rollback: true,
                        sample_results: None,
                    });
                }
            }

            total_recall += recall;
            total_precision += precision;
            total_rr += rr;

            sample_results.push(SampleResult {
                id: sample.id.clone(),
                recall,
                precision,
                first_match_rank: if rr > 0.0 { Some((1.0 / rr) as usize) } else { None },
            });
        }

        let avg_recall = total_recall / num_samples as f32;
        let avg_precision = total_precision / num_samples as f32;
        let mrr = total_rr / num_samples as f32;

        let passed = avg_recall >= self.config.min_recall_at_k
            && avg_precision >= self.config.min_precision;

        let failure_reason = if !passed {
            if avg_recall < self.config.min_recall_at_k {
                Some(format!(
                    "Average Recall@{} = {:.4} < {:.4}",
                    self.config.k, avg_recall, self.config.min_recall_at_k
                ))
            } else {
                Some(format!(
                    "Average Precision = {:.4} < {:.4}",
                    avg_precision, self.config.min_precision
                ))
            }
        } else {
            None
        };

        Ok(QualityResult {
            passed,
            metrics: QualityMetrics {
                recall_at_k: avg_recall,
                k: self.config.k,
                precision: avg_precision,
                mrr,
                samples_validated: num_samples,
                samples_failed,
                duration: start.elapsed(),
            },
            failure_reason,
            recommend_rollback: !passed,
            sample_results: Some(sample_results),
        })
    }

    /// Select samples for validation (random sampling if configured).
    fn select_samples<'a>(&self, samples: &'a [ValidationSample]) -> Vec<&'a ValidationSample> {
        if let Some(sample_size) = self.config.sample_size {
            if sample_size >= samples.len() {
                return samples.iter().collect();
            }

            use rand::SeedableRng;
            let mut rng = if let Some(seed) = self.config.seed {
                rand::rngs::StdRng::seed_from_u64(seed)
            } else {
                rand::rngs::StdRng::from_rng(thread_rng()).unwrap_or_else(|_| {
                    rand::rngs::StdRng::seed_from_u64(42)
                })
            };

            let mut indices: Vec<usize> = (0..samples.len()).collect();
            indices.shuffle(&mut rng);
            indices.truncate(sample_size);

            indices.iter().map(|&i| &samples[i]).collect()
        } else {
            samples.iter().collect()
        }
    }

    /// Build reference index from old embeddings.
    fn build_reference_index(&self, samples: &[ValidationSample]) -> Vec<(String, Array1<f32>)> {
        let mut rng = thread_rng();
        let mut indices: Vec<usize> = (0..samples.len()).collect();
        indices.shuffle(&mut rng);
        indices.truncate(self.config.reference_size);

        indices
            .iter()
            .map(|&i| {
                let sample = &samples[i];
                let arr = normalize_l2(&Array1::from_vec(sample.old_embedding.clone()).view());
                (sample.id.clone(), arr)
            })
            .collect()
    }

    /// Find K nearest neighbors using old embedding.
    fn find_neighbors_old(
        &self,
        query: &[f32],
        reference: &[(String, Array1<f32>)],
        k: usize,
    ) -> Vec<String> {
        let query_arr = normalize_l2(&Array1::from_vec(query.to_vec()).view());

        let mut similarities: Vec<(String, f32)> = reference
            .iter()
            .map(|(id, emb)| {
                let sim = cosine_similarity(&query_arr.view(), &emb.view());
                (id.clone(), sim)
            })
            .collect();

        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        similarities.into_iter().take(k).map(|(id, _)| id).collect()
    }

    /// Find K nearest neighbors using new embedding.
    fn find_neighbors_new(
        &self,
        query: &[f32],
        samples: &[ValidationSample],
        k: usize,
    ) -> Vec<String> {
        let query_arr = normalize_l2(&Array1::from_vec(query.to_vec()).view());

        let mut similarities: Vec<(String, f32)> = samples
            .iter()
            .map(|s| {
                let emb = normalize_l2(&Array1::from_vec(s.new_embedding.clone()).view());
                let sim = cosine_similarity(&query_arr.view(), &emb.view());
                (s.id.clone(), sim)
            })
            .collect();

        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        similarities.into_iter().take(k).map(|(id, _)| id).collect()
    }

    /// Calculate recall, precision, and reciprocal rank for a sample.
    fn calculate_sample_metrics(
        &self,
        ground_truth: &[String],
        predicted: &[String],
    ) -> (f32, f32, f32) {
        if ground_truth.is_empty() || predicted.is_empty() {
            return (0.0, 0.0, 0.0);
        }

        let gt_set: HashSet<_> = ground_truth.iter().collect();
        let _pred_set: HashSet<_> = predicted.iter().collect();

        // Recall: what fraction of ground truth was found
        let true_positives = predicted.iter().filter(|p| gt_set.contains(p)).count();
        let recall = true_positives as f32 / ground_truth.len() as f32;

        // Precision: what fraction of predictions were correct
        let precision = true_positives as f32 / predicted.len() as f32;

        // Reciprocal Rank: 1 / rank of first correct prediction
        let rr = predicted
            .iter()
            .enumerate()
            .find(|(_, p)| gt_set.contains(p))
            .map(|(rank, _)| 1.0 / (rank + 1) as f32)
            .unwrap_or(0.0);

        (recall, precision, rr)
    }

    /// Validate a single embedding pair.
    pub fn validate_single(
        &self,
        old_embedding: &[f32],
        new_embedding: &[f32],
    ) -> MlResult<bool> {
        // Normalize both
        let _old_arr = normalize_l2(&Array1::from_vec(old_embedding.to_vec()).view());
        let _new_arr = normalize_l2(&Array1::from_vec(new_embedding.to_vec()).view());

        // Check that new embedding is valid
        let norm: f32 = new_embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if (norm - 1.0).abs() > 0.1 {
            return Ok(false);
        }

        // Check that new embedding has reasonable values
        if new_embedding.iter().any(|x| x.is_nan() || x.is_infinite()) {
            return Ok(false);
        }

        Ok(true)
    }

    /// Get the configuration.
    pub fn config(&self) -> &QualityConfig {
        &self.config
    }
}

/// Validate embeddings function (convenience wrapper).
pub fn validate_embeddings(
    samples: &[ValidationSample],
    config: QualityConfig,
) -> MlResult<QualityResult> {
    let gate = QualityGate::new(config);
    gate.validate(samples)
}

/// Quick validation that checks only basic properties.
pub fn validate_embedding_basic(embedding: &[f32], expected_dim: usize) -> MlResult<()> {
    // Check dimension
    if embedding.len() != expected_dim {
        return Err(MlError::DimensionMismatch {
            expected: expected_dim,
            actual: embedding.len(),
        });
    }

    // Check for NaN/Inf
    if embedding.iter().any(|x| x.is_nan() || x.is_infinite()) {
        return Err(MlError::Migration(
            "Embedding contains NaN or infinite values".to_string(),
        ));
    }

    // Check normalization (should be close to 1.0)
    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    if (norm - 1.0).abs() > 0.01 {
        return Err(MlError::Migration(format!(
            "Embedding is not normalized: norm = {}",
            norm
        )));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_sample(id: &str, dim_old: usize, dim_new: usize) -> ValidationSample {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        let mut old_embedding: Vec<f32> = (0..dim_old).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let mut new_embedding: Vec<f32> = (0..dim_new).map(|_| rng.gen_range(-1.0..1.0)).collect();

        // Normalize
        let old_norm: f32 = old_embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        old_embedding.iter_mut().for_each(|x| *x /= old_norm);

        let new_norm: f32 = new_embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        new_embedding.iter_mut().for_each(|x| *x /= new_norm);

        ValidationSample::new(id, format!("Text for {}", id), old_embedding, new_embedding)
    }

    #[test]
    fn test_quality_config_default() {
        let config = QualityConfig::default();
        assert_eq!(config.min_recall_at_k, 0.85);
        assert_eq!(config.k, 10);
        assert_eq!(config.min_precision, 0.80);
    }

    #[test]
    fn test_quality_config_strict() {
        let config = QualityConfig::strict();
        assert_eq!(config.min_recall_at_k, 0.95);
        assert!(config.fail_fast);
    }

    #[test]
    fn test_validate_empty_samples() {
        let gate = QualityGate::new(QualityConfig::default());
        let result = gate.validate(&[]).unwrap();

        assert!(result.passed);
        assert_eq!(result.metrics.samples_validated, 0);
    }

    #[test]
    fn test_validate_basic_valid() {
        let mut embedding = vec![0.1f32; 128];
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        embedding.iter_mut().for_each(|x| *x /= norm);

        assert!(validate_embedding_basic(&embedding, 128).is_ok());
    }

    #[test]
    fn test_validate_basic_wrong_dim() {
        let embedding = vec![0.1f32; 64];
        let result = validate_embedding_basic(&embedding, 128);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_basic_nan() {
        let embedding = vec![f32::NAN; 128];
        let result = validate_embedding_basic(&embedding, 128);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_basic_not_normalized() {
        let embedding = vec![1.0f32; 128]; // Norm = sqrt(128) != 1.0
        let result = validate_embedding_basic(&embedding, 128);
        assert!(result.is_err());
    }

    #[test]
    fn test_quality_metrics_f1() {
        let metrics = QualityMetrics {
            recall_at_k: 0.9,
            k: 10,
            precision: 0.8,
            mrr: 0.75,
            samples_validated: 100,
            samples_failed: 5,
            duration: Duration::from_secs(1),
        };

        let f1 = metrics.f1_score();
        assert!(f1 > 0.84 && f1 < 0.85); // Approximately 0.847
    }

    #[test]
    fn test_validation_sample_new() {
        let sample = ValidationSample::new("id1", "test text", vec![0.5; 384], vec![0.5; 128]);

        assert_eq!(sample.id, "id1");
        assert_eq!(sample.text, "test text");
        assert_eq!(sample.old_embedding.len(), 384);
        assert_eq!(sample.new_embedding.len(), 128);
        assert!(sample.ground_truth_neighbors.is_none());
    }

    #[test]
    fn test_validation_sample_with_ground_truth() {
        let sample = ValidationSample::new("id1", "test", vec![0.5; 384], vec![0.5; 128])
            .with_ground_truth(vec!["id2".to_string(), "id3".to_string()]);

        assert!(sample.ground_truth_neighbors.is_some());
        assert_eq!(sample.ground_truth_neighbors.unwrap().len(), 2);
    }

    #[test]
    fn test_quality_gate_validate_single() {
        let gate = QualityGate::new(QualityConfig::default());

        // Valid normalized embedding
        let mut old = vec![0.1f32; 384];
        let old_norm: f32 = old.iter().map(|x| x * x).sum::<f32>().sqrt();
        old.iter_mut().for_each(|x| *x /= old_norm);

        let mut new = vec![0.1f32; 128];
        let new_norm: f32 = new.iter().map(|x| x * x).sum::<f32>().sqrt();
        new.iter_mut().for_each(|x| *x /= new_norm);

        assert!(gate.validate_single(&old, &new).unwrap());
    }
}
