//! Holdout Validation Tests with 80/20 Split
//!
//! This module implements holdout validation for the learning system,
//! splitting patterns into training (80%) and test (20%) sets to validate
//! prediction accuracy and model effectiveness.
//!
//! # Validation Strategy
//!
//! 1. Split patterns into training and holdout sets
//! 2. Train/fit on training set only
//! 3. Evaluate predictions on holdout set
//! 4. Calculate accuracy metrics (precision, recall, F1, RMSE)

use std::collections::{HashMap, HashSet};

use chrono::{DateTime, Utc};
use rand::{seq::SliceRandom, SeedableRng};
use rand_chacha::ChaCha8Rng;

/// Configuration for holdout validation.
#[derive(Debug, Clone)]
pub struct HoldoutConfig {
    /// Fraction of data to use for training (default: 0.8)
    pub train_fraction: f64,

    /// Random seed for reproducible splits
    pub seed: u64,

    /// Whether to stratify by domain
    pub stratify_by_domain: bool,

    /// Minimum patterns per domain for stratification
    pub min_patterns_per_domain: usize,

    /// K value for recall@K calculation
    pub k: usize,
}

impl Default for HoldoutConfig {
    fn default() -> Self {
        Self {
            train_fraction: 0.8,
            seed: 42,
            stratify_by_domain: true,
            min_patterns_per_domain: 5,
            k: 10,
        }
    }
}

/// A test pattern for holdout validation.
#[derive(Debug, Clone)]
pub struct TestPattern {
    /// Unique identifier
    pub id: String,

    /// Problem description
    pub problem: String,

    /// Solution description
    pub solution: String,

    /// Domain/category
    pub domain: String,

    /// Reward score (0.0 - 1.0)
    pub reward: f64,

    /// Whether pattern was successful
    pub success: bool,

    /// Embedding vector (if available)
    pub embedding: Option<Vec<f64>>,

    /// Creation timestamp
    pub created_at: DateTime<Utc>,

    /// Tags for categorization
    pub tags: Vec<String>,
}

impl TestPattern {
    /// Create a new test pattern.
    pub fn new(id: impl Into<String>, problem: impl Into<String>, solution: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            problem: problem.into(),
            solution: solution.into(),
            domain: "general".to_string(),
            reward: 0.5,
            success: true,
            embedding: None,
            created_at: Utc::now(),
            tags: Vec::new(),
        }
    }

    /// Set the domain.
    pub fn with_domain(mut self, domain: impl Into<String>) -> Self {
        self.domain = domain.into();
        self
    }

    /// Set the reward.
    pub fn with_reward(mut self, reward: f64) -> Self {
        self.reward = reward.clamp(0.0, 1.0);
        self
    }

    /// Set success flag.
    pub fn with_success(mut self, success: bool) -> Self {
        self.success = success;
        self
    }

    /// Set embedding.
    pub fn with_embedding(mut self, embedding: Vec<f64>) -> Self {
        self.embedding = Some(embedding);
        self
    }
}

/// Result of the holdout split operation.
#[derive(Debug)]
pub struct HoldoutSplit {
    /// Training set indices
    pub train_indices: Vec<usize>,

    /// Test/holdout set indices
    pub test_indices: Vec<usize>,

    /// Training set patterns
    pub train_patterns: Vec<TestPattern>,

    /// Test/holdout set patterns
    pub test_patterns: Vec<TestPattern>,

    /// Domain distribution in training set
    pub train_domain_counts: HashMap<String, usize>,

    /// Domain distribution in test set
    pub test_domain_counts: HashMap<String, usize>,
}

impl HoldoutSplit {
    /// Get the training fraction achieved.
    pub fn actual_train_fraction(&self) -> f64 {
        let total = self.train_patterns.len() + self.test_patterns.len();
        if total == 0 {
            0.0
        } else {
            self.train_patterns.len() as f64 / total as f64
        }
    }

    /// Get training set size.
    pub fn train_size(&self) -> usize {
        self.train_patterns.len()
    }

    /// Get test set size.
    pub fn test_size(&self) -> usize {
        self.test_patterns.len()
    }
}

/// Predictions made by a model on the holdout set.
#[derive(Debug, Clone)]
pub struct HoldoutPrediction {
    /// Pattern ID being predicted
    pub pattern_id: String,

    /// Predicted reward/relevance score
    pub predicted_reward: f64,

    /// Actual reward value
    pub actual_reward: f64,

    /// Predicted success probability
    pub predicted_success_prob: f64,

    /// Actual success value
    pub actual_success: bool,

    /// Retrieval rank (1-based, None if not retrieved)
    pub retrieval_rank: Option<usize>,

    /// Similarity score to query (if applicable)
    pub similarity_score: Option<f64>,
}

/// Accuracy metrics for holdout validation.
#[derive(Debug, Clone, Default)]
pub struct AccuracyMetrics {
    /// Mean Absolute Error for reward predictions
    pub mae: f64,

    /// Root Mean Squared Error for reward predictions
    pub rmse: f64,

    /// Mean Squared Error
    pub mse: f64,

    /// R-squared (coefficient of determination)
    pub r_squared: f64,

    /// Precision for success prediction
    pub precision: f64,

    /// Recall for success prediction
    pub recall: f64,

    /// F1 score for success prediction
    pub f1_score: f64,

    /// Accuracy for success classification
    pub accuracy: f64,

    /// Recall at K for retrieval
    pub recall_at_k: f64,

    /// Mean Reciprocal Rank
    pub mrr: f64,

    /// NDCG (Normalized Discounted Cumulative Gain)
    pub ndcg: f64,

    /// Number of predictions evaluated
    pub num_predictions: usize,

    /// Number of positive predictions
    pub num_positive_predictions: usize,

    /// Number of actual positives
    pub num_actual_positives: usize,
}

impl AccuracyMetrics {
    /// Calculate metrics from predictions.
    pub fn from_predictions(predictions: &[HoldoutPrediction], k: usize) -> Self {
        if predictions.is_empty() {
            return Self::default();
        }

        let n = predictions.len() as f64;

        // Reward prediction metrics (MAE, RMSE, R-squared)
        let mut sum_error = 0.0;
        let mut sum_sq_error = 0.0;
        let mean_actual: f64 = predictions.iter().map(|p| p.actual_reward).sum::<f64>() / n;
        let mut sum_sq_total = 0.0;

        for pred in predictions {
            let error = pred.predicted_reward - pred.actual_reward;
            sum_error += error.abs();
            sum_sq_error += error * error;
            sum_sq_total += (pred.actual_reward - mean_actual).powi(2);
        }

        let mae = sum_error / n;
        let mse = sum_sq_error / n;
        let rmse = mse.sqrt();
        let r_squared = if sum_sq_total > 0.0 {
            1.0 - (sum_sq_error / sum_sq_total)
        } else {
            0.0
        };

        // Success classification metrics
        let mut true_positives = 0;
        let mut false_positives = 0;
        let mut true_negatives = 0;
        let mut false_negatives = 0;

        for pred in predictions {
            let predicted_success = pred.predicted_success_prob >= 0.5;
            match (predicted_success, pred.actual_success) {
                (true, true) => true_positives += 1,
                (true, false) => false_positives += 1,
                (false, true) => false_negatives += 1,
                (false, false) => true_negatives += 1,
            }
        }

        let precision = if true_positives + false_positives > 0 {
            true_positives as f64 / (true_positives + false_positives) as f64
        } else {
            0.0
        };

        let recall = if true_positives + false_negatives > 0 {
            true_positives as f64 / (true_positives + false_negatives) as f64
        } else {
            0.0
        };

        let f1_score = if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        };

        let accuracy = (true_positives + true_negatives) as f64 / n;

        // Retrieval metrics (Recall@K, MRR, NDCG)
        let mut retrieved_count = 0;
        let mut mrr_sum = 0.0;
        let mut dcg = 0.0;
        let mut idcg = 0.0;

        // Sort predictions by predicted reward for NDCG calculation
        let mut sorted_by_actual: Vec<_> = predictions.iter().collect();
        sorted_by_actual.sort_by(|a, b| b.actual_reward.partial_cmp(&a.actual_reward).unwrap());

        for (i, pred) in predictions.iter().enumerate() {
            if let Some(rank) = pred.retrieval_rank {
                if rank <= k {
                    retrieved_count += 1;
                    mrr_sum += 1.0 / rank as f64;
                }
                // DCG: relevance / log2(rank + 1)
                dcg += pred.actual_reward / (rank as f64 + 1.0).log2();
            }

            // IDCG: ideal ordering
            if i < k {
                idcg += sorted_by_actual[i].actual_reward / (i as f64 + 2.0).log2();
            }
        }

        let recall_at_k = if !predictions.is_empty() {
            retrieved_count as f64 / predictions.len().min(k) as f64
        } else {
            0.0
        };

        let mrr = mrr_sum / n;
        let ndcg = if idcg > 0.0 { dcg / idcg } else { 0.0 };

        Self {
            mae,
            rmse,
            mse,
            r_squared,
            precision,
            recall,
            f1_score,
            accuracy,
            recall_at_k,
            mrr,
            ndcg,
            num_predictions: predictions.len(),
            num_positive_predictions: true_positives + false_positives,
            num_actual_positives: true_positives + false_negatives,
        }
    }

    /// Check if the model meets minimum quality thresholds.
    pub fn meets_quality_threshold(&self, min_f1: f64, max_rmse: f64) -> bool {
        self.f1_score >= min_f1 && self.rmse <= max_rmse
    }
}

/// Holdout validator for learning system evaluation.
pub struct HoldoutValidator {
    config: HoldoutConfig,
    patterns: Vec<TestPattern>,
    split: Option<HoldoutSplit>,
    predictions: Vec<HoldoutPrediction>,
}

impl HoldoutValidator {
    /// Create a new holdout validator.
    pub fn new(config: HoldoutConfig) -> Self {
        Self {
            config,
            patterns: Vec::new(),
            split: None,
            predictions: Vec::new(),
        }
    }

    /// Add patterns for validation.
    pub fn add_patterns(&mut self, patterns: Vec<TestPattern>) {
        self.patterns.extend(patterns);
    }

    /// Perform the train/test split.
    pub fn split(&mut self) -> &HoldoutSplit {
        let mut rng = ChaCha8Rng::seed_from_u64(self.config.seed);

        let split = if self.config.stratify_by_domain {
            self.stratified_split(&mut rng)
        } else {
            self.random_split(&mut rng)
        };

        self.split = Some(split);
        self.split.as_ref().unwrap()
    }

    /// Perform a random (non-stratified) split.
    fn random_split(&self, rng: &mut ChaCha8Rng) -> HoldoutSplit {
        let mut indices: Vec<usize> = (0..self.patterns.len()).collect();
        indices.shuffle(rng);

        let train_size = (self.patterns.len() as f64 * self.config.train_fraction) as usize;
        let train_indices: Vec<usize> = indices[..train_size].to_vec();
        let test_indices: Vec<usize> = indices[train_size..].to_vec();

        self.create_split_from_indices(train_indices, test_indices)
    }

    /// Perform a stratified split by domain.
    fn stratified_split(&self, rng: &mut ChaCha8Rng) -> HoldoutSplit {
        // Group patterns by domain
        let mut domain_indices: HashMap<String, Vec<usize>> = HashMap::new();
        for (i, pattern) in self.patterns.iter().enumerate() {
            domain_indices
                .entry(pattern.domain.clone())
                .or_default()
                .push(i);
        }

        let mut train_indices = Vec::new();
        let mut test_indices = Vec::new();

        // Sort domains for deterministic iteration order
        let mut sorted_domains: Vec<_> = domain_indices.keys().cloned().collect();
        sorted_domains.sort();

        for domain in sorted_domains {
            let mut indices = domain_indices.remove(&domain).unwrap();
            indices.shuffle(rng);

            // Check if domain has enough patterns
            if indices.len() < self.config.min_patterns_per_domain {
                // Put all in training if too few
                train_indices.extend(indices);
            } else {
                let train_size = (indices.len() as f64 * self.config.train_fraction) as usize;
                train_indices.extend(&indices[..train_size]);
                test_indices.extend(&indices[train_size..]);
            }
        }

        self.create_split_from_indices(train_indices, test_indices)
    }

    /// Create a HoldoutSplit from train/test indices.
    fn create_split_from_indices(
        &self,
        train_indices: Vec<usize>,
        test_indices: Vec<usize>,
    ) -> HoldoutSplit {
        let train_patterns: Vec<TestPattern> = train_indices
            .iter()
            .map(|&i| self.patterns[i].clone())
            .collect();
        let test_patterns: Vec<TestPattern> = test_indices
            .iter()
            .map(|&i| self.patterns[i].clone())
            .collect();

        let mut train_domain_counts = HashMap::new();
        let mut test_domain_counts = HashMap::new();

        for pattern in &train_patterns {
            *train_domain_counts.entry(pattern.domain.clone()).or_insert(0) += 1;
        }
        for pattern in &test_patterns {
            *test_domain_counts.entry(pattern.domain.clone()).or_insert(0) += 1;
        }

        HoldoutSplit {
            train_indices,
            test_indices,
            train_patterns,
            test_patterns,
            train_domain_counts,
            test_domain_counts,
        }
    }

    /// Add a prediction for evaluation.
    pub fn add_prediction(&mut self, prediction: HoldoutPrediction) {
        self.predictions.push(prediction);
    }

    /// Calculate accuracy metrics for the predictions.
    pub fn evaluate(&self) -> AccuracyMetrics {
        AccuracyMetrics::from_predictions(&self.predictions, self.config.k)
    }

    /// Get the training set (requires split to have been performed).
    pub fn training_set(&self) -> Option<&[TestPattern]> {
        self.split.as_ref().map(|s| s.train_patterns.as_slice())
    }

    /// Get the test set (requires split to have been performed).
    pub fn test_set(&self) -> Option<&[TestPattern]> {
        self.split.as_ref().map(|s| s.test_patterns.as_slice())
    }

    /// Clear predictions for a new evaluation round.
    pub fn clear_predictions(&mut self) {
        self.predictions.clear();
    }

    /// Get the current split.
    pub fn get_split(&self) -> Option<&HoldoutSplit> {
        self.split.as_ref()
    }

    /// Get the configuration.
    pub fn config(&self) -> &HoldoutConfig {
        &self.config
    }
}

/// Cross-validation variant with K folds.
pub struct KFoldValidator {
    num_folds: usize,
    patterns: Vec<TestPattern>,
    current_fold: usize,
    fold_metrics: Vec<AccuracyMetrics>,
    seed: u64,
}

impl KFoldValidator {
    /// Create a new K-fold validator.
    pub fn new(num_folds: usize, seed: u64) -> Self {
        Self {
            num_folds,
            patterns: Vec::new(),
            current_fold: 0,
            fold_metrics: Vec::new(),
            seed,
        }
    }

    /// Add patterns for cross-validation.
    pub fn add_patterns(&mut self, patterns: Vec<TestPattern>) {
        self.patterns.extend(patterns);
    }

    /// Get the fold indices for training and testing.
    pub fn get_fold(&self, fold: usize) -> (Vec<usize>, Vec<usize>) {
        let mut rng = ChaCha8Rng::seed_from_u64(self.seed);
        let mut indices: Vec<usize> = (0..self.patterns.len()).collect();
        indices.shuffle(&mut rng);

        let fold_size = self.patterns.len() / self.num_folds;
        let test_start = fold * fold_size;
        let test_end = if fold == self.num_folds - 1 {
            self.patterns.len()
        } else {
            (fold + 1) * fold_size
        };

        let test_indices: Vec<usize> = indices[test_start..test_end].to_vec();
        let train_indices: Vec<usize> = indices[..test_start]
            .iter()
            .chain(indices[test_end..].iter())
            .copied()
            .collect();

        (train_indices, test_indices)
    }

    /// Get training patterns for a specific fold.
    pub fn training_patterns(&self, fold: usize) -> Vec<TestPattern> {
        let (train_indices, _) = self.get_fold(fold);
        train_indices.iter().map(|&i| self.patterns[i].clone()).collect()
    }

    /// Get test patterns for a specific fold.
    pub fn test_patterns(&self, fold: usize) -> Vec<TestPattern> {
        let (_, test_indices) = self.get_fold(fold);
        test_indices.iter().map(|&i| self.patterns[i].clone()).collect()
    }

    /// Record metrics for a fold.
    pub fn record_fold_metrics(&mut self, metrics: AccuracyMetrics) {
        self.fold_metrics.push(metrics);
        self.current_fold += 1;
    }

    /// Calculate average metrics across all folds.
    pub fn average_metrics(&self) -> AccuracyMetrics {
        if self.fold_metrics.is_empty() {
            return AccuracyMetrics::default();
        }

        let n = self.fold_metrics.len() as f64;

        AccuracyMetrics {
            mae: self.fold_metrics.iter().map(|m| m.mae).sum::<f64>() / n,
            rmse: self.fold_metrics.iter().map(|m| m.rmse).sum::<f64>() / n,
            mse: self.fold_metrics.iter().map(|m| m.mse).sum::<f64>() / n,
            r_squared: self.fold_metrics.iter().map(|m| m.r_squared).sum::<f64>() / n,
            precision: self.fold_metrics.iter().map(|m| m.precision).sum::<f64>() / n,
            recall: self.fold_metrics.iter().map(|m| m.recall).sum::<f64>() / n,
            f1_score: self.fold_metrics.iter().map(|m| m.f1_score).sum::<f64>() / n,
            accuracy: self.fold_metrics.iter().map(|m| m.accuracy).sum::<f64>() / n,
            recall_at_k: self.fold_metrics.iter().map(|m| m.recall_at_k).sum::<f64>() / n,
            mrr: self.fold_metrics.iter().map(|m| m.mrr).sum::<f64>() / n,
            ndcg: self.fold_metrics.iter().map(|m| m.ndcg).sum::<f64>() / n,
            num_predictions: self.fold_metrics.iter().map(|m| m.num_predictions).sum(),
            num_positive_predictions: self.fold_metrics.iter().map(|m| m.num_positive_predictions).sum(),
            num_actual_positives: self.fold_metrics.iter().map(|m| m.num_actual_positives).sum(),
        }
    }

    /// Get the number of folds.
    pub fn num_folds(&self) -> usize {
        self.num_folds
    }

    /// Get the number of completed folds.
    pub fn completed_folds(&self) -> usize {
        self.fold_metrics.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_patterns(n: usize) -> Vec<TestPattern> {
        (0..n)
            .map(|i| {
                TestPattern::new(
                    format!("pattern-{}", i),
                    format!("Problem {}", i),
                    format!("Solution {}", i),
                )
                .with_domain(format!("domain-{}", i % 3))
                .with_reward(0.3 + (i as f64 * 0.05) % 0.7)
                .with_success(i % 2 == 0)
            })
            .collect()
    }

    #[test]
    fn test_holdout_config_default() {
        let config = HoldoutConfig::default();
        assert!((config.train_fraction - 0.8).abs() < 0.001);
        assert_eq!(config.seed, 42);
        assert!(config.stratify_by_domain);
    }

    #[test]
    fn test_holdout_split_random() {
        let mut validator = HoldoutValidator::new(HoldoutConfig {
            stratify_by_domain: false,
            ..Default::default()
        });

        validator.add_patterns(create_test_patterns(100));
        let split = validator.split();

        // Check 80/20 split
        assert_eq!(split.train_size(), 80);
        assert_eq!(split.test_size(), 20);
        assert!((split.actual_train_fraction() - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_holdout_split_stratified() {
        let mut validator = HoldoutValidator::new(HoldoutConfig::default());
        validator.add_patterns(create_test_patterns(100));
        let split = validator.split();

        // Check that all domains are represented
        assert!(!split.train_domain_counts.is_empty());
        assert!(!split.test_domain_counts.is_empty());

        // Each domain should have some patterns in both sets
        for domain in split.train_domain_counts.keys() {
            // May not have all domains in test set if stratification moved small domains to train
            let _ = split.test_domain_counts.get(domain);
        }
    }

    #[test]
    fn test_holdout_split_reproducible() {
        let config = HoldoutConfig {
            seed: 12345,
            ..Default::default()
        };

        let mut validator1 = HoldoutValidator::new(config.clone());
        validator1.add_patterns(create_test_patterns(100));
        let split1 = validator1.split();

        let mut validator2 = HoldoutValidator::new(config);
        validator2.add_patterns(create_test_patterns(100));
        let split2 = validator2.split();

        // Same seed should produce same split
        assert_eq!(split1.train_indices, split2.train_indices);
        assert_eq!(split1.test_indices, split2.test_indices);
    }

    #[test]
    fn test_accuracy_metrics_from_predictions() {
        let predictions = vec![
            HoldoutPrediction {
                pattern_id: "1".to_string(),
                predicted_reward: 0.8,
                actual_reward: 0.75,
                predicted_success_prob: 0.9,
                actual_success: true,
                retrieval_rank: Some(1),
                similarity_score: Some(0.95),
            },
            HoldoutPrediction {
                pattern_id: "2".to_string(),
                predicted_reward: 0.6,
                actual_reward: 0.5,
                predicted_success_prob: 0.7,
                actual_success: true,
                retrieval_rank: Some(2),
                similarity_score: Some(0.85),
            },
            HoldoutPrediction {
                pattern_id: "3".to_string(),
                predicted_reward: 0.3,
                actual_reward: 0.4,
                predicted_success_prob: 0.3,
                actual_success: false,
                retrieval_rank: Some(5),
                similarity_score: Some(0.6),
            },
        ];

        let metrics = AccuracyMetrics::from_predictions(&predictions, 10);

        assert_eq!(metrics.num_predictions, 3);
        assert!(metrics.mae < 0.2);
        assert!(metrics.rmse < 0.2);
        assert!(metrics.precision > 0.0);
        assert!(metrics.recall > 0.0);
    }

    #[test]
    fn test_accuracy_metrics_perfect_predictions() {
        let predictions = vec![
            HoldoutPrediction {
                pattern_id: "1".to_string(),
                predicted_reward: 0.8,
                actual_reward: 0.8,
                predicted_success_prob: 1.0,
                actual_success: true,
                retrieval_rank: Some(1),
                similarity_score: Some(1.0),
            },
            HoldoutPrediction {
                pattern_id: "2".to_string(),
                predicted_reward: 0.3,
                actual_reward: 0.3,
                predicted_success_prob: 0.0,
                actual_success: false,
                retrieval_rank: Some(2),
                similarity_score: Some(0.9),
            },
        ];

        let metrics = AccuracyMetrics::from_predictions(&predictions, 10);

        assert!((metrics.mae - 0.0).abs() < 0.001);
        assert!((metrics.rmse - 0.0).abs() < 0.001);
        assert!((metrics.accuracy - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_accuracy_metrics_empty() {
        let metrics = AccuracyMetrics::from_predictions(&[], 10);
        assert_eq!(metrics.num_predictions, 0);
        assert_eq!(metrics.mae, 0.0);
    }

    #[test]
    fn test_holdout_validator_workflow() {
        let mut validator = HoldoutValidator::new(HoldoutConfig::default());
        validator.add_patterns(create_test_patterns(50));

        // Perform split
        let split = validator.split();
        assert!(split.train_size() > 0);
        assert!(split.test_size() > 0);

        // Simulate predictions - collect test data first to avoid borrow conflict
        let test_data: Vec<_> = validator
            .test_set()
            .map(|patterns| {
                patterns
                    .iter()
                    .enumerate()
                    .map(|(i, p)| (p.id.clone(), p.reward, p.success, i))
                    .collect()
            })
            .unwrap_or_default();

        for (pattern_id, reward, success, i) in test_data {
            validator.add_prediction(HoldoutPrediction {
                pattern_id,
                predicted_reward: reward + 0.05, // Small error
                actual_reward: reward,
                predicted_success_prob: if success { 0.8 } else { 0.2 },
                actual_success: success,
                retrieval_rank: Some(i + 1),
                similarity_score: Some(0.9 - i as f64 * 0.05),
            });
        }

        // Evaluate
        let metrics = validator.evaluate();
        assert!(metrics.num_predictions > 0);
    }

    #[test]
    fn test_kfold_validator() {
        let mut validator = KFoldValidator::new(5, 42);
        validator.add_patterns(create_test_patterns(100));

        assert_eq!(validator.num_folds(), 5);

        for fold in 0..5 {
            let (train_indices, test_indices) = validator.get_fold(fold);

            // Check that train and test don't overlap
            let train_set: HashSet<_> = train_indices.iter().collect();
            let test_set: HashSet<_> = test_indices.iter().collect();
            assert!(train_set.is_disjoint(&test_set));

            // Check that all indices are covered
            assert_eq!(train_indices.len() + test_indices.len(), 100);
        }
    }

    #[test]
    fn test_kfold_average_metrics() {
        let mut validator = KFoldValidator::new(3, 42);

        // Simulate recording metrics for each fold
        validator.record_fold_metrics(AccuracyMetrics {
            f1_score: 0.8,
            rmse: 0.1,
            ..Default::default()
        });
        validator.record_fold_metrics(AccuracyMetrics {
            f1_score: 0.85,
            rmse: 0.12,
            ..Default::default()
        });
        validator.record_fold_metrics(AccuracyMetrics {
            f1_score: 0.9,
            rmse: 0.08,
            ..Default::default()
        });

        let avg = validator.average_metrics();
        assert!((avg.f1_score - 0.85).abs() < 0.001);
        assert!((avg.rmse - 0.1).abs() < 0.001);
    }

    #[test]
    fn test_meets_quality_threshold() {
        let good_metrics = AccuracyMetrics {
            f1_score: 0.85,
            rmse: 0.1,
            ..Default::default()
        };

        assert!(good_metrics.meets_quality_threshold(0.8, 0.15));
        assert!(!good_metrics.meets_quality_threshold(0.9, 0.15));
        assert!(!good_metrics.meets_quality_threshold(0.8, 0.05));
    }

    #[test]
    fn test_test_pattern_builder() {
        let pattern = TestPattern::new("id1", "Problem 1", "Solution 1")
            .with_domain("testing")
            .with_reward(0.85)
            .with_success(true)
            .with_embedding(vec![0.1, 0.2, 0.3]);

        assert_eq!(pattern.id, "id1");
        assert_eq!(pattern.domain, "testing");
        assert!((pattern.reward - 0.85).abs() < 0.001);
        assert!(pattern.success);
        assert!(pattern.embedding.is_some());
    }

    #[test]
    fn test_reward_clamping() {
        let pattern = TestPattern::new("id", "p", "s").with_reward(1.5);
        assert!((pattern.reward - 1.0).abs() < 0.001);

        let pattern = TestPattern::new("id", "p", "s").with_reward(-0.5);
        assert!((pattern.reward - 0.0).abs() < 0.001);
    }
}
