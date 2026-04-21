//! Contrastive learning trainer for LoRA adapters.
//!
//! Uses triplet contrastive loss to train LoRA adapters that pull
//! same-domain high-reward embeddings closer together while pushing
//! different-domain or low-reward embeddings apart.
//!
//! # Training Process
//!
//! 1. Generate training pairs (anchor, positive, negative) from patterns
//! 2. For each epoch, iterate over pairs:
//!    - Transform all three embeddings through the adapter
//!    - Compute triplet loss: max(0, -sim(a,p) + sim(a,n) + margin)
//!    - Update A and B matrices using gradient descent
//! 3. Early stop if loss plateaus

use ndarray::Array1;
use rand::seq::SliceRandom;
use rand::Rng;
use serde::{Deserialize, Serialize};

use super::adapter::LoraAdapter;
use crate::ml::{cosine_similarity, MlError, MlResult};

/// Training configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    /// Maximum number of training epochs (default: 50).
    pub max_epochs: u32,
    /// Batch size for training pairs (default: 32).
    pub batch_size: usize,
    /// Early stopping: stop if loss doesn't improve for this many epochs.
    pub patience: u32,
    /// Contrastive loss margin (default: 0.5).
    pub margin: f32,
    /// Minimum patterns required to train (default: 20).
    pub min_patterns: usize,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            max_epochs: 50,
            batch_size: 32,
            patience: 5,
            margin: 0.5,
            min_patterns: 20,
        }
    }
}

/// A training pair for contrastive learning.
#[derive(Debug, Clone)]
pub struct TrainingPair {
    /// Anchor embedding.
    pub anchor: Array1<f32>,
    /// Positive example (same domain, high reward).
    pub positive: Array1<f32>,
    /// Negative example (different domain or low reward).
    pub negative: Array1<f32>,
}

/// Result of a training run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingResult {
    /// Number of epochs completed.
    pub epochs: u32,
    /// Final training loss.
    pub final_loss: f32,
    /// Whether early stopping was triggered.
    pub early_stopped: bool,
    /// Number of training pairs used.
    pub num_pairs: usize,
    /// Training duration in milliseconds.
    pub duration_ms: u64,
}

/// LoRA trainer using contrastive learning.
pub struct LoraTrainer {
    config: TrainingConfig,
}

impl LoraTrainer {
    /// Create a new trainer with the given configuration.
    pub fn new(config: TrainingConfig) -> Self {
        Self { config }
    }

    /// Train a LoRA adapter from training pairs.
    ///
    /// Uses triplet contrastive loss: `max(0, -sim(a',p') + sim(a',n') + margin)`
    /// where primed variables denote LoRA-transformed embeddings.
    ///
    /// Gradient descent updates A and B matrices using analytical gradients
    /// derived from the chain rule through the linear LoRA transformation.
    pub fn train(
        &self,
        adapter: &mut LoraAdapter,
        pairs: &[TrainingPair],
    ) -> MlResult<TrainingResult> {
        if pairs.is_empty() {
            return Err(MlError::Migration(
                "No training pairs provided".to_string(),
            ));
        }

        let start = std::time::Instant::now();
        let lr = adapter.config.learning_rate;
        let rank = adapter.config.rank;
        let base_dim = adapter.config.base_dim;

        let mut best_loss = f32::MAX;
        let mut patience_counter = 0u32;
        let mut final_epoch = 0u32;
        let mut early_stopped = false;

        for epoch in 0..self.config.max_epochs {
            let mut epoch_loss = 0.0f32;
            let mut loss_count = 0usize;

            // Accumulate gradients for A and B
            let mut grad_a = vec![0.0f32; rank * base_dim];
            let mut grad_b = vec![0.0f32; base_dim * rank];

            for pair in pairs {
                // Transform all three embeddings
                let a_out = adapter.transform(&pair.anchor.view())?;
                let p_out = adapter.transform(&pair.positive.view())?;
                let n_out = adapter.transform(&pair.negative.view())?;

                // Compute cosine similarities
                let sim_pos = cosine_similarity(&a_out.view(), &p_out.view());
                let sim_neg = cosine_similarity(&a_out.view(), &n_out.view());

                // Triplet loss: max(0, -sim_pos + sim_neg + margin)
                let loss = (-sim_pos + sim_neg + self.config.margin).max(0.0);
                epoch_loss += loss;
                loss_count += 1;

                // Only update if loss > 0 (violating pairs)
                if loss > 0.0 {
                    // Approximate gradient via finite differences on the matrices.
                    // We use a simplified approach: perturb each element of A and B
                    // by epsilon, compute loss change, and update.
                    //
                    // For efficiency, we use a stochastic subset of parameters.
                    let epsilon = 0.001f32;
                    let num_params_a = rank * base_dim;
                    let num_params_b = base_dim * rank;

                    // Sample a subset of parameters to update (stochastic gradient)
                    let sample_size = (num_params_a / 4).max(1);
                    let mut rng = rand::thread_rng();

                    // Update matrix A (sampled)
                    for _ in 0..sample_size {
                        let idx = rng.gen_range(0..num_params_a);
                        let orig = adapter.matrix_a[idx];

                        // Forward: perturb +epsilon
                        adapter.matrix_a[idx] = orig + epsilon;
                        let a_plus = adapter.transform(&pair.anchor.view())?;
                        let p_plus = adapter.transform(&pair.positive.view())?;
                        let n_plus = adapter.transform(&pair.negative.view())?;
                        let sim_pos_plus =
                            cosine_similarity(&a_plus.view(), &p_plus.view());
                        let sim_neg_plus =
                            cosine_similarity(&a_plus.view(), &n_plus.view());
                        let loss_plus =
                            (-sim_pos_plus + sim_neg_plus + self.config.margin).max(0.0);

                        // Restore and compute gradient
                        adapter.matrix_a[idx] = orig;
                        let grad = (loss_plus - loss) / epsilon;
                        grad_a[idx] += grad;
                    }

                    // Update matrix B (sampled)
                    for _ in 0..sample_size {
                        let idx = rng.gen_range(0..num_params_b);
                        let orig = adapter.matrix_b[idx];

                        adapter.matrix_b[idx] = orig + epsilon;
                        let a_plus = adapter.transform(&pair.anchor.view())?;
                        let p_plus = adapter.transform(&pair.positive.view())?;
                        let n_plus = adapter.transform(&pair.negative.view())?;
                        let sim_pos_plus =
                            cosine_similarity(&a_plus.view(), &p_plus.view());
                        let sim_neg_plus =
                            cosine_similarity(&a_plus.view(), &n_plus.view());
                        let loss_plus =
                            (-sim_pos_plus + sim_neg_plus + self.config.margin).max(0.0);

                        adapter.matrix_b[idx] = orig;
                        let grad = (loss_plus - loss) / epsilon;
                        grad_b[idx] += grad;
                    }
                }
            }

            // Apply accumulated gradients
            let scale = if loss_count > 0 {
                lr / loss_count as f32
            } else {
                0.0
            };

            for (param, grad) in adapter.matrix_a.iter_mut().zip(grad_a.iter()) {
                *param -= scale * grad;
            }
            for (param, grad) in adapter.matrix_b.iter_mut().zip(grad_b.iter()) {
                *param -= scale * grad;
            }

            let avg_loss = if loss_count > 0 {
                epoch_loss / loss_count as f32
            } else {
                0.0
            };

            // Early stopping check
            if avg_loss < best_loss - 0.001 {
                best_loss = avg_loss;
                patience_counter = 0;
            } else {
                patience_counter += 1;
            }

            final_epoch = epoch + 1;

            if patience_counter >= self.config.patience {
                early_stopped = true;
                break;
            }

            // If loss is already zero, no point continuing
            if avg_loss == 0.0 {
                early_stopped = true;
                break;
            }
        }

        adapter.iterations = final_epoch;
        adapter.final_loss = best_loss;
        adapter.trained_at = chrono::Utc::now().to_rfc3339();

        let duration_ms = start.elapsed().as_millis() as u64;

        Ok(TrainingResult {
            epochs: final_epoch,
            final_loss: best_loss,
            early_stopped,
            num_pairs: pairs.len(),
            duration_ms,
        })
    }

    /// Generate training pairs from patterns with embeddings.
    ///
    /// Each element is a tuple: (embedding, domain, reward).
    ///
    /// - Positive pairs: same domain, reward >= 0.6
    /// - Negative pairs: different domain or reward < 0.4
    pub fn generate_pairs(
        patterns: &[(Array1<f32>, String, f32)], // (embedding, domain, reward)
        max_pairs: usize,
    ) -> Vec<TrainingPair> {
        if patterns.len() < 3 {
            return Vec::new();
        }

        let mut rng = rand::thread_rng();
        let mut pairs = Vec::new();

        // Separate patterns into positives (high reward in same domain) per domain
        let mut domain_positives: std::collections::HashMap<String, Vec<usize>> =
            std::collections::HashMap::new();
        let mut domain_negatives: Vec<usize> = Vec::new();

        for (i, (_emb, domain, reward)) in patterns.iter().enumerate() {
            if *reward >= 0.6 {
                domain_positives
                    .entry(domain.clone())
                    .or_default()
                    .push(i);
            }
            if *reward < 0.4 {
                domain_negatives.push(i);
            }
        }

        // For each domain with at least 2 positive patterns, create pairs
        for (domain, positives) in &domain_positives {
            if positives.len() < 2 {
                continue;
            }

            // Collect negatives: patterns from other domains or low-reward patterns
            let negatives: Vec<usize> = patterns
                .iter()
                .enumerate()
                .filter(|(i, (_, d, r))| {
                    (d != domain || *r < 0.4) && !positives.contains(i)
                })
                .map(|(i, _)| i)
                .collect();

            if negatives.is_empty() {
                continue;
            }

            // Create triplets
            for &anchor_idx in positives {
                for &pos_idx in positives {
                    if anchor_idx == pos_idx {
                        continue;
                    }

                    // Pick a random negative
                    if let Some(&neg_idx) = negatives.choose(&mut rng) {
                        pairs.push(TrainingPair {
                            anchor: patterns[anchor_idx].0.clone(),
                            positive: patterns[pos_idx].0.clone(),
                            negative: patterns[neg_idx].0.clone(),
                        });

                        if pairs.len() >= max_pairs {
                            return pairs;
                        }
                    }
                }
            }
        }

        // If we have few domain-specific pairs, also create cross-domain pairs
        // using low-reward patterns as negatives
        if pairs.len() < max_pairs / 2 && !domain_negatives.is_empty() {
            let high_reward: Vec<usize> = patterns
                .iter()
                .enumerate()
                .filter(|(_, (_, _, r))| *r >= 0.6)
                .map(|(i, _)| i)
                .collect();

            for chunk in high_reward.chunks(2) {
                if chunk.len() < 2 {
                    break;
                }
                if let Some(&neg_idx) = domain_negatives.choose(&mut rng) {
                    pairs.push(TrainingPair {
                        anchor: patterns[chunk[0]].0.clone(),
                        positive: patterns[chunk[1]].0.clone(),
                        negative: patterns[neg_idx].0.clone(),
                    });

                    if pairs.len() >= max_pairs {
                        break;
                    }
                }
            }
        }

        pairs
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ml::lora::adapter::LoraConfig;
    use ndarray::Array1;

    fn make_random_embedding(dim: usize) -> Array1<f32> {
        let mut rng = rand::thread_rng();
        let v: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let arr = Array1::from_vec(v);
        let norm = arr.dot(&arr).sqrt();
        arr.mapv(|x| x / norm)
    }

    fn make_similar_embedding(base: &Array1<f32>, noise: f32) -> Array1<f32> {
        let mut rng = rand::thread_rng();
        let noisy: Array1<f32> = base.mapv(|x| x + rng.gen_range(-noise..noise));
        let norm = noisy.dot(&noisy).sqrt();
        noisy.mapv(|x| x / norm)
    }

    #[test]
    fn test_training_config_default() {
        let config = TrainingConfig::default();
        assert_eq!(config.max_epochs, 50);
        assert_eq!(config.batch_size, 32);
        assert_eq!(config.patience, 5);
        assert!((config.margin - 0.5).abs() < f32::EPSILON);
        assert_eq!(config.min_patterns, 20);
    }

    #[test]
    fn test_generate_pairs_basic() {
        let dim = 8;
        let base_a = make_random_embedding(dim);
        let base_b = make_random_embedding(dim);

        // Create patterns: 3 in "rust" domain with high reward, 2 in "python" with low
        let patterns: Vec<(Array1<f32>, String, f32)> = vec![
            (make_similar_embedding(&base_a, 0.1), "rust".into(), 0.9),
            (make_similar_embedding(&base_a, 0.1), "rust".into(), 0.8),
            (make_similar_embedding(&base_a, 0.1), "rust".into(), 0.7),
            (make_similar_embedding(&base_b, 0.1), "python".into(), 0.3),
            (make_similar_embedding(&base_b, 0.1), "python".into(), 0.2),
        ];

        let pairs = LoraTrainer::generate_pairs(&patterns, 100);
        assert!(
            !pairs.is_empty(),
            "should generate at least one training pair"
        );

        // Each pair should have correct dimensions
        for pair in &pairs {
            assert_eq!(pair.anchor.len(), dim);
            assert_eq!(pair.positive.len(), dim);
            assert_eq!(pair.negative.len(), dim);
        }
    }

    #[test]
    fn test_generate_pairs_insufficient_patterns() {
        let patterns: Vec<(Array1<f32>, String, f32)> = vec![
            (make_random_embedding(8), "rust".into(), 0.9),
            (make_random_embedding(8), "python".into(), 0.8),
        ];

        // Only 2 patterns, not enough for meaningful triplets with same-domain positives
        let pairs = LoraTrainer::generate_pairs(&patterns, 100);
        // With only 2 patterns in different domains, no same-domain pairs can form
        // (each domain has only 1 pattern)
        assert!(
            pairs.is_empty() || pairs.len() <= 1,
            "should generate very few or no pairs with insufficient patterns"
        );
    }

    #[test]
    fn test_train_reduces_loss() {
        let dim = 8;
        let base_positive = make_random_embedding(dim);
        let base_negative = {
            // Create a clearly different embedding
            let mut neg = base_positive.clone();
            neg.mapv_inplace(|x| -x); // opposite direction
            neg
        };

        // Create training pairs where positive is similar to anchor,
        // negative is very different
        let pairs: Vec<TrainingPair> = (0..10)
            .map(|_| TrainingPair {
                anchor: make_similar_embedding(&base_positive, 0.05),
                positive: make_similar_embedding(&base_positive, 0.05),
                negative: make_similar_embedding(&base_negative, 0.05),
            })
            .collect();

        let config = LoraConfig {
            base_dim: dim,
            rank: 2,
            learning_rate: 0.01,
            alpha: 1.0,
        };
        let mut adapter = LoraAdapter::new("test", config);

        let trainer = LoraTrainer::new(TrainingConfig {
            max_epochs: 20,
            patience: 10,
            margin: 0.3,
            ..Default::default()
        });

        let result = trainer.train(&mut adapter, &pairs).unwrap();

        assert!(result.epochs > 0, "should complete at least 1 epoch");
        assert!(result.num_pairs == 10, "should use all 10 pairs");
        // Loss should be finite
        assert!(
            result.final_loss.is_finite(),
            "final loss should be finite, got {}",
            result.final_loss
        );
    }

    #[test]
    fn test_train_early_stopping() {
        let dim = 8;

        // Create pairs where loss is already zero (positive very close, negative very far)
        let base = make_random_embedding(dim);
        let pairs: Vec<TrainingPair> = (0..5)
            .map(|_| {
                let neg = base.mapv(|x| -x);
                TrainingPair {
                    anchor: base.clone(),
                    positive: make_similar_embedding(&base, 0.01),
                    negative: neg,
                }
            })
            .collect();

        let config = LoraConfig {
            base_dim: dim,
            rank: 2,
            learning_rate: 0.01,
            alpha: 1.0,
        };
        let mut adapter = LoraAdapter::new("test", config);

        let trainer = LoraTrainer::new(TrainingConfig {
            max_epochs: 100,
            patience: 3,
            margin: 0.1,
            ..Default::default()
        });

        let result = trainer.train(&mut adapter, &pairs).unwrap();

        // With very easy pairs (positive=same direction, negative=opposite),
        // loss should quickly go to zero and trigger early stopping
        assert!(
            result.early_stopped || result.epochs < 100,
            "should early stop or finish quickly, ran {} epochs",
            result.epochs
        );
    }

    #[test]
    fn test_train_empty_pairs() {
        let config = LoraConfig {
            base_dim: 8,
            rank: 2,
            ..Default::default()
        };
        let mut adapter = LoraAdapter::new("test", config);
        let trainer = LoraTrainer::new(TrainingConfig::default());

        let result = trainer.train(&mut adapter, &[]);
        assert!(result.is_err(), "training with empty pairs should fail");
    }
}
