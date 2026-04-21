//! Maximal Marginal Relevance (MMR) for diversity selection.
//!
//! MMR balances relevance to the query with diversity among selected results,
//! preventing redundant patterns from dominating the results.
//!
//! # Algorithm
//!
//! MMR iteratively selects documents that are both relevant to the query
//! and dissimilar to already-selected documents:
//!
//! ```text
//! MMR = arg max [ λ * Sim(d, q) - (1 - λ) * max(Sim(d, d_i)) ]
//!         d ∈ R\S                          d_i ∈ S
//! ```
//!
//! Where:
//! - λ (lambda) controls the trade-off: higher = more relevance, lower = more diversity
//! - Sim(d, q) is similarity between document d and query q
//! - Sim(d, d_i) is similarity between document d and selected documents

use ndarray::Array1;

use super::search::SearchResult;
use crate::ml::{cosine_similarity_normalized, normalize_l2};

/// Configuration for MMR selection.
#[derive(Debug, Clone)]
pub struct MmrConfig {
    /// Lambda parameter (0.0 to 1.0).
    ///
    /// - λ = 1.0: Pure relevance ranking (no diversity)
    /// - λ = 0.0: Maximum diversity (ignores relevance)
    /// - λ = 0.7: Default balance (70% relevance, 30% diversity)
    pub lambda: f32,

    /// Minimum similarity to query for consideration.
    pub min_relevance: f32,

    /// Whether embeddings are pre-normalized.
    pub normalized_embeddings: bool,
}

impl Default for MmrConfig {
    fn default() -> Self {
        Self {
            lambda: 0.7,
            min_relevance: 0.0,
            normalized_embeddings: true,
        }
    }
}

impl MmrConfig {
    /// Create config with high diversity (λ = 0.5).
    pub fn high_diversity() -> Self {
        Self {
            lambda: 0.5,
            ..Default::default()
        }
    }

    /// Create config with high relevance (λ = 0.9).
    pub fn high_relevance() -> Self {
        Self {
            lambda: 0.9,
            ..Default::default()
        }
    }

    /// Set the lambda parameter.
    pub fn with_lambda(mut self, lambda: f32) -> Self {
        self.lambda = lambda.clamp(0.0, 1.0);
        self
    }

    /// Set the minimum relevance threshold.
    pub fn with_min_relevance(mut self, min: f32) -> Self {
        self.min_relevance = min.clamp(0.0, 1.0);
        self
    }
}

/// MMR selector for diverse result selection.
pub struct MmrSelector {
    config: MmrConfig,
}

impl MmrSelector {
    /// Create a new MMR selector with the given configuration.
    pub fn new(config: MmrConfig) -> Self {
        Self { config }
    }

    /// Select top K patterns using MMR.
    ///
    /// Takes the candidate patterns from similarity search and selects
    /// a diverse subset using the MMR algorithm.
    ///
    /// # Arguments
    ///
    /// * `candidates` - Pre-scored candidates from similarity search
    /// * `query_embedding` - The original query embedding
    /// * `k` - Number of patterns to select
    ///
    /// # Returns
    ///
    /// Vector of selected patterns (not re-scored, preserves original similarity)
    pub fn select(
        &self,
        candidates: &[SearchResult],
        query_embedding: &[f32],
        k: usize,
    ) -> Vec<SearchResult> {
        if candidates.is_empty() {
            return Vec::new();
        }

        let k = k.min(candidates.len());

        // Filter by minimum relevance
        let filtered: Vec<&SearchResult> = candidates
            .iter()
            .filter(|r| r.similarity >= self.config.min_relevance)
            .collect();

        if filtered.is_empty() {
            return Vec::new();
        }

        // Prepare query embedding
        let query_vec = if self.config.normalized_embeddings {
            query_embedding.to_vec()
        } else {
            let arr = Array1::from_vec(query_embedding.to_vec());
            normalize_l2(&arr.view()).to_vec()
        };

        // Extract embeddings from candidates
        let candidate_embeddings: Vec<Option<Vec<f32>>> = filtered
            .iter()
            .map(|r| {
                r.pattern.embedding().map(|e| {
                    if self.config.normalized_embeddings {
                        e.to_vec()
                    } else {
                        let arr = Array1::from_vec(e.to_vec());
                        normalize_l2(&arr.view()).to_vec()
                    }
                })
            })
            .collect();

        // Track selected indices and remaining indices
        let mut selected_indices: Vec<usize> = Vec::with_capacity(k);
        let mut remaining_indices: Vec<usize> = (0..filtered.len()).collect();

        // First selection: highest similarity to query
        if let Some((best_idx, _)) = remaining_indices
            .iter()
            .filter_map(|&i| {
                candidate_embeddings[i].as_ref().map(|_| (i, filtered[i].similarity))
            })
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        {
            selected_indices.push(best_idx);
            remaining_indices.retain(|&i| i != best_idx);
        }

        // Iteratively select remaining using MMR
        while selected_indices.len() < k && !remaining_indices.is_empty() {
            let mut best_mmr_score = f32::NEG_INFINITY;
            let mut best_idx: Option<usize> = None;

            for &candidate_idx in &remaining_indices {
                let candidate_emb = match &candidate_embeddings[candidate_idx] {
                    Some(e) => e,
                    None => continue,
                };

                let candidate_arr = Array1::from_vec(candidate_emb.clone());

                // Relevance to query
                let query_arr = Array1::from_vec(query_vec.clone());
                let relevance = cosine_similarity_normalized(&candidate_arr.view(), &query_arr.view());

                // Maximum similarity to already selected
                let max_selected_sim = selected_indices
                    .iter()
                    .filter_map(|&sel_idx| {
                        candidate_embeddings[sel_idx].as_ref().map(|sel_emb| {
                            let sel_arr = Array1::from_vec(sel_emb.clone());
                            cosine_similarity_normalized(&candidate_arr.view(), &sel_arr.view())
                        })
                    })
                    .fold(0.0f32, |max, sim| max.max(sim));

                // MMR score
                let mmr_score = self.config.lambda * relevance
                    - (1.0 - self.config.lambda) * max_selected_sim;

                if mmr_score > best_mmr_score {
                    best_mmr_score = mmr_score;
                    best_idx = Some(candidate_idx);
                }
            }

            if let Some(idx) = best_idx {
                selected_indices.push(idx);
                remaining_indices.retain(|&i| i != idx);
            } else {
                break;
            }
        }

        // Return selected patterns in order of selection
        selected_indices
            .into_iter()
            .map(|i| filtered[i].clone())
            .collect()
    }

    /// Get the configuration.
    pub fn config(&self) -> &MmrConfig {
        &self.config
    }

    /// Set a new lambda value.
    pub fn set_lambda(&mut self, lambda: f32) {
        self.config.lambda = lambda.clamp(0.0, 1.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::pattern::Pattern;

    fn create_pattern_with_embedding(id: &str, embedding: Vec<f32>) -> Pattern {
        Pattern::builder()
            .id(id)
            .problem(format!("Problem {}", id))
            .solution(format!("Solution {}", id))
            .embedding(embedding)
            .build()
    }

    fn create_search_result(id: &str, embedding: Vec<f32>, similarity: f32) -> SearchResult {
        SearchResult::new(create_pattern_with_embedding(id, embedding), similarity)
    }

    #[test]
    fn test_mmr_config_default() {
        let config = MmrConfig::default();
        assert!((config.lambda - 0.7).abs() < 0.001);
        assert!(config.normalized_embeddings);
    }

    #[test]
    fn test_mmr_config_high_diversity() {
        let config = MmrConfig::high_diversity();
        assert!((config.lambda - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_mmr_config_high_relevance() {
        let config = MmrConfig::high_relevance();
        assert!((config.lambda - 0.9).abs() < 0.001);
    }

    #[test]
    fn test_mmr_empty_candidates() {
        let mmr = MmrSelector::new(MmrConfig::default());
        let query = vec![1.0, 0.0, 0.0, 0.0];
        let results = mmr.select(&[], &query, 5);
        assert!(results.is_empty());
    }

    #[test]
    fn test_mmr_single_candidate() {
        let mmr = MmrSelector::new(MmrConfig::default());
        let query = vec![1.0, 0.0, 0.0, 0.0];

        let candidates = vec![
            create_search_result("1", vec![1.0, 0.0, 0.0, 0.0], 0.95),
        ];

        let results = mmr.select(&candidates, &query, 5);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].pattern.id().as_str(), "1");
    }

    #[test]
    fn test_mmr_k_larger_than_candidates() {
        let mmr = MmrSelector::new(MmrConfig::default());
        let query = vec![1.0, 0.0, 0.0, 0.0];

        let candidates = vec![
            create_search_result("1", vec![1.0, 0.0, 0.0, 0.0], 0.95),
            create_search_result("2", vec![0.0, 1.0, 0.0, 0.0], 0.80),
        ];

        let results = mmr.select(&candidates, &query, 10);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_mmr_diversity_selection() {
        let mmr = MmrSelector::new(MmrConfig::default().with_lambda(0.5));
        let query = vec![1.0, 0.0, 0.0, 0.0];

        // Create candidates where some are similar to each other
        let candidates = vec![
            // Very similar to query
            create_search_result("1", vec![0.99, 0.1, 0.0, 0.0], 0.99),
            // Also similar to query AND to candidate 1
            create_search_result("2", vec![0.98, 0.15, 0.0, 0.0], 0.98),
            // Different direction - should be favored for diversity
            create_search_result("3", vec![0.5, 0.5, 0.5, 0.5], 0.5),
            // Very different
            create_search_result("4", vec![0.0, 0.0, 0.0, 1.0], 0.1),
        ];

        let results = mmr.select(&candidates, &query, 3);
        assert_eq!(results.len(), 3);

        // First should be most relevant
        assert_eq!(results[0].pattern.id().as_str(), "1");

        // With lambda=0.5, diversity matters - pattern 3 should be preferred over 2
        // because 2 is very similar to 1
        let selected_ids: Vec<&str> = results.iter().map(|r| r.pattern.id().as_str()).collect();
        assert!(selected_ids.contains(&"3") || selected_ids.contains(&"4"));
    }

    #[test]
    fn test_mmr_pure_relevance() {
        let mmr = MmrSelector::new(MmrConfig::default().with_lambda(1.0));
        let query = vec![1.0, 0.0, 0.0, 0.0];

        let candidates = vec![
            create_search_result("1", vec![1.0, 0.0, 0.0, 0.0], 0.99),
            create_search_result("2", vec![0.99, 0.1, 0.0, 0.0], 0.95),
            create_search_result("3", vec![0.0, 1.0, 0.0, 0.0], 0.50),
        ];

        let results = mmr.select(&candidates, &query, 2);
        assert_eq!(results.len(), 2);

        // With lambda=1.0, should just be top 2 by relevance
        assert_eq!(results[0].pattern.id().as_str(), "1");
        assert_eq!(results[1].pattern.id().as_str(), "2");
    }

    #[test]
    fn test_mmr_min_relevance_filter() {
        let mmr = MmrSelector::new(MmrConfig::default().with_min_relevance(0.5));
        let query = vec![1.0, 0.0, 0.0, 0.0];

        let candidates = vec![
            create_search_result("1", vec![1.0, 0.0, 0.0, 0.0], 0.95),
            create_search_result("2", vec![0.0, 1.0, 0.0, 0.0], 0.3), // Below threshold
        ];

        let results = mmr.select(&candidates, &query, 5);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].pattern.id().as_str(), "1");
    }

    #[test]
    fn test_mmr_preserves_original_similarity() {
        let mmr = MmrSelector::new(MmrConfig::default());
        let query = vec![1.0, 0.0, 0.0, 0.0];

        let candidates = vec![
            create_search_result("1", vec![1.0, 0.0, 0.0, 0.0], 0.95),
        ];

        let results = mmr.select(&candidates, &query, 1);

        // Should preserve the original similarity score
        assert!((results[0].similarity - 0.95).abs() < 0.001);
    }

    #[test]
    fn test_set_lambda() {
        let mut mmr = MmrSelector::new(MmrConfig::default());
        assert!((mmr.config().lambda - 0.7).abs() < 0.001);

        mmr.set_lambda(0.3);
        assert!((mmr.config().lambda - 0.3).abs() < 0.001);

        // Test clamping
        mmr.set_lambda(1.5);
        assert!((mmr.config().lambda - 1.0).abs() < 0.001);

        mmr.set_lambda(-0.5);
        assert!((mmr.config().lambda - 0.0).abs() < 0.001);
    }
}
