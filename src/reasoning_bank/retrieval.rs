//! Pattern retrieval with similarity search, MMR reranking, and multi-factor scoring.
//!
//! This module implements the core retrieval pipeline:
//! 1. Similarity search - Find candidates by vector similarity
//! 2. Domain filtering - Filter by domain/category hierarchy
//! 3. Reward threshold - Filter by minimum reward score
//! 4. MMR reranking - Maximal Marginal Relevance for diversity
//! 5. Multi-factor scoring - Combine similarity, recency, reliability, reuse
//!
//! # Example
//!
//! ```ignore
//! use nagual::reasoning_bank::{PatternQuery, retrieve_patterns};
//!
//! let query = PatternQuery::new("How to handle timeouts?")
//!     .with_domains(vec!["database", "networking"])
//!     .with_min_reward(0.7)
//!     .with_limit(5);
//!
//! let result = retrieve_patterns(&patterns, &query_embedding, &query)?;
//! ```

use std::collections::HashMap;
use std::collections::HashSet;

use ndarray::ArrayView1;
use serde::{Deserialize, Serialize};

use super::{Pattern, ReasoningBankError, ReasoningBankResult};
use crate::ml::cosine_similarity;

/// Configuration for pattern retrieval.
#[derive(Debug, Clone)]
pub struct RetrievalConfig {
    /// Maximum number of initial candidates from similarity search.
    pub max_candidates: usize,

    /// MMR configuration for diversity.
    pub mmr: MmrConfig,

    /// Scoring weights for multi-factor ranking.
    pub scoring_weights: ScoringWeights,

    /// Default minimum reward threshold (0.0 = no filter).
    pub default_min_reward: f32,

    /// Default number of results to return.
    pub default_limit: usize,
}

impl Default for RetrievalConfig {
    fn default() -> Self {
        Self {
            max_candidates: 100,
            mmr: MmrConfig::default(),
            scoring_weights: ScoringWeights::default(),
            default_min_reward: 0.0,
            default_limit: 10,
        }
    }
}

/// Configuration for Maximal Marginal Relevance (MMR) algorithm.
///
/// MMR balances relevance to the query with diversity among results:
/// `MMR = lambda * similarity(doc, query) - (1 - lambda) * max(similarity(doc, selected))`
#[derive(Debug, Clone)]
pub struct MmrConfig {
    /// Lambda parameter (0.0 = maximum diversity, 1.0 = no diversity).
    /// Default is 0.7 (bias toward relevance with some diversity).
    pub lambda: f32,

    /// Whether to enable MMR reranking.
    pub enabled: bool,
}

impl Default for MmrConfig {
    fn default() -> Self {
        Self {
            lambda: 0.7,
            enabled: true,
        }
    }
}

impl MmrConfig {
    /// Create a config with custom lambda.
    pub fn with_lambda(mut self, lambda: f32) -> Self {
        self.lambda = lambda.clamp(0.0, 1.0);
        self
    }

    /// Disable MMR (pure similarity ranking).
    pub fn disabled() -> Self {
        Self {
            lambda: 1.0,
            enabled: false,
        }
    }
}

/// Weights for multi-factor scoring.
///
/// Final score = sum(weight_i * factor_i) / sum(weights)
#[derive(Debug, Clone)]
pub struct ScoringWeights {
    /// Weight for vector similarity (0.0 - 1.0).
    pub similarity: f32,

    /// Weight for recency (newer patterns score higher).
    pub recency: f32,

    /// Weight for reliability (confidence + success rate).
    pub reliability: f32,

    /// Weight for reuse count (frequently used patterns).
    pub reuse: f32,

    /// Weight for reward score.
    pub reward: f32,
}

impl Default for ScoringWeights {
    fn default() -> Self {
        Self {
            similarity: 0.5,
            recency: 0.1,
            reliability: 0.2,
            reuse: 0.1,
            reward: 0.1,
        }
    }
}

impl ScoringWeights {
    /// Validate that weights sum to a reasonable value.
    pub fn validate(&self) -> ReasoningBankResult<()> {
        let sum = self.similarity + self.recency + self.reliability + self.reuse + self.reward;
        if sum <= 0.0 {
            return Err(ReasoningBankError::InvalidQuery {
                reason: "Scoring weights must sum to a positive value".to_string(),
            });
        }
        Ok(())
    }

    /// Create similarity-only weights.
    pub fn similarity_only() -> Self {
        Self {
            similarity: 1.0,
            recency: 0.0,
            reliability: 0.0,
            reuse: 0.0,
            reward: 0.0,
        }
    }

    /// Create balanced weights.
    pub fn balanced() -> Self {
        Self {
            similarity: 0.35,
            recency: 0.15,
            reliability: 0.2,
            reuse: 0.15,
            reward: 0.15,
        }
    }
}

/// Configuration for hybrid FTS5 + vector search.
#[derive(Debug, Clone)]
pub struct HybridSearchConfig {
    /// Weight for FTS5 BM25 keyword score (0.0 - 1.0). Default: 0.3
    pub fts_weight: f32,
    /// Weight for cosine embedding similarity (0.0 - 1.0). Default: 0.7
    pub vector_weight: f32,
    /// Maximum FTS candidates to consider.
    pub fts_max_candidates: usize,
}

impl Default for HybridSearchConfig {
    fn default() -> Self {
        Self {
            fts_weight: 0.3,
            vector_weight: 0.7,
            fts_max_candidates: 50,
        }
    }
}

impl HybridSearchConfig {
    pub fn with_weights(fts_weight: f32, vector_weight: f32) -> Self {
        Self {
            fts_weight: fts_weight.clamp(0.0, 1.0),
            vector_weight: vector_weight.clamp(0.0, 1.0),
            fts_max_candidates: 50,
        }
    }

    pub fn validate(&self) -> ReasoningBankResult<()> {
        let sum = self.fts_weight + self.vector_weight;
        if sum <= 0.0 {
            return Err(ReasoningBankError::InvalidQuery {
                reason: "Hybrid search weights must sum to a positive value".to_string(),
            });
        }
        Ok(())
    }
}

/// Query parameters for pattern retrieval.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternQuery {
    /// The query text (for display/logging).
    pub query_text: String,

    /// Domains to filter by (OR logic, empty = all domains).
    pub domains: Vec<String>,

    /// Minimum reward threshold (0.0 - 1.0).
    pub min_reward: f32,

    /// Maximum number of results to return.
    pub limit: usize,

    /// Minimum similarity score (0.0 - 1.0).
    pub min_similarity: f32,

    /// Tags to filter by (AND logic, empty = all tags).
    pub tags: Vec<String>,

    /// Whether to include patterns from the same session.
    pub include_same_session: bool,

    /// Session ID to exclude (optional).
    pub exclude_session_id: Option<String>,

    /// Whether to only include successful patterns.
    pub only_successful: bool,
}

impl PatternQuery {
    /// Create a new query with the given text.
    pub fn new(query_text: impl Into<String>) -> Self {
        Self {
            query_text: query_text.into(),
            domains: Vec::new(),
            min_reward: 0.0,
            limit: 10,
            min_similarity: 0.0,
            tags: Vec::new(),
            include_same_session: true,
            exclude_session_id: None,
            only_successful: false,
        }
    }

    /// Set domains to filter by (OR logic).
    pub fn with_domains(mut self, domains: Vec<impl Into<String>>) -> Self {
        self.domains = domains.into_iter().map(|d| d.into()).collect();
        self
    }

    /// Set minimum reward threshold.
    pub fn with_min_reward(mut self, min_reward: f32) -> Self {
        self.min_reward = min_reward.clamp(0.0, 1.0);
        self
    }

    /// Set maximum number of results.
    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = limit.max(1);
        self
    }

    /// Set minimum similarity threshold.
    pub fn with_min_similarity(mut self, min_similarity: f32) -> Self {
        self.min_similarity = min_similarity.clamp(0.0, 1.0);
        self
    }

    /// Set tags to filter by (AND logic).
    pub fn with_tags(mut self, tags: Vec<impl Into<String>>) -> Self {
        self.tags = tags.into_iter().map(|t| t.into()).collect();
        self
    }

    /// Exclude patterns from the specified session.
    pub fn exclude_session(mut self, session_id: impl Into<String>) -> Self {
        self.exclude_session_id = Some(session_id.into());
        self.include_same_session = false;
        self
    }

    /// Only include patterns with success_rate > 0.5.
    pub fn only_successful(mut self) -> Self {
        self.only_successful = true;
        self
    }
}

impl Default for PatternQuery {
    fn default() -> Self {
        Self::new("")
    }
}

/// A pattern with its computed scores.
#[derive(Debug, Clone)]
pub struct ScoredPattern {
    /// The pattern.
    pub pattern: Pattern,

    /// Similarity score to the query (0.0 - 1.0).
    pub similarity: f32,

    /// Final combined score after multi-factor scoring.
    pub final_score: f32,

    /// Individual factor scores for debugging/transparency.
    pub factor_scores: FactorScores,
}

/// Individual factor scores for a pattern.
#[derive(Debug, Clone, Default)]
pub struct FactorScores {
    /// Similarity score contribution.
    pub similarity: f32,
    /// Recency score contribution.
    pub recency: f32,
    /// Reliability score contribution.
    pub reliability: f32,
    /// Reuse score contribution.
    pub reuse: f32,
    /// Reward score contribution.
    pub reward: f32,
}

/// Result of a pattern retrieval operation.
#[derive(Debug, Clone)]
pub struct RetrievalResult {
    /// Retrieved patterns with scores.
    pub patterns: Vec<ScoredPattern>,

    /// Total number of patterns before filtering.
    pub total_candidates: usize,

    /// Number of patterns after domain filtering.
    pub after_domain_filter: usize,

    /// Number of patterns after reward threshold filtering.
    pub after_reward_filter: usize,

    /// Time taken for retrieval (in milliseconds).
    pub retrieval_time_ms: u64,

    /// The query that was used.
    pub query: PatternQuery,
}

impl RetrievalResult {
    /// Check if any patterns were found.
    pub fn is_empty(&self) -> bool {
        self.patterns.is_empty()
    }

    /// Get the number of patterns returned.
    pub fn len(&self) -> usize {
        self.patterns.len()
    }

    /// Get the top pattern (highest score).
    pub fn top(&self) -> Option<&ScoredPattern> {
        self.patterns.first()
    }

    /// Get patterns as a slice.
    pub fn as_slice(&self) -> &[ScoredPattern] {
        &self.patterns
    }
}

/// Main retrieval function that implements the full pipeline.
///
/// Pipeline:
/// 1. Compute similarity scores for all patterns with embeddings
/// 2. Filter by domain (OR logic across domains)
/// 3. Filter by minimum reward threshold
/// 4. Apply additional filters (tags, session, success)
/// 5. Apply MMR for diversity
/// 6. Compute multi-factor scores
/// 7. Sort and return top K
///
/// # Arguments
///
/// * `patterns` - All available patterns
/// * `query_embedding` - The query embedding vector
/// * `query` - Query parameters including filters and limits
/// * `config` - Retrieval configuration
///
/// # Returns
///
/// `RetrievalResult` containing scored patterns and metadata
pub fn retrieve_patterns(
    patterns: &[Pattern],
    query_embedding: &ArrayView1<f32>,
    query: &PatternQuery,
    config: &RetrievalConfig,
) -> ReasoningBankResult<RetrievalResult> {
    let start = std::time::Instant::now();
    let total_candidates = patterns.len();

    // Step 1: Compute similarity scores for patterns with embeddings
    let mut candidates: Vec<(usize, f32)> = patterns
        .iter()
        .enumerate()
        .filter_map(|(idx, p)| {
            p.embedding.as_ref().map(|emb| {
                let sim = cosine_similarity(query_embedding, &emb.view());
                (idx, sim)
            })
        })
        .filter(|(_, sim)| *sim >= query.min_similarity)
        .collect();

    // Sort by similarity descending
    candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Limit to max_candidates for efficiency
    candidates.truncate(config.max_candidates);

    // Step 2: Filter by domain (OR logic)
    let after_domain: Vec<(usize, f32)> = if query.domains.is_empty() {
        candidates
    } else {
        candidates
            .into_iter()
            .filter(|(idx, _)| {
                let pattern = &patterns[*idx];
                query.domains.iter().any(|d| pattern.matches_domain(d))
            })
            .collect()
    };
    let after_domain_filter = after_domain.len();

    // Step 3: Filter by minimum reward threshold
    let after_reward: Vec<(usize, f32)> = after_domain
        .into_iter()
        .filter(|(idx, _)| patterns[*idx].reward >= query.min_reward)
        .collect();
    let after_reward_filter = after_reward.len();

    // Step 4: Apply additional filters
    let filtered: Vec<(usize, f32)> = after_reward
        .into_iter()
        .filter(|(idx, _)| {
            let pattern = &patterns[*idx];

            // Tag filter (AND logic)
            if !query.tags.is_empty() {
                let pattern_tags: HashSet<_> = pattern.tags.iter().collect();
                if !query.tags.iter().all(|t| pattern_tags.contains(t)) {
                    return false;
                }
            }

            // Session filter
            if let Some(ref exclude_session) = query.exclude_session_id {
                if let Some(ref pattern_session) = pattern.session_id {
                    if pattern_session == exclude_session {
                        return false;
                    }
                }
            }

            // Success filter
            if query.only_successful && pattern.success_rate <= 0.5 {
                return false;
            }

            true
        })
        .collect();

    // Step 5: Apply MMR for diversity
    let diverse_candidates = if config.mmr.enabled && !filtered.is_empty() {
        apply_mmr(
            &filtered,
            patterns,
            query_embedding,
            config.mmr.lambda,
            query.limit,
        )
    } else {
        filtered.into_iter().take(query.limit).collect()
    };

    // Step 6: Compute multi-factor scores
    let mut scored_patterns: Vec<ScoredPattern> = diverse_candidates
        .into_iter()
        .map(|(idx, similarity)| {
            let pattern = patterns[idx].clone();
            let factor_scores = compute_factor_scores(&pattern, similarity, &config.scoring_weights);
            let final_score = compute_final_score(&factor_scores, &config.scoring_weights);

            ScoredPattern {
                pattern,
                similarity,
                final_score,
                factor_scores,
            }
        })
        .collect();

    // Step 7: Sort by final score
    scored_patterns.sort_by(|a, b| {
        b.final_score
            .partial_cmp(&a.final_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let retrieval_time_ms = start.elapsed().as_millis() as u64;

    Ok(RetrievalResult {
        patterns: scored_patterns,
        total_candidates,
        after_domain_filter,
        after_reward_filter,
        retrieval_time_ms,
        query: query.clone(),
    })
}

/// Hybrid FTS5 + vector retrieval pipeline.
///
/// Combines keyword (BM25/FTS5) scores with cosine embedding similarity
/// for retrieval that benefits from both lexical and semantic matching.
///
/// Pipeline:
/// 1. Build normalized FTS score map (BM25 ranks are negative, normalize to [0.1, 1.0])
/// 2. Compute cosine similarity for patterns with embeddings
/// 3. Combine: `vector_weight * cosine_sim + fts_weight * fts_norm`
/// 4. Apply domain, reward, tag, session, and success filters
/// 5. Apply MMR for diversity
/// 6. Compute multi-factor scores
/// 7. Sort and return top K
///
/// # Arguments
///
/// * `patterns` - All available patterns
/// * `query_embedding` - The query embedding vector
/// * `fts_results` - FTS5 results as (pattern_id, bm25_rank) pairs
/// * `query` - Query parameters including filters and limits
/// * `config` - Retrieval configuration
/// * `hybrid_config` - Hybrid search weights and parameters
///
/// # Returns
///
/// `RetrievalResult` containing scored patterns and metadata
pub fn retrieve_patterns_hybrid(
    patterns: &[Pattern],
    query_embedding: &ArrayView1<f32>,
    fts_results: &[(String, f64)],
    query: &PatternQuery,
    config: &RetrievalConfig,
    hybrid_config: &HybridSearchConfig,
) -> ReasoningBankResult<RetrievalResult> {
    hybrid_config.validate()?;
    let start = std::time::Instant::now();
    let total_candidates = patterns.len();

    // Step 1: Build normalized FTS score map.
    // BM25 ranks are typically negative (lower = better match).
    // Normalize to [0.1, 1.0] range where 1.0 is the best match.
    let fts_scores: HashMap<&str, f32> = if fts_results.is_empty() {
        HashMap::new()
    } else {
        // Find min/max BM25 ranks for normalization
        let min_rank = fts_results.iter().map(|(_, r)| *r).fold(f64::INFINITY, f64::min);
        let max_rank = fts_results.iter().map(|(_, r)| *r).fold(f64::NEG_INFINITY, f64::max);
        let range = max_rank - min_rank;

        fts_results
            .iter()
            .take(hybrid_config.fts_max_candidates)
            .map(|(id, rank)| {
                let normalized = if range.abs() < f64::EPSILON {
                    // All ranks are the same, give a uniform score
                    0.5
                } else {
                    // BM25 ranks: lower (more negative) = better match
                    // Invert so that best match -> highest score
                    1.0 - ((rank - min_rank) / range) as f32
                };
                // Clamp to [0.1, 1.0] -- even worst FTS match gets a small boost
                let clamped = normalized.clamp(0.1, 1.0);
                (id.as_str(), clamped)
            })
            .collect()
    };

    // Step 2: Compute combined scores for all patterns with embeddings
    let mut candidates: Vec<(usize, f32)> = patterns
        .iter()
        .enumerate()
        .filter_map(|(idx, p)| {
            let cosine_sim = p.embedding.as_ref().map(|emb| {
                cosine_similarity(query_embedding, &emb.view())
            });

            let fts_score = fts_scores.get(p.id.as_str()).copied();

            // Must have at least one score source
            if cosine_sim.is_none() && fts_score.is_none() {
                return None;
            }

            let vec_component = cosine_sim.unwrap_or(0.0) * hybrid_config.vector_weight;
            let fts_component = fts_score.unwrap_or(0.0) * hybrid_config.fts_weight;
            let combined = vec_component + fts_component;

            if combined >= query.min_similarity {
                Some((idx, combined))
            } else {
                None
            }
        })
        .collect();

    // Sort by combined score descending
    candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Limit to max_candidates for efficiency
    candidates.truncate(config.max_candidates);

    // Step 3: Filter by domain (OR logic)
    let after_domain: Vec<(usize, f32)> = if query.domains.is_empty() {
        candidates
    } else {
        candidates
            .into_iter()
            .filter(|(idx, _)| {
                let pattern = &patterns[*idx];
                query.domains.iter().any(|d| pattern.matches_domain(d))
            })
            .collect()
    };
    let after_domain_filter = after_domain.len();

    // Step 4: Filter by minimum reward threshold
    let after_reward: Vec<(usize, f32)> = after_domain
        .into_iter()
        .filter(|(idx, _)| patterns[*idx].reward >= query.min_reward)
        .collect();
    let after_reward_filter = after_reward.len();

    // Step 5: Apply additional filters (tags, session, success)
    let filtered: Vec<(usize, f32)> = after_reward
        .into_iter()
        .filter(|(idx, _)| {
            let pattern = &patterns[*idx];

            // Tag filter (AND logic)
            if !query.tags.is_empty() {
                let pattern_tags: HashSet<_> = pattern.tags.iter().collect();
                if !query.tags.iter().all(|t| pattern_tags.contains(t)) {
                    return false;
                }
            }

            // Session filter
            if let Some(ref exclude_session) = query.exclude_session_id {
                if let Some(ref pattern_session) = pattern.session_id {
                    if pattern_session == exclude_session {
                        return false;
                    }
                }
            }

            // Success filter
            if query.only_successful && pattern.success_rate <= 0.5 {
                return false;
            }

            true
        })
        .collect();

    // Step 6: Apply MMR for diversity
    let diverse_candidates = if config.mmr.enabled && !filtered.is_empty() {
        apply_mmr(
            &filtered,
            patterns,
            query_embedding,
            config.mmr.lambda,
            query.limit,
        )
    } else {
        filtered.into_iter().take(query.limit).collect()
    };

    // Step 7: Compute multi-factor scores
    let mut scored_patterns: Vec<ScoredPattern> = diverse_candidates
        .into_iter()
        .map(|(idx, similarity)| {
            let pattern = patterns[idx].clone();
            let factor_scores = compute_factor_scores(&pattern, similarity, &config.scoring_weights);
            let final_score = compute_final_score(&factor_scores, &config.scoring_weights);

            ScoredPattern {
                pattern,
                similarity,
                final_score,
                factor_scores,
            }
        })
        .collect();

    // Step 8: Sort by final score
    scored_patterns.sort_by(|a, b| {
        b.final_score
            .partial_cmp(&a.final_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let retrieval_time_ms = start.elapsed().as_millis() as u64;

    Ok(RetrievalResult {
        patterns: scored_patterns,
        total_candidates,
        after_domain_filter,
        after_reward_filter,
        retrieval_time_ms,
        query: query.clone(),
    })
}

/// Apply Maximal Marginal Relevance (MMR) for diverse selection.
///
/// MMR iteratively selects documents that are both relevant to the query
/// and different from already selected documents.
fn apply_mmr(
    candidates: &[(usize, f32)],
    patterns: &[Pattern],
    _query_embedding: &ArrayView1<f32>,
    lambda: f32,
    k: usize,
) -> Vec<(usize, f32)> {
    if candidates.is_empty() || k == 0 {
        return Vec::new();
    }

    let mut selected: Vec<(usize, f32)> = Vec::with_capacity(k);
    let mut remaining: Vec<(usize, f32)> = candidates.to_vec();

    // Select the first document (highest similarity)
    if let Some(first) = remaining.first().cloned() {
        selected.push(first);
        remaining.remove(0);
    }

    // Iteratively select remaining documents
    while selected.len() < k && !remaining.is_empty() {
        let mut best_idx = 0;
        let mut best_mmr_score = f32::NEG_INFINITY;

        for (i, (pattern_idx, similarity)) in remaining.iter().enumerate() {
            // Compute max similarity to selected documents
            let max_sim_to_selected = selected
                .iter()
                .map(|(sel_idx, _)| {
                    if let (Some(emb1), Some(emb2)) = (
                        patterns[*pattern_idx].embedding.as_ref(),
                        patterns[*sel_idx].embedding.as_ref(),
                    ) {
                        cosine_similarity(&emb1.view(), &emb2.view())
                    } else {
                        0.0
                    }
                })
                .fold(f32::NEG_INFINITY, f32::max);

            // MMR score: lambda * similarity - (1 - lambda) * max_sim_to_selected
            let mmr_score = lambda * similarity - (1.0 - lambda) * max_sim_to_selected;

            if mmr_score > best_mmr_score {
                best_mmr_score = mmr_score;
                best_idx = i;
            }
        }

        selected.push(remaining.remove(best_idx));
    }

    selected
}

/// Compute individual factor scores for a pattern.
fn compute_factor_scores(
    pattern: &Pattern,
    similarity: f32,
    weights: &ScoringWeights,
) -> FactorScores {
    // Normalize reuse count (log scale, capped at 100)
    let reuse_score = if pattern.usage_count > 0 {
        (pattern.usage_count as f32).ln() / (100.0_f32).ln()
    } else {
        0.0
    }
    .min(1.0);

    FactorScores {
        similarity: similarity * weights.similarity,
        recency: pattern.recency_score() * weights.recency,
        reliability: pattern.reliability_score() * weights.reliability,
        reuse: reuse_score * weights.reuse,
        reward: pattern.reward * weights.reward,
    }
}

/// Compute the final combined score from factor scores.
fn compute_final_score(factors: &FactorScores, weights: &ScoringWeights) -> f32 {
    let total_weight =
        weights.similarity + weights.recency + weights.reliability + weights.reuse + weights.reward;

    if total_weight <= 0.0 {
        return 0.0;
    }

    let raw_score =
        factors.similarity + factors.recency + factors.reliability + factors.reuse + factors.reward;

    // Normalize to 0-1 range
    (raw_score / total_weight).clamp(0.0, 1.0)
}

/// Hyperbolic retrieval mode configuration.
#[derive(Debug, Clone)]
pub struct HyperbolicRetrievalConfig {
    /// Hyperbolic geometry configuration.
    pub hyperbolic_config: crate::ml::HyperbolicConfig,
    /// Weight for Poincare distance vs cosine similarity (0.0 = pure cosine, 1.0 = pure Poincare).
    pub poincare_weight: f32,
    /// Whether to use hierarchy-aware distance (penalizes cross-level jumps).
    pub hierarchy_aware: bool,
}

impl Default for HyperbolicRetrievalConfig {
    fn default() -> Self {
        Self {
            hyperbolic_config: crate::ml::HyperbolicConfig::default(),
            poincare_weight: 0.3,
            hierarchy_aware: true,
        }
    }
}

/// Retrieve patterns using hyperbolic distance for hierarchy-aware ranking.
///
/// This combines cosine similarity (Euclidean) with Poincare distance (hyperbolic)
/// for retrieval that respects domain hierarchy. Patterns in the same hierarchy
/// branch are ranked higher than patterns at similar Euclidean distance but
/// different hierarchy levels.
///
/// # Arguments
///
/// * `patterns` - All available patterns
/// * `query_embedding` - The query embedding vector
/// * `query` - Query parameters including filters and limits
/// * `config` - Base retrieval configuration
/// * `hyper_config` - Hyperbolic retrieval configuration
///
/// # Returns
///
/// `RetrievalResult` containing scored patterns re-ranked using hyperbolic distance
pub fn retrieve_patterns_hyperbolic(
    patterns: &[Pattern],
    query_embedding: &ArrayView1<f32>,
    query: &PatternQuery,
    config: &RetrievalConfig,
    hyper_config: &HyperbolicRetrievalConfig,
) -> ReasoningBankResult<RetrievalResult> {
    // 1. Get base retrieval result using existing pipeline
    let mut result = retrieve_patterns(patterns, query_embedding, query, config)?;

    // 2. Re-score using hyperbolic distance
    let embedder = crate::ml::HyperbolicEmbedder::new(hyper_config.hyperbolic_config.clone());

    // Convert query to Poincare point
    let query_domain = query
        .domains
        .first()
        .map(|s| s.as_str())
        .unwrap_or("general");
    let query_hyper = match embedder.embed_from_euclidean(query_embedding, query_domain) {
        Ok(p) => p,
        Err(_) => return Ok(result), // Fallback to Euclidean-only
    };

    // Re-score each pattern
    let w = hyper_config.poincare_weight;
    for scored in &mut result.patterns {
        if let Some(ref emb) = scored.pattern.embedding {
            let pattern_hyper = match embedder.embed_from_euclidean(&emb.view(), &scored.pattern.domain) {
                Ok(p) => p,
                Err(_) => continue,
            };

            let poincare_dist = crate::ml::poincare_distance(
                &query_hyper.coords().view(),
                &pattern_hyper.coords().view(),
            );

            // Convert distance to similarity (inverse mapping)
            let poincare_sim = 1.0 / (1.0 + poincare_dist as f32);

            // Blend cosine similarity with Poincare similarity
            let blended = (1.0 - w) * scored.similarity + w * poincare_sim;

            // Apply hierarchy bonus if hierarchy_aware
            let hierarchy_bonus = if hyper_config.hierarchy_aware {
                let depth_diff = (query_hyper.depth - pattern_hyper.depth).abs() as f32;
                1.0 - (depth_diff * 0.1).min(0.3) // Up to 30% penalty for depth mismatch
            } else {
                1.0
            };

            scored.final_score = scored.final_score * (1.0 - w) + blended * hierarchy_bonus * w;
        }
    }

    // Re-sort by new final_score
    result.patterns.sort_by(|a, b| {
        b.final_score
            .partial_cmp(&a.final_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use ndarray::Array1;

    fn create_test_pattern(id: &str, problem: &str, domain: &str, reward: f32) -> Pattern {
        let mut p = Pattern::new(problem, "Solution", domain).with_reward(reward);
        p.id = id.to_string();
        // Create a simple embedding for testing
        let embedding = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
        p.embedding = Some(embedding);
        p
    }

    fn create_test_embedding() -> Array1<f32> {
        Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0])
    }

    #[test]
    fn test_pattern_query_builder() {
        let query = PatternQuery::new("How to handle errors?")
            .with_domains(vec!["rust", "python"])
            .with_min_reward(0.7)
            .with_limit(5)
            .with_min_similarity(0.5);

        assert_eq!(query.query_text, "How to handle errors?");
        assert_eq!(query.domains, vec!["rust", "python"]);
        assert_eq!(query.min_reward, 0.7);
        assert_eq!(query.limit, 5);
        assert_eq!(query.min_similarity, 0.5);
    }

    #[test]
    fn test_retrieve_patterns_basic() {
        let patterns = vec![
            create_test_pattern("p1", "Error handling", "rust", 0.8),
            create_test_pattern("p2", "Database connection", "database", 0.6),
            create_test_pattern("p3", "Async programming", "rust.async", 0.9),
        ];

        let query_embedding = create_test_embedding();
        let query = PatternQuery::new("Error handling").with_limit(10);
        let config = RetrievalConfig::default();

        let result = retrieve_patterns(&patterns, &query_embedding.view(), &query, &config).unwrap();

        assert!(!result.is_empty());
        assert_eq!(result.total_candidates, 3);
    }

    #[test]
    fn test_retrieve_patterns_domain_filter() {
        let patterns = vec![
            create_test_pattern("p1", "Error handling", "rust", 0.8),
            create_test_pattern("p2", "Database connection", "database", 0.6),
            create_test_pattern("p3", "Async programming", "rust.async", 0.9),
        ];

        let query_embedding = create_test_embedding();
        let query = PatternQuery::new("Test")
            .with_domains(vec!["rust"])
            .with_limit(10);
        let config = RetrievalConfig::default();

        let result = retrieve_patterns(&patterns, &query_embedding.view(), &query, &config).unwrap();

        // Should match "rust" and "rust.async" (hierarchy match)
        assert_eq!(result.patterns.len(), 2);
        assert!(result
            .patterns
            .iter()
            .all(|p| p.pattern.domain.starts_with("rust")));
    }

    #[test]
    fn test_retrieve_patterns_reward_filter() {
        let patterns = vec![
            create_test_pattern("p1", "Low reward", "test", 0.3),
            create_test_pattern("p2", "Medium reward", "test", 0.5),
            create_test_pattern("p3", "High reward", "test", 0.9),
        ];

        let query_embedding = create_test_embedding();
        let query = PatternQuery::new("Test").with_min_reward(0.5).with_limit(10);
        let config = RetrievalConfig::default();

        let result = retrieve_patterns(&patterns, &query_embedding.view(), &query, &config).unwrap();

        // Should only include patterns with reward >= 0.5
        assert_eq!(result.patterns.len(), 2);
        assert!(result.patterns.iter().all(|p| p.pattern.reward >= 0.5));
    }

    #[test]
    fn test_retrieve_patterns_combined_filters() {
        let patterns = vec![
            create_test_pattern("p1", "Rust error", "rust", 0.8),
            create_test_pattern("p2", "Rust async", "rust.async", 0.4), // Low reward
            create_test_pattern("p3", "Python error", "python", 0.9),   // Wrong domain
            create_test_pattern("p4", "Rust tokio", "rust.async.tokio", 0.9),
        ];

        let query_embedding = create_test_embedding();
        let query = PatternQuery::new("Test")
            .with_domains(vec!["rust"])
            .with_min_reward(0.7)
            .with_limit(10);
        let config = RetrievalConfig::default();

        let result = retrieve_patterns(&patterns, &query_embedding.view(), &query, &config).unwrap();

        // Should match rust domain hierarchy AND reward >= 0.7
        assert_eq!(result.patterns.len(), 2);
    }

    #[test]
    fn test_mmr_config() {
        let default_config = MmrConfig::default();
        assert_eq!(default_config.lambda, 0.7);
        assert!(default_config.enabled);

        let custom_config = MmrConfig::default().with_lambda(0.5);
        assert_eq!(custom_config.lambda, 0.5);

        let disabled_config = MmrConfig::disabled();
        assert!(!disabled_config.enabled);
    }

    #[test]
    fn test_scoring_weights() {
        let default_weights = ScoringWeights::default();
        assert!(default_weights.validate().is_ok());

        let similarity_only = ScoringWeights::similarity_only();
        assert_eq!(similarity_only.similarity, 1.0);
        assert_eq!(similarity_only.recency, 0.0);
    }

    #[test]
    fn test_scored_pattern() {
        let pattern = create_test_pattern("p1", "Test", "test", 0.8);
        let scored = ScoredPattern {
            pattern: pattern.clone(),
            similarity: 0.9,
            final_score: 0.85,
            factor_scores: FactorScores::default(),
        };

        assert_eq!(scored.similarity, 0.9);
        assert_eq!(scored.final_score, 0.85);
    }

    #[test]
    fn test_retrieval_result() {
        let result = RetrievalResult {
            patterns: vec![],
            total_candidates: 100,
            after_domain_filter: 50,
            after_reward_filter: 25,
            retrieval_time_ms: 5,
            query: PatternQuery::default(),
        };

        assert!(result.is_empty());
        assert_eq!(result.len(), 0);
        assert!(result.top().is_none());
    }

    #[test]
    fn test_apply_mmr() {
        // Create patterns with distinct embeddings
        let mut patterns = vec![
            create_test_pattern("p1", "A", "test", 0.8),
            create_test_pattern("p2", "B", "test", 0.8),
            create_test_pattern("p3", "C", "test", 0.8),
        ];

        // Give them different embeddings
        patterns[0].embedding = Some(Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]));
        patterns[1].embedding = Some(Array1::from_vec(vec![0.9, 0.1, 0.0, 0.0])); // Similar to p1
        patterns[2].embedding = Some(Array1::from_vec(vec![0.0, 0.0, 1.0, 0.0])); // Different

        let query_embedding = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
        let candidates: Vec<(usize, f32)> = vec![(0, 1.0), (1, 0.9), (2, 0.0)];

        // With lambda = 0.5, should prefer diversity
        let selected = apply_mmr(&candidates, &patterns, &query_embedding.view(), 0.5, 2);

        assert_eq!(selected.len(), 2);
        // First should be p1 (highest similarity)
        assert_eq!(selected[0].0, 0);
    }

    #[test]
    fn test_compute_factor_scores() {
        let mut pattern = create_test_pattern("p1", "Test", "test", 0.8);
        pattern.usage_count = 10;
        pattern.success_count = 8;
        pattern.success_rate = 0.8;
        pattern.confidence = 0.9;

        let weights = ScoringWeights::default();
        let scores = compute_factor_scores(&pattern, 0.95, &weights);

        assert!(scores.similarity > 0.0);
        assert!(scores.reward > 0.0);
        assert!(scores.reliability > 0.0);
    }

    #[test]
    fn test_compute_final_score() {
        let factors = FactorScores {
            similarity: 0.5,
            recency: 0.1,
            reliability: 0.2,
            reuse: 0.1,
            reward: 0.1,
        };

        let weights = ScoringWeights::default();
        let final_score = compute_final_score(&factors, &weights);

        assert!(final_score >= 0.0);
        assert!(final_score <= 1.0);
    }

    #[test]
    fn test_hyperbolic_retrieval_config_default() {
        let config = HyperbolicRetrievalConfig::default();
        assert_eq!(config.poincare_weight, 0.3);
        assert!(config.hierarchy_aware);
        assert_eq!(config.hyperbolic_config.dimension, 128);
    }

    #[test]
    fn test_retrieve_patterns_hyperbolic_basic() {
        let patterns = vec![
            create_test_pattern("p1", "Error handling", "rust", 0.8),
            create_test_pattern("p2", "Database connection", "database", 0.6),
            create_test_pattern("p3", "Async programming", "rust.async", 0.9),
        ];

        let query_embedding = create_test_embedding();
        let query = PatternQuery::new("Error handling").with_limit(10);
        let config = RetrievalConfig::default();
        let hyper_config = HyperbolicRetrievalConfig::default();

        let result = retrieve_patterns_hyperbolic(
            &patterns,
            &query_embedding.view(),
            &query,
            &config,
            &hyper_config,
        )
        .unwrap();

        assert!(!result.is_empty());
        assert_eq!(result.total_candidates, 3);
    }

    #[test]
    fn test_retrieve_patterns_hyperbolic_preserves_count() {
        let patterns = vec![
            create_test_pattern("p1", "Pattern A", "rust", 0.8),
            create_test_pattern("p2", "Pattern B", "rust.async", 0.7),
        ];

        let query_embedding = create_test_embedding();
        let query = PatternQuery::new("Test").with_limit(10);
        let config = RetrievalConfig::default();
        let hyper_config = HyperbolicRetrievalConfig::default();

        let euclidean_result =
            retrieve_patterns(&patterns, &query_embedding.view(), &query, &config).unwrap();
        let hyperbolic_result = retrieve_patterns_hyperbolic(
            &patterns,
            &query_embedding.view(),
            &query,
            &config,
            &hyper_config,
        )
        .unwrap();

        // Same number of results should be returned
        assert_eq!(euclidean_result.len(), hyperbolic_result.len());
    }

    #[test]
    fn test_retrieve_patterns_hyperbolic_with_domain_filter() {
        let patterns = vec![
            create_test_pattern("p1", "Rust error", "rust", 0.8),
            create_test_pattern("p2", "Python error", "python", 0.9),
            create_test_pattern("p3", "Rust async", "rust.async", 0.7),
        ];

        let query_embedding = create_test_embedding();
        let query = PatternQuery::new("Error handling")
            .with_domains(vec!["rust"])
            .with_limit(10);
        let config = RetrievalConfig::default();
        let hyper_config = HyperbolicRetrievalConfig::default();

        let result = retrieve_patterns_hyperbolic(
            &patterns,
            &query_embedding.view(),
            &query,
            &config,
            &hyper_config,
        )
        .unwrap();

        // Should only return rust domain patterns
        assert_eq!(result.len(), 2);
        assert!(result
            .patterns
            .iter()
            .all(|p| p.pattern.domain.starts_with("rust")));
    }

    #[test]
    fn test_retrieve_patterns_hyperbolic_no_hierarchy() {
        let patterns = vec![
            create_test_pattern("p1", "Pattern A", "rust", 0.8),
            create_test_pattern("p2", "Pattern B", "rust.async.tokio", 0.7),
        ];

        let query_embedding = create_test_embedding();
        let query = PatternQuery::new("Test").with_limit(10);
        let config = RetrievalConfig::default();
        let mut hyper_config = HyperbolicRetrievalConfig::default();
        hyper_config.hierarchy_aware = false;

        let result = retrieve_patterns_hyperbolic(
            &patterns,
            &query_embedding.view(),
            &query,
            &config,
            &hyper_config,
        )
        .unwrap();

        // Should still return results, just without hierarchy penalty
        assert!(!result.is_empty());
    }

    #[test]
    fn test_hybrid_search_config_default() {
        let config = HybridSearchConfig::default();
        assert!((config.fts_weight - 0.3).abs() < f32::EPSILON);
        assert!((config.vector_weight - 0.7).abs() < f32::EPSILON);
        assert_eq!(config.fts_max_candidates, 50);
    }

    #[test]
    fn test_hybrid_search_weights_sum_to_one() {
        let config = HybridSearchConfig::default();
        let sum = config.fts_weight + config.vector_weight;
        assert!((sum - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_hybrid_search_config_with_weights() {
        let config = HybridSearchConfig::with_weights(0.5, 0.5);
        assert!((config.fts_weight - 0.5).abs() < f32::EPSILON);
        assert!((config.vector_weight - 0.5).abs() < f32::EPSILON);

        // Test clamping
        let config2 = HybridSearchConfig::with_weights(1.5, -0.5);
        assert!((config2.fts_weight - 1.0).abs() < f32::EPSILON);
        assert!((config2.vector_weight - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_hybrid_search_config_validate() {
        let valid = HybridSearchConfig::default();
        assert!(valid.validate().is_ok());

        let invalid = HybridSearchConfig {
            fts_weight: 0.0,
            vector_weight: 0.0,
            fts_max_candidates: 50,
        };
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn test_hybrid_search_fts_only() {
        let patterns = vec![
            create_test_pattern("p1", "Error handling", "rust", 0.8),
            create_test_pattern("p2", "Database connection", "database", 0.6),
        ];

        let query_embedding = create_test_embedding();
        let fts_results = vec![
            ("p1".to_string(), -5.0), // best BM25 match
            ("p2".to_string(), -2.0),
        ];

        let query = PatternQuery::new("Error handling").with_limit(10);
        let config = RetrievalConfig::default();
        let hybrid_config = HybridSearchConfig::with_weights(1.0, 0.0); // FTS only

        let result = retrieve_patterns_hybrid(
            &patterns,
            &query_embedding.view(),
            &fts_results,
            &query,
            &config,
            &hybrid_config,
        )
        .unwrap();

        assert!(!result.is_empty());
        assert_eq!(result.total_candidates, 2);
    }

    #[test]
    fn test_hybrid_search_vector_only() {
        let patterns = vec![
            create_test_pattern("p1", "Error handling", "rust", 0.8),
            create_test_pattern("p2", "Database connection", "database", 0.6),
        ];

        let query_embedding = create_test_embedding();
        let fts_results: Vec<(String, f64)> = vec![]; // No FTS results

        let query = PatternQuery::new("Error handling").with_limit(10);
        let config = RetrievalConfig::default();
        let hybrid_config = HybridSearchConfig::with_weights(0.0, 1.0); // Vector only

        let result = retrieve_patterns_hybrid(
            &patterns,
            &query_embedding.view(),
            &fts_results,
            &query,
            &config,
            &hybrid_config,
        )
        .unwrap();

        assert!(!result.is_empty());
        assert_eq!(result.total_candidates, 2);
    }

    #[test]
    fn test_hybrid_search_combined() {
        // Pattern p1: matches both FTS and vector (same embedding as query)
        // Pattern p2: matches vector only
        // Pattern p3: no embedding, matches FTS only
        let mut patterns = vec![
            create_test_pattern("p1", "Error handling", "rust", 0.8),
            create_test_pattern("p2", "Database connection", "database", 0.6),
        ];
        // p2 has a different embedding (lower cosine sim)
        patterns[1].embedding = Some(Array1::from_vec(vec![0.0, 1.0, 0.0, 0.0]));

        // Add a pattern with no embedding but found by FTS
        let mut p3 = Pattern::new("Async error", "Fix", "rust").with_reward(0.7);
        p3.id = "p3".to_string();
        p3.embedding = None; // no embedding
        patterns.push(p3);

        let query_embedding = create_test_embedding();
        let fts_results = vec![
            ("p1".to_string(), -10.0), // best BM25 match
            ("p3".to_string(), -5.0),  // moderate FTS match
        ];

        let query = PatternQuery::new("Error").with_limit(10);
        let config = RetrievalConfig {
            mmr: MmrConfig::disabled(),
            ..RetrievalConfig::default()
        };
        let hybrid_config = HybridSearchConfig::default(); // 0.3 FTS + 0.7 vector

        let result = retrieve_patterns_hybrid(
            &patterns,
            &query_embedding.view(),
            &fts_results,
            &query,
            &config,
            &hybrid_config,
        )
        .unwrap();

        // p1 should be ranked highest (has both FTS and high cosine sim)
        assert!(!result.is_empty());
        assert_eq!(result.patterns[0].pattern.id, "p1");
    }

    #[test]
    fn test_hybrid_search_with_domain_filter() {
        let patterns = vec![
            create_test_pattern("p1", "Rust error", "rust", 0.8),
            create_test_pattern("p2", "Python error", "python", 0.9),
            create_test_pattern("p3", "Rust async", "rust.async", 0.7),
        ];

        let query_embedding = create_test_embedding();
        let fts_results = vec![
            ("p1".to_string(), -5.0),
            ("p2".to_string(), -4.0),
            ("p3".to_string(), -3.0),
        ];

        let query = PatternQuery::new("Error")
            .with_domains(vec!["rust"])
            .with_limit(10);
        let config = RetrievalConfig::default();
        let hybrid_config = HybridSearchConfig::default();

        let result = retrieve_patterns_hybrid(
            &patterns,
            &query_embedding.view(),
            &fts_results,
            &query,
            &config,
            &hybrid_config,
        )
        .unwrap();

        // Should only return rust domain patterns (p1 and p3)
        assert_eq!(result.patterns.len(), 2);
        assert!(result
            .patterns
            .iter()
            .all(|p| p.pattern.domain.starts_with("rust")));
    }

    #[test]
    fn test_hybrid_search_with_reward_filter() {
        let patterns = vec![
            create_test_pattern("p1", "Low reward", "test", 0.3),
            create_test_pattern("p2", "High reward", "test", 0.9),
        ];

        let query_embedding = create_test_embedding();
        let fts_results = vec![
            ("p1".to_string(), -5.0),
            ("p2".to_string(), -3.0),
        ];

        let query = PatternQuery::new("Test").with_min_reward(0.5).with_limit(10);
        let config = RetrievalConfig::default();
        let hybrid_config = HybridSearchConfig::default();

        let result = retrieve_patterns_hybrid(
            &patterns,
            &query_embedding.view(),
            &fts_results,
            &query,
            &config,
            &hybrid_config,
        )
        .unwrap();

        // Should only include patterns with reward >= 0.5
        assert_eq!(result.patterns.len(), 1);
        assert!(result.patterns[0].pattern.reward >= 0.5);
    }

    #[test]
    fn test_retrieve_patterns_hyperbolic_pure_poincare() {
        let patterns = vec![
            create_test_pattern("p1", "Pattern A", "rust", 0.8),
            create_test_pattern("p2", "Pattern B", "database", 0.7),
        ];

        let query_embedding = create_test_embedding();
        let query = PatternQuery::new("Test").with_limit(10);
        let config = RetrievalConfig::default();
        let mut hyper_config = HyperbolicRetrievalConfig::default();
        hyper_config.poincare_weight = 1.0; // Pure Poincare

        let result = retrieve_patterns_hyperbolic(
            &patterns,
            &query_embedding.view(),
            &query,
            &config,
            &hyper_config,
        )
        .unwrap();

        assert!(!result.is_empty());
        // All final scores should be non-negative
        assert!(result.patterns.iter().all(|p| p.final_score >= 0.0));
    }
}
