//! Pattern retrieval benchmarks for Nagual
//!
//! Benchmarks for the ReasoningBank pattern retrieval pipeline including:
//! - Similarity search at various scales (100, 1000, 10000 patterns)
//! - MMR (Maximal Marginal Relevance) reranking
//! - Multi-factor scoring
//!
//! Performance targets:
//! - p95 < 50ms for 10,000 patterns
//! - Linear or sub-linear scaling with pattern count

use criterion::{
    black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput,
};
use ndarray::Array1;
use rand::prelude::*;
use std::hint::black_box as hint_black_box;

// Mock implementations for benchmarking without full database setup
// This allows us to benchmark the core algorithms in isolation

/// Generate a random normalized embedding vector
fn generate_random_embedding(dim: usize, rng: &mut StdRng) -> Vec<f32> {
    let mut embedding: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect();
    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > f32::EPSILON {
        embedding.iter_mut().for_each(|x| *x /= norm);
    }
    embedding
}

/// Generate test patterns with embeddings
fn generate_test_patterns(count: usize, embedding_dim: usize, seed: u64) -> Vec<TestPattern> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..count)
        .map(|i| TestPattern {
            id: format!("pattern-{}", i),
            embedding: generate_random_embedding(embedding_dim, &mut rng),
            reward: rng.gen_range(0.0..1.0),
            effectiveness: rng.gen_range(0.3..1.0),
            reuse_count: rng.gen_range(0..100),
            confidence: rng.gen_range(0.5..1.0),
            domain: format!("domain.sub{}", i % 10),
        })
        .collect()
}

/// Simplified pattern struct for benchmarking
#[derive(Clone)]
struct TestPattern {
    id: String,
    embedding: Vec<f32>,
    reward: f32,
    effectiveness: f32,
    reuse_count: u32,
    confidence: f32,
    domain: String,
}

/// Compute cosine similarity between two vectors
#[inline]
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a > f32::EPSILON && norm_b > f32::EPSILON {
        dot / (norm_a * norm_b)
    } else {
        0.0
    }
}

/// Cosine similarity for normalized vectors (just dot product)
#[inline]
fn cosine_similarity_normalized(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Benchmark pure similarity search (brute force)
fn brute_force_search(
    patterns: &[TestPattern],
    query: &[f32],
    k: usize,
    min_similarity: f32,
) -> Vec<(usize, f32)> {
    let mut scores: Vec<(usize, f32)> = patterns
        .iter()
        .enumerate()
        .map(|(i, p)| (i, cosine_similarity_normalized(query, &p.embedding)))
        .filter(|(_, sim)| *sim >= min_similarity)
        .collect();

    scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scores.truncate(k);
    scores
}

/// Benchmark similarity search with domain filtering
fn filtered_search(
    patterns: &[TestPattern],
    query: &[f32],
    k: usize,
    min_reward: f32,
    domain_prefix: Option<&str>,
) -> Vec<(usize, f32)> {
    let mut scores: Vec<(usize, f32)> = patterns
        .iter()
        .enumerate()
        .filter(|(_, p)| p.reward >= min_reward)
        .filter(|(_, p)| {
            domain_prefix
                .map(|prefix| p.domain.starts_with(prefix))
                .unwrap_or(true)
        })
        .map(|(i, p)| (i, cosine_similarity_normalized(query, &p.embedding)))
        .collect();

    scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scores.truncate(k);
    scores
}

/// Benchmark MMR (Maximal Marginal Relevance) selection
fn mmr_select(
    patterns: &[TestPattern],
    candidates: &[(usize, f32)],
    query: &[f32],
    k: usize,
    lambda: f32,
) -> Vec<(usize, f32)> {
    if candidates.is_empty() || k == 0 {
        return Vec::new();
    }

    let k = k.min(candidates.len());
    let mut selected: Vec<(usize, f32)> = Vec::with_capacity(k);
    let mut remaining: Vec<(usize, f32)> = candidates.to_vec();

    // Select first by highest similarity
    if let Some((pos, _)) = remaining
        .iter()
        .enumerate()
        .max_by(|a, b| a.1 .1.partial_cmp(&b.1 .1).unwrap())
    {
        selected.push(remaining.remove(pos));
    }

    // Iteratively select using MMR
    while selected.len() < k && !remaining.is_empty() {
        let mut best_idx = 0;
        let mut best_mmr = f32::NEG_INFINITY;

        for (i, &(pattern_idx, similarity)) in remaining.iter().enumerate() {
            // Max similarity to selected patterns
            let max_selected_sim = selected
                .iter()
                .map(|&(sel_idx, _)| {
                    cosine_similarity_normalized(
                        &patterns[pattern_idx].embedding,
                        &patterns[sel_idx].embedding,
                    )
                })
                .fold(f32::NEG_INFINITY, f32::max);

            // MMR score
            let mmr = lambda * similarity - (1.0 - lambda) * max_selected_sim;

            if mmr > best_mmr {
                best_mmr = mmr;
                best_idx = i;
            }
        }

        selected.push(remaining.remove(best_idx));
    }

    selected
}

/// Benchmark multi-factor scoring
fn multi_factor_score(
    patterns: &[TestPattern],
    candidates: &[(usize, f32)],
    weights: &ScoringWeights,
) -> Vec<(usize, f32, f32)> {
    let mut scored: Vec<(usize, f32, f32)> = candidates
        .iter()
        .map(|&(idx, similarity)| {
            let pattern = &patterns[idx];

            // Recency score (simplified - assume all patterns are recent)
            let recency = 0.9;

            // Reliability score
            let reliability = pattern.effectiveness * 0.6
                + pattern.confidence * 0.3
                + 0.1; // success bonus

            // Reuse score (log scale)
            let reuse = if pattern.reuse_count > 0 {
                (pattern.reuse_count as f32).ln() / (100.0_f32).ln()
            } else {
                0.0
            }
            .min(1.0);

            // Combined score
            let final_score = weights.similarity * similarity
                + weights.recency * recency
                + weights.reliability * reliability
                + weights.reuse * reuse
                + weights.reward * pattern.reward;

            (idx, similarity, final_score)
        })
        .collect();

    scored.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
    scored
}

/// Scoring weights for multi-factor ranking
#[derive(Clone)]
struct ScoringWeights {
    similarity: f32,
    recency: f32,
    reliability: f32,
    reuse: f32,
    reward: f32,
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

/// Full retrieval pipeline benchmark
fn full_retrieval_pipeline(
    patterns: &[TestPattern],
    query: &[f32],
    k: usize,
    min_reward: f32,
    mmr_lambda: f32,
    weights: &ScoringWeights,
) -> Vec<(usize, f32)> {
    // Step 1: Filter and search
    let candidates = filtered_search(patterns, query, k * 3, min_reward, None);

    // Step 2: MMR reranking
    let diverse = mmr_select(patterns, &candidates, query, k * 2, mmr_lambda);

    // Step 3: Multi-factor scoring
    let scored = multi_factor_score(patterns, &diverse, weights);

    // Step 4: Return top k
    scored
        .into_iter()
        .take(k)
        .map(|(idx, sim, _)| (idx, sim))
        .collect()
}

// Benchmark functions

fn bench_similarity_search(c: &mut Criterion) {
    let embedding_dim = 128;
    let k = 10;

    let mut group = c.benchmark_group("similarity_search");
    group.sample_size(50);

    for pattern_count in [100, 1000, 10000].iter() {
        let patterns = generate_test_patterns(*pattern_count, embedding_dim, 42);
        let mut rng = StdRng::seed_from_u64(123);
        let query = generate_random_embedding(embedding_dim, &mut rng);

        group.throughput(Throughput::Elements(*pattern_count as u64));
        group.bench_with_input(
            BenchmarkId::new("brute_force", pattern_count),
            pattern_count,
            |b, _| {
                b.iter(|| {
                    black_box(brute_force_search(&patterns, &query, k, 0.0))
                })
            },
        );
    }

    group.finish();
}

fn bench_filtered_search(c: &mut Criterion) {
    let embedding_dim = 128;
    let k = 10;
    let patterns = generate_test_patterns(10000, embedding_dim, 42);
    let mut rng = StdRng::seed_from_u64(123);
    let query = generate_random_embedding(embedding_dim, &mut rng);

    let mut group = c.benchmark_group("filtered_search");
    group.sample_size(50);

    // Benchmark with different filter selectivities
    for (min_reward, domain_filter) in [
        (0.0, None),
        (0.5, None),
        (0.0, Some("domain.sub0")),
        (0.5, Some("domain.sub0")),
    ] {
        let filter_desc = format!(
            "reward>{}_domain={}",
            min_reward,
            domain_filter.unwrap_or("any")
        );

        group.bench_with_input(
            BenchmarkId::new("10k_patterns", filter_desc),
            &(min_reward, domain_filter),
            |b, &(min_r, dom)| {
                b.iter(|| {
                    black_box(filtered_search(&patterns, &query, k, min_r, dom))
                })
            },
        );
    }

    group.finish();
}

fn bench_mmr_reranking(c: &mut Criterion) {
    let embedding_dim = 128;
    let patterns = generate_test_patterns(1000, embedding_dim, 42);
    let mut rng = StdRng::seed_from_u64(123);
    let query = generate_random_embedding(embedding_dim, &mut rng);

    // Pre-compute candidates
    let candidates = brute_force_search(&patterns, &query, 100, 0.0);

    let mut group = c.benchmark_group("mmr_reranking");
    group.sample_size(50);

    // Benchmark different k values and lambda
    for k in [5, 10, 20, 50].iter() {
        for lambda in [0.5, 0.7, 0.9].iter() {
            group.bench_with_input(
                BenchmarkId::new(format!("k={}_lambda={}", k, lambda), k),
                &(*k, *lambda),
                |b, &(k_val, lambda_val)| {
                    b.iter(|| {
                        black_box(mmr_select(&patterns, &candidates, &query, k_val, lambda_val))
                    })
                },
            );
        }
    }

    group.finish();
}

fn bench_multi_factor_scoring(c: &mut Criterion) {
    let embedding_dim = 128;
    let patterns = generate_test_patterns(1000, embedding_dim, 42);
    let mut rng = StdRng::seed_from_u64(123);
    let query = generate_random_embedding(embedding_dim, &mut rng);
    let weights = ScoringWeights::default();

    // Pre-compute candidates at different sizes
    let candidates_50 = brute_force_search(&patterns, &query, 50, 0.0);
    let candidates_100 = brute_force_search(&patterns, &query, 100, 0.0);
    let candidates_200 = brute_force_search(&patterns, &query, 200, 0.0);

    let mut group = c.benchmark_group("multi_factor_scoring");
    group.sample_size(50);

    group.bench_function("50_candidates", |b| {
        b.iter(|| black_box(multi_factor_score(&patterns, &candidates_50, &weights)))
    });

    group.bench_function("100_candidates", |b| {
        b.iter(|| black_box(multi_factor_score(&patterns, &candidates_100, &weights)))
    });

    group.bench_function("200_candidates", |b| {
        b.iter(|| black_box(multi_factor_score(&patterns, &candidates_200, &weights)))
    });

    group.finish();
}

fn bench_full_pipeline(c: &mut Criterion) {
    let embedding_dim = 128;
    let k = 10;
    let weights = ScoringWeights::default();

    let mut group = c.benchmark_group("full_retrieval_pipeline");
    group.sample_size(30);

    // Performance targets:
    // - 100 patterns: < 1ms
    // - 1000 patterns: < 5ms
    // - 10000 patterns: < 50ms (p95)

    for pattern_count in [100, 1000, 10000].iter() {
        let patterns = generate_test_patterns(*pattern_count, embedding_dim, 42);
        let mut rng = StdRng::seed_from_u64(123);
        let query = generate_random_embedding(embedding_dim, &mut rng);

        group.throughput(Throughput::Elements(*pattern_count as u64));
        group.bench_with_input(
            BenchmarkId::new("patterns", pattern_count),
            pattern_count,
            |b, _| {
                b.iter(|| {
                    black_box(full_retrieval_pipeline(
                        &patterns,
                        &query,
                        k,
                        0.5,
                        0.7,
                        &weights,
                    ))
                })
            },
        );
    }

    group.finish();
}

fn bench_cosine_similarity(c: &mut Criterion) {
    let mut group = c.benchmark_group("cosine_similarity");
    group.sample_size(100);

    let mut rng = StdRng::seed_from_u64(42);

    // Benchmark different embedding dimensions
    for dim in [128, 256, 384, 768].iter() {
        let a = generate_random_embedding(*dim, &mut rng);
        let b = generate_random_embedding(*dim, &mut rng);

        group.throughput(Throughput::Elements(*dim as u64));

        group.bench_with_input(
            BenchmarkId::new("unnormalized", dim),
            dim,
            |bench, _| {
                bench.iter(|| black_box(cosine_similarity(&a, &b)))
            },
        );

        group.bench_with_input(
            BenchmarkId::new("normalized", dim),
            dim,
            |bench, _| {
                bench.iter(|| black_box(cosine_similarity_normalized(&a, &b)))
            },
        );
    }

    group.finish();
}

fn bench_scaling_characteristics(c: &mut Criterion) {
    let embedding_dim = 128;
    let k = 10;

    let mut group = c.benchmark_group("scaling");
    group.sample_size(20);

    // Test scaling from 100 to 50000 patterns
    for &pattern_count in &[100, 500, 1000, 2500, 5000, 10000, 25000, 50000] {
        let patterns = generate_test_patterns(pattern_count, embedding_dim, 42);
        let mut rng = StdRng::seed_from_u64(123);
        let query = generate_random_embedding(embedding_dim, &mut rng);

        group.throughput(Throughput::Elements(pattern_count as u64));
        group.bench_with_input(
            BenchmarkId::new("brute_force", pattern_count),
            &pattern_count,
            |b, _| {
                b.iter(|| black_box(brute_force_search(&patterns, &query, k, 0.0)))
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_cosine_similarity,
    bench_similarity_search,
    bench_filtered_search,
    bench_mmr_reranking,
    bench_multi_factor_scoring,
    bench_full_pipeline,
    bench_scaling_characteristics,
);

criterion_main!(benches);
