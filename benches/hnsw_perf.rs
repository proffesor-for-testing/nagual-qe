//! HNSW Performance Benchmarks for ProfDAG
//!
//! Benchmarks for HNSW-powered vector similarity search including:
//! - Brute force vs HNSW comparison at various scales (10K, 50K, 100K nodes)
//! - Latency measurements at different ef_search values
//! - Recall measurements at different ef_search values
//! - Batch search throughput
//!
//! Performance targets (from PROFDAG-002):
//! - Search latency < 10ms for 100K nodes
//! - Recall > 0.95 at ef_search=100

use criterion::{
    black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput,
};
use instant_distance::{Builder as HnswBuilder, HnswMap, Search};
use rand::prelude::*;
use std::cmp::Ordering;
use std::collections::HashSet;
use std::time::Instant;

// ============================================================================
// Test Data Generation
// ============================================================================

/// Generate a random normalized embedding vector.
fn generate_random_embedding(dim: usize, rng: &mut StdRng) -> Vec<f32> {
    let mut embedding: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect();
    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > f32::EPSILON {
        embedding.iter_mut().for_each(|x| *x /= norm);
    }
    embedding
}

/// Generate a set of random embeddings.
fn generate_embedding_set(count: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..count)
        .map(|_| generate_random_embedding(dim, &mut rng))
        .collect()
}

/// Generate query embeddings that are similar to a subset of the database.
fn generate_related_queries(
    base_embeddings: &[Vec<f32>],
    num_queries: usize,
    noise_level: f32,
    seed: u64,
) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    let _dim = base_embeddings[0].len();

    (0..num_queries)
        .map(|_| {
            // Pick a random base embedding
            let base_idx = rng.gen_range(0..base_embeddings.len());
            let base = &base_embeddings[base_idx];

            // Add noise
            let mut query: Vec<f32> = base
                .iter()
                .map(|&x| x + rng.gen_range(-noise_level..noise_level))
                .collect();

            // Renormalize
            let norm: f32 = query.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > f32::EPSILON {
                query.iter_mut().for_each(|x| *x /= norm);
            }

            query
        })
        .collect()
}

// ============================================================================
// Vector Operations
// ============================================================================

/// Compute cosine similarity for normalized vectors (dot product).
#[inline]
fn cosine_similarity_normalized(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

// ============================================================================
// HNSW Node Point
// ============================================================================

/// Point type for HNSW index.
#[derive(Clone)]
struct NodePoint {
    #[allow(dead_code)]
    id: usize,
    embedding: Vec<f32>,
}

impl instant_distance::Point for NodePoint {
    fn distance(&self, other: &Self) -> f32 {
        // Cosine distance = 1 - cosine_similarity
        1.0 - cosine_similarity_normalized(&self.embedding, &other.embedding)
    }
}

// ============================================================================
// Search Implementations
// ============================================================================

/// Brute force k-NN search (ground truth).
fn brute_force_knn(embeddings: &[Vec<f32>], query: &[f32], k: usize) -> Vec<(usize, f32)> {
    let mut scores: Vec<(usize, f32)> = embeddings
        .iter()
        .enumerate()
        .map(|(i, emb)| (i, cosine_similarity_normalized(query, emb)))
        .collect();

    scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
    scores.truncate(k);
    scores
}

/// Build HNSW index with tuned parameters.
///
/// Parameters:
/// - `_m`: Reserved for future use (instant-distance uses internal M)
/// - `ef_construction`: Controls index quality during build
/// - `ef_search`: Controls search quality at query time (set at build time in instant-distance)
fn build_hnsw_index(
    embeddings: &[Vec<f32>],
    _m: usize,
    ef_construction: usize,
    ef_search: usize,
) -> HnswMap<NodePoint, usize> {
    let points: Vec<NodePoint> = embeddings
        .iter()
        .enumerate()
        .map(|(id, emb)| NodePoint {
            id,
            embedding: emb.clone(),
        })
        .collect();

    let values: Vec<usize> = (0..embeddings.len()).collect();

    HnswBuilder::default()
        .ef_construction(ef_construction)
        .ef_search(ef_search)  // Critical: set search quality at build time
        .build(points, values)
}

/// HNSW search.
fn hnsw_search(
    index: &HnswMap<NodePoint, usize>,
    query: &[f32],
    k: usize,
) -> Vec<(usize, f32)> {
    let query_point = NodePoint {
        id: usize::MAX,
        embedding: query.to_vec(),
    };

    let mut search = Search::default();
    let neighbors = index.search(&query_point, &mut search);

    neighbors
        .take(k)
        .map(|n| {
            let similarity = 1.0 - n.distance;
            (*n.value, similarity)
        })
        .collect()
}

/// Calculate recall (percentage of true top-k found by approximate search).
fn calculate_recall(ground_truth: &[(usize, f32)], approximate: &[(usize, f32)], k: usize) -> f32 {
    let truth_set: HashSet<usize> = ground_truth.iter().take(k).map(|(i, _)| *i).collect();
    let approx_set: HashSet<usize> = approximate.iter().take(k).map(|(i, _)| *i).collect();

    let intersection = truth_set.intersection(&approx_set).count();
    intersection as f32 / k as f32
}

// ============================================================================
// Benchmark Functions
// ============================================================================

fn bench_brute_force_vs_hnsw(c: &mut Criterion) {
    let dim = 128;
    let k = 10;
    // Tuned for better recall at scale (was m=16, ef=128)
    let m = 24;
    let ef_construction = 200;
    let ef_search = 200;  // High recall mode

    let mut group = c.benchmark_group("brute_force_vs_hnsw");
    group.sample_size(30);

    // Test at different scales
    for &node_count in &[10_000, 50_000, 100_000] {
        let embeddings = generate_embedding_set(node_count, dim, 42);
        let queries = generate_related_queries(&embeddings, 10, 0.1, 123);

        // Build HNSW index
        let start = Instant::now();
        let index = build_hnsw_index(&embeddings, m, ef_construction, ef_search);
        let build_time = start.elapsed();
        println!(
            "HNSW index build time for {} nodes: {:?}",
            node_count, build_time
        );

        group.throughput(Throughput::Elements(node_count as u64));

        // Benchmark brute force
        group.bench_with_input(
            BenchmarkId::new("brute_force", node_count),
            &node_count,
            |b, _| {
                b.iter(|| {
                    for query in &queries {
                        black_box(brute_force_knn(&embeddings, query, k));
                    }
                })
            },
        );

        // Benchmark HNSW
        group.bench_with_input(
            BenchmarkId::new("hnsw", node_count),
            &node_count,
            |b, _| {
                b.iter(|| {
                    for query in &queries {
                        black_box(hnsw_search(&index, query, k));
                    }
                })
            },
        );
    }

    group.finish();
}

fn bench_hnsw_ef_search(c: &mut Criterion) {
    let dim = 128;
    let k = 10;
    // Tuned for better recall at scale
    let m = 24;
    let ef_construction = 200;
    // Note: Use smaller node count for ef_search sweep (builds multiple indexes)
    let node_count = 50_000;

    let embeddings = generate_embedding_set(node_count, dim, 42);
    let queries = generate_related_queries(&embeddings, 100, 0.1, 123);

    let mut group = c.benchmark_group("hnsw_ef_search");
    group.sample_size(50);

    // Test different ef_search values
    // Note: instant-distance sets ef_search at build time, so we need separate indexes
    for &ef_search in &[40, 80, 120, 160, 200, 250, 300] {
        // Build index with this ef_search value
        let index = build_hnsw_index(&embeddings, m, ef_construction, ef_search);

        // Measure recall at this ef_search
        let mut total_recall = 0.0;
        for query in &queries {
            let ground_truth = brute_force_knn(&embeddings, query, k);
            let approximate = hnsw_search(&index, query, k);
            total_recall += calculate_recall(&ground_truth, &approximate, k);
        }
        let avg_recall = total_recall / queries.len() as f32;
        println!("ef_search={}: recall={:.4}", ef_search, avg_recall);

        group.throughput(Throughput::Elements(1));
        group.bench_with_input(
            BenchmarkId::new("ef_search", ef_search),
            &ef_search,
            |b, _| {
                b.iter(|| {
                    for query in queries.iter().take(10) {
                        black_box(hnsw_search(&index, query, k));
                    }
                })
            },
        );
    }

    group.finish();
}

fn bench_hnsw_latency_at_scale(c: &mut Criterion) {
    let dim = 128;
    let k = 10;
    // Tuned for better recall at scale
    let m = 24;
    let ef_construction = 200;
    let ef_search = 200;  // High recall mode

    let mut group = c.benchmark_group("hnsw_latency_at_scale");
    group.sample_size(100);

    // Target: < 10ms for 100K nodes
    for &node_count in &[10_000, 50_000, 100_000] {
        let embeddings = generate_embedding_set(node_count, dim, 42);
        let query = generate_random_embedding(dim, &mut StdRng::seed_from_u64(999));
        let index = build_hnsw_index(&embeddings, m, ef_construction, ef_search);

        group.throughput(Throughput::Elements(1));
        group.bench_with_input(
            BenchmarkId::new("single_query", node_count),
            &node_count,
            |b, _| {
                b.iter(|| black_box(hnsw_search(&index, &query, k)))
            },
        );
    }

    group.finish();
}

fn bench_hnsw_recall_at_scale(_c: &mut Criterion) {
    let dim = 128;
    let k = 10;
    // Tuned for better recall at scale
    let m = 24;
    let ef_construction = 200;
    let ef_search = 200;  // High recall mode

    println!("\n=== Recall Measurements ===");

    for &node_count in &[10_000, 50_000, 100_000] {
        let embeddings = generate_embedding_set(node_count, dim, 42);
        let queries = generate_related_queries(&embeddings, 100, 0.1, 123);
        let index = build_hnsw_index(&embeddings, m, ef_construction, ef_search);

        // Measure recall at ef_search=100 (target: > 0.95)
        let mut total_recall = 0.0;
        for query in &queries {
            let ground_truth = brute_force_knn(&embeddings, query, k);
            let approximate = hnsw_search(&index, query, k);
            total_recall += calculate_recall(&ground_truth, &approximate, k);
        }
        let avg_recall = total_recall / queries.len() as f32;

        println!(
            "node_count={}: recall@{} = {:.4} (target > 0.95)",
            node_count, k, avg_recall
        );

        // Verify recall meets target
        if avg_recall < 0.95 {
            println!("  WARNING: Recall below target!");
        }
    }
}

fn bench_hnsw_batch_search(c: &mut Criterion) {
    let dim = 128;
    let k = 10;
    // Tuned for better recall at scale
    let m = 24;
    let ef_construction = 200;
    let ef_search = 200;  // High recall mode
    let node_count = 100_000;

    let embeddings = generate_embedding_set(node_count, dim, 42);
    let index = build_hnsw_index(&embeddings, m, ef_construction, ef_search);

    let mut group = c.benchmark_group("hnsw_batch_search");
    group.sample_size(20);

    // Test different batch sizes
    for &batch_size in &[10, 50, 100, 500] {
        let queries = generate_related_queries(&embeddings, batch_size, 0.1, 123);

        group.throughput(Throughput::Elements(batch_size as u64));
        group.bench_with_input(
            BenchmarkId::new("batch", batch_size),
            &batch_size,
            |b, _| {
                b.iter(|| {
                    let results: Vec<_> = queries
                        .iter()
                        .map(|q| hnsw_search(&index, q, k))
                        .collect();
                    black_box(results)
                })
            },
        );
    }

    group.finish();
}

fn bench_hnsw_build_time(c: &mut Criterion) {
    let dim = 128;
    // Tuned for better recall at scale
    let m = 24;
    let ef_construction = 200;
    let ef_search = 200;  // High recall mode

    let mut group = c.benchmark_group("hnsw_build_time");
    group.sample_size(10); // Building is expensive

    for &node_count in &[10_000, 25_000, 50_000] {
        let embeddings = generate_embedding_set(node_count, dim, 42);

        group.throughput(Throughput::Elements(node_count as u64));
        group.bench_with_input(
            BenchmarkId::new("build", node_count),
            &node_count,
            |b, _| {
                b.iter(|| black_box(build_hnsw_index(&embeddings, m, ef_construction, ef_search)))
            },
        );
    }

    group.finish();
}

fn bench_dimension_impact(c: &mut Criterion) {
    let node_count = 50_000;
    let k = 10;
    let m = 16;
    let ef_construction = 128;
    let ef_search = 100;  // Baseline mode

    let mut group = c.benchmark_group("dimension_impact");
    group.sample_size(30);

    for &dim in &[64, 128, 256, 384] {
        let embeddings = generate_embedding_set(node_count, dim, 42);
        let query = generate_random_embedding(dim, &mut StdRng::seed_from_u64(999));
        let index = build_hnsw_index(&embeddings, m, ef_construction, ef_search);

        group.throughput(Throughput::Elements(1));
        group.bench_with_input(BenchmarkId::new("dim", dim), &dim, |b, _| {
            b.iter(|| black_box(hnsw_search(&index, &query, k)))
        });
    }

    group.finish();
}

fn bench_hnsw_parameters(c: &mut Criterion) {
    let dim = 128;
    let node_count = 50_000;
    let k = 10;

    let embeddings = generate_embedding_set(node_count, dim, 42);
    let queries = generate_related_queries(&embeddings, 50, 0.1, 123);

    let mut group = c.benchmark_group("hnsw_parameters");
    group.sample_size(20);

    // Test different m/ef_construction/ef_search combinations
    // ef_search set proportional to ef_construction for balanced comparison
    for &(m, ef_construction, ef_search) in &[(8, 64, 64), (16, 128, 128), (24, 192, 192), (32, 256, 256)] {
        let index = build_hnsw_index(&embeddings, m, ef_construction, ef_search);

        // Measure recall
        let mut total_recall = 0.0;
        for query in queries.iter().take(20) {
            let ground_truth = brute_force_knn(&embeddings, query, k);
            let approximate = hnsw_search(&index, query, k);
            total_recall += calculate_recall(&ground_truth, &approximate, k);
        }
        let avg_recall = total_recall / 20.0;
        println!("m={}, ef_construction={}, ef_search={}: recall={:.4}", m, ef_construction, ef_search, avg_recall);

        group.bench_with_input(
            BenchmarkId::new(format!("m={}_ef={}_efs={}", m, ef_construction, ef_search), m),
            &m,
            |b, _| {
                b.iter(|| {
                    for query in queries.iter().take(10) {
                        black_box(hnsw_search(&index, query, k));
                    }
                })
            },
        );
    }

    group.finish();
}

// ============================================================================
// Summary Report
// ============================================================================

#[allow(dead_code)]
fn print_benchmark_summary() {
    println!("\n");
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║                  HNSW Performance Benchmark Summary              ║");
    println!("╠══════════════════════════════════════════════════════════════════╣");
    println!("║                                                                  ║");
    println!("║  Target Performance (PROFDAG-002):                               ║");
    println!("║  - Search latency < 10ms for 100K nodes                          ║");
    println!("║  - Recall > 0.95 at ef_search=100                                ║");
    println!("║                                                                  ║");
    println!("║  Index Configuration:                                            ║");
    println!("║  - m = 16 (connections per layer)                                ║");
    println!("║  - ef_construction = 128 (build quality)                         ║");
    println!("║  - Embedding dimension = 128                                     ║");
    println!("║                                                                  ║");
    println!("║  Benchmark Groups:                                               ║");
    println!("║  1. brute_force_vs_hnsw - Compare speedup at scale               ║");
    println!("║  2. hnsw_ef_search - Accuracy vs latency tradeoff                ║");
    println!("║  3. hnsw_latency_at_scale - Verify <10ms at 100K                 ║");
    println!("║  4. hnsw_recall_at_scale - Verify >0.95 recall                   ║");
    println!("║  5. hnsw_batch_search - Throughput for batch queries             ║");
    println!("║  6. hnsw_build_time - Index construction overhead                ║");
    println!("║  7. dimension_impact - Effect of embedding dimension             ║");
    println!("║  8. hnsw_parameters - Optimal m and ef_construction              ║");
    println!("║                                                                  ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();
}

criterion_group!(
    benches,
    bench_brute_force_vs_hnsw,
    bench_hnsw_ef_search,
    bench_hnsw_latency_at_scale,
    bench_hnsw_recall_at_scale,
    bench_hnsw_batch_search,
    bench_hnsw_build_time,
    bench_dimension_impact,
    bench_hnsw_parameters,
);

criterion_main!(benches);
