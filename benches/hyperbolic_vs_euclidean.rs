//! Benchmark: Hyperbolic (Poincare) vs Euclidean Distance and KNN Search
//!
//! Compares performance of:
//! - Euclidean cosine_similarity vs poincare_distance
//! - PoincareKNN.search vs brute-force Poincare distance
//! - Scaling behavior at 1K, 10K, and 50K points
//!
//! Run with: `cargo bench --bench hyperbolic_vs_euclidean`

use criterion::{
    black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput,
};
use rand::prelude::*;
use std::cmp::Ordering;

// ============================================================================
// Test Data Generation
// ============================================================================

/// Generate a random point inside the Poincare ball.
///
/// Points are uniformly distributed by radius (not by volume) to get
/// a mix of general (near-origin) and specific (near-boundary) points.
fn generate_poincare_point(dim: usize, max_norm: f64, rng: &mut StdRng) -> Vec<f64> {
    // Random direction
    let mut direction: Vec<f64> = (0..dim).map(|_| rng.gen::<f64>() * 2.0 - 1.0).collect();
    let norm: f64 = direction.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm > f64::EPSILON {
        direction.iter_mut().for_each(|x| *x /= norm);
    }

    // Random radius in [0, max_norm)
    let radius = rng.gen::<f64>() * max_norm;
    direction.iter_mut().for_each(|x| *x *= radius);

    direction
}

/// Generate a set of random Poincare ball points.
fn generate_poincare_set(count: usize, dim: usize, max_norm: f64, seed: u64) -> Vec<Vec<f64>> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..count)
        .map(|_| generate_poincare_point(dim, max_norm, &mut rng))
        .collect()
}

/// Generate a random normalized Euclidean embedding.
fn generate_euclidean_embedding(dim: usize, rng: &mut StdRng) -> Vec<f32> {
    let mut emb: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect();
    let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > f32::EPSILON {
        emb.iter_mut().for_each(|x| *x /= norm);
    }
    emb
}

/// Generate a set of random Euclidean embeddings.
fn generate_euclidean_set(count: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..count)
        .map(|_| generate_euclidean_embedding(dim, &mut rng))
        .collect()
}

// ============================================================================
// Distance Functions
// ============================================================================

/// Euclidean cosine similarity for normalized vectors.
#[inline]
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Poincare distance between two points in the ball.
#[inline]
fn poincare_dist(u: &[f64], v: &[f64]) -> f64 {
    let diff_sq: f64 = u.iter().zip(v.iter()).map(|(a, b)| (a - b) * (a - b)).sum();
    let u_sq: f64 = u.iter().map(|x| x * x).sum();
    let v_sq: f64 = v.iter().map(|x| x * x).sum();

    let denom = (1.0 - u_sq) * (1.0 - v_sq);
    if denom < 1e-15 {
        return f64::MAX;
    }

    let arg = 1.0 + 2.0 * diff_sq / denom;
    arg.max(1.0).acosh()
}

/// Brute-force KNN using Poincare distance.
fn brute_force_poincare_knn(
    points: &[Vec<f64>],
    query: &[f64],
    k: usize,
) -> Vec<(usize, f64)> {
    let mut distances: Vec<(usize, f64)> = points
        .iter()
        .enumerate()
        .map(|(i, p)| (i, poincare_dist(query, p)))
        .collect();

    distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
    distances.truncate(k);
    distances
}

/// Brute-force KNN using Euclidean cosine similarity (returns as distance = 1 - similarity).
fn brute_force_euclidean_knn(
    points: &[Vec<f32>],
    query: &[f32],
    k: usize,
) -> Vec<(usize, f32)> {
    let mut similarities: Vec<(usize, f32)> = points
        .iter()
        .enumerate()
        .map(|(i, p)| (i, cosine_similarity(query, p)))
        .collect();

    // Sort by descending similarity (highest first = nearest)
    similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
    similarities.truncate(k);
    similarities
}

// ============================================================================
// Benchmark: Single Distance Computation
// ============================================================================

fn bench_distance_computation(c: &mut Criterion) {
    let dim = 128;
    let mut rng = StdRng::seed_from_u64(42);

    // Generate test pairs
    let euclidean_a = generate_euclidean_embedding(dim, &mut rng);
    let euclidean_b = generate_euclidean_embedding(dim, &mut rng);
    let poincare_a = generate_poincare_point(dim, 0.99, &mut rng);
    let poincare_b = generate_poincare_point(dim, 0.99, &mut rng);

    let mut group = c.benchmark_group("distance_computation");
    group.sample_size(1000);

    group.bench_function("euclidean_cosine_similarity", |b| {
        b.iter(|| black_box(cosine_similarity(&euclidean_a, &euclidean_b)))
    });

    group.bench_function("poincare_distance", |b| {
        b.iter(|| black_box(poincare_dist(&poincare_a, &poincare_b)))
    });

    group.finish();
}

// ============================================================================
// Benchmark: Batch Distance Computation (100 pairs)
// ============================================================================

fn bench_batch_distance(c: &mut Criterion) {
    let dim = 128;
    let num_pairs = 100;
    let mut rng = StdRng::seed_from_u64(42);

    let euclidean_pairs: Vec<(Vec<f32>, Vec<f32>)> = (0..num_pairs)
        .map(|_| {
            let a = generate_euclidean_embedding(dim, &mut rng);
            let b = generate_euclidean_embedding(dim, &mut rng);
            (a, b)
        })
        .collect();

    let poincare_pairs: Vec<(Vec<f64>, Vec<f64>)> = (0..num_pairs)
        .map(|_| {
            let a = generate_poincare_point(dim, 0.99, &mut rng);
            let b = generate_poincare_point(dim, 0.99, &mut rng);
            (a, b)
        })
        .collect();

    let mut group = c.benchmark_group("batch_distance_100_pairs");
    group.sample_size(200);

    group.bench_function("euclidean_cosine_batch", |b| {
        b.iter(|| {
            let total: f32 = euclidean_pairs
                .iter()
                .map(|(a, b)| cosine_similarity(a, b))
                .sum();
            black_box(total)
        })
    });

    group.bench_function("poincare_distance_batch", |b| {
        b.iter(|| {
            let total: f64 = poincare_pairs
                .iter()
                .map(|(a, b)| poincare_dist(a, b))
                .sum();
            black_box(total)
        })
    });

    group.finish();
}

// ============================================================================
// Benchmark: KNN Search at Scale
// ============================================================================

fn bench_knn_search_at_scale(c: &mut Criterion) {
    let dim = 128;
    let k = 10;
    let max_norm = 0.99;
    let num_queries = 10;

    let mut group = c.benchmark_group("knn_search_at_scale");
    group.sample_size(30);

    for &point_count in &[1_000, 10_000, 50_000] {
        let poincare_points = generate_poincare_set(point_count, dim, max_norm, 42);
        let euclidean_points = generate_euclidean_set(point_count, dim, 42);

        let mut rng = StdRng::seed_from_u64(123);
        let poincare_queries: Vec<Vec<f64>> = (0..num_queries)
            .map(|_| generate_poincare_point(dim, max_norm, &mut rng))
            .collect();
        let euclidean_queries: Vec<Vec<f32>> = (0..num_queries)
            .map(|_| generate_euclidean_embedding(dim, &mut rng))
            .collect();

        group.throughput(Throughput::Elements(point_count as u64));

        // Euclidean brute-force KNN
        group.bench_with_input(
            BenchmarkId::new("euclidean_brute_force", point_count),
            &point_count,
            |b, _| {
                b.iter(|| {
                    for query in &euclidean_queries {
                        black_box(brute_force_euclidean_knn(&euclidean_points, query, k));
                    }
                })
            },
        );

        // Poincare brute-force KNN
        group.bench_with_input(
            BenchmarkId::new("poincare_brute_force", point_count),
            &point_count,
            |b, _| {
                b.iter(|| {
                    for query in &poincare_queries {
                        black_box(brute_force_poincare_knn(&poincare_points, query, k));
                    }
                })
            },
        );
    }

    group.finish();
}

// ============================================================================
// Benchmark: PoincareKNN Index vs Brute Force
// ============================================================================

fn bench_poincare_knn_index(c: &mut Criterion) {
    let dim = 128;
    let k = 10;
    let max_norm = 0.99;
    let num_queries = 10;

    let mut group = c.benchmark_group("poincare_knn_index_vs_brute_force");
    group.sample_size(30);

    for &point_count in &[1_000, 10_000, 50_000] {
        let points = generate_poincare_set(point_count, dim, max_norm, 42);

        let mut rng = StdRng::seed_from_u64(123);
        let queries: Vec<Vec<f64>> = (0..num_queries)
            .map(|_| generate_poincare_point(dim, max_norm, &mut rng))
            .collect();

        // Build PoincareKNN index (from the library, but we inline a similar approach)
        // Since we cannot import the library crate from a bench, we use inline brute force
        // with the same heap approach to measure algorithmic overhead.
        let indexed_points: Vec<(usize, Vec<f64>)> = points
            .iter()
            .enumerate()
            .map(|(i, p)| (i, p.clone()))
            .collect();

        group.throughput(Throughput::Elements(point_count as u64));

        // Brute force with sort
        group.bench_with_input(
            BenchmarkId::new("brute_force_sort", point_count),
            &point_count,
            |b, _| {
                b.iter(|| {
                    for query in &queries {
                        black_box(brute_force_poincare_knn(&points, query, k));
                    }
                })
            },
        );

        // Brute force with max-heap (same approach as PoincareKNN)
        group.bench_with_input(
            BenchmarkId::new("brute_force_heap", point_count),
            &point_count,
            |b, _| {
                b.iter(|| {
                    for query in &queries {
                        black_box(heap_knn(&indexed_points, query, k));
                    }
                })
            },
        );
    }

    group.finish();
}

/// Heap-based KNN (same algorithm as PoincareKNN::search).
fn heap_knn(points: &[(usize, Vec<f64>)], query: &[f64], k: usize) -> Vec<(usize, f64)> {
    use std::collections::BinaryHeap;

    #[derive(PartialEq)]
    struct Entry {
        id: usize,
        dist: f64,
    }

    impl Eq for Entry {}

    impl PartialOrd for Entry {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }

    impl Ord for Entry {
        fn cmp(&self, other: &Self) -> Ordering {
            // Max-heap: reverse order
            other
                .dist
                .partial_cmp(&self.dist)
                .unwrap_or(Ordering::Equal)
        }
    }

    let mut heap: BinaryHeap<Entry> = BinaryHeap::new();

    for (id, point) in points {
        let dist = poincare_dist(query, point);
        if heap.len() < k {
            heap.push(Entry { id: *id, dist });
        } else if let Some(top) = heap.peek() {
            if dist < top.dist {
                heap.pop();
                heap.push(Entry { id: *id, dist });
            }
        }
    }

    let mut results: Vec<(usize, f64)> = heap.into_iter().map(|e| (e.id, e.dist)).collect();
    results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
    results
}

// ============================================================================
// Benchmark: Dimension Impact on Poincare Distance
// ============================================================================

fn bench_poincare_dimension_impact(c: &mut Criterion) {
    let max_norm = 0.99;

    let mut group = c.benchmark_group("poincare_dimension_impact");
    group.sample_size(500);

    for &dim in &[32, 64, 128, 256] {
        let mut rng = StdRng::seed_from_u64(42);
        let a = generate_poincare_point(dim, max_norm, &mut rng);
        let b = generate_poincare_point(dim, max_norm, &mut rng);

        group.bench_with_input(BenchmarkId::new("dim", dim), &dim, |bench, _| {
            bench.iter(|| black_box(poincare_dist(&a, &b)))
        });
    }

    group.finish();
}

// ============================================================================
// Criterion Groups
// ============================================================================

criterion_group!(
    benches,
    bench_distance_computation,
    bench_batch_distance,
    bench_knn_search_at_scale,
    bench_poincare_knn_index,
    bench_poincare_dimension_impact,
);

criterion_main!(benches);
