//! Embedding search benchmarks for Nagual
//!
//! Benchmarks for vector similarity search operations including:
//! - Brute force search at various vector counts (1K, 10K, 100K)
//! - HNSW index building and querying
//! - Query throughput measurements
//!
//! Performance targets:
//! - 10K vectors: < 100ms for similarity search
//! - HNSW index building: O(n log n)
//! - Query throughput: > 100 queries/second for 10K vectors

use criterion::{
    black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput,
};
use rand::prelude::*;
use std::collections::BinaryHeap;
use std::cmp::Ordering;

// ============================================================================
// Test Data Generation
// ============================================================================

/// Generate a random normalized embedding vector
fn generate_random_embedding(dim: usize, rng: &mut StdRng) -> Vec<f32> {
    let mut embedding: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect();
    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > f32::EPSILON {
        embedding.iter_mut().for_each(|x| *x /= norm);
    }
    embedding
}

/// Generate a set of random embeddings
fn generate_embedding_set(count: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..count)
        .map(|_| generate_random_embedding(dim, &mut rng))
        .collect()
}

// ============================================================================
// Vector Similarity Operations
// ============================================================================

/// Compute dot product (cosine similarity for normalized vectors)
#[inline]
fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Compute L2 squared distance
#[inline]
fn l2_squared(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum()
}

/// SIMD-friendly dot product using chunks
#[inline]
fn dot_product_chunked(a: &[f32], b: &[f32]) -> f32 {
    const CHUNK_SIZE: usize = 8;
    let mut sum = 0.0f32;

    let chunks_a = a.chunks_exact(CHUNK_SIZE);
    let chunks_b = b.chunks_exact(CHUNK_SIZE);
    let remainder_a = chunks_a.remainder();
    let remainder_b = chunks_b.remainder();

    for (chunk_a, chunk_b) in chunks_a.zip(chunks_b) {
        let mut chunk_sum = 0.0f32;
        for i in 0..CHUNK_SIZE {
            chunk_sum += chunk_a[i] * chunk_b[i];
        }
        sum += chunk_sum;
    }

    // Handle remainder
    for (a_val, b_val) in remainder_a.iter().zip(remainder_b.iter()) {
        sum += a_val * b_val;
    }

    sum
}

// ============================================================================
// Brute Force Search
// ============================================================================

/// Result entry for k-NN search
#[derive(Clone, Copy)]
struct SearchEntry {
    index: usize,
    score: f32,
}

impl PartialEq for SearchEntry {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score
    }
}

impl Eq for SearchEntry {}

impl PartialOrd for SearchEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SearchEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Min-heap: lower scores have higher priority
        other.score.partial_cmp(&self.score).unwrap_or(Ordering::Equal)
    }
}

/// Brute force k-NN search using a min-heap
fn brute_force_knn(
    vectors: &[Vec<f32>],
    query: &[f32],
    k: usize,
) -> Vec<(usize, f32)> {
    let mut heap: BinaryHeap<SearchEntry> = BinaryHeap::with_capacity(k + 1);

    for (idx, vec) in vectors.iter().enumerate() {
        let score = dot_product(query, vec);

        if heap.len() < k {
            heap.push(SearchEntry { index: idx, score });
        } else if let Some(min) = heap.peek() {
            if score > min.score {
                heap.pop();
                heap.push(SearchEntry { index: idx, score });
            }
        }
    }

    let mut results: Vec<(usize, f32)> = heap
        .into_iter()
        .map(|e| (e.index, e.score))
        .collect();
    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
    results
}

/// Brute force search returning all scores (for comparison)
fn brute_force_all(vectors: &[Vec<f32>], query: &[f32]) -> Vec<f32> {
    vectors.iter().map(|v| dot_product(query, v)).collect()
}

// ============================================================================
// Simple HNSW Implementation (for benchmarking)
// ============================================================================

/// Simple HNSW-like index for benchmarking
/// This is a simplified implementation for performance testing
struct SimpleHnswIndex {
    vectors: Vec<Vec<f32>>,
    // Graph layers: layer -> node -> neighbors
    layers: Vec<Vec<Vec<usize>>>,
    entry_point: usize,
    m: usize,        // Max connections per node
    ef_construction: usize,
}

impl SimpleHnswIndex {
    fn new(m: usize, ef_construction: usize) -> Self {
        Self {
            vectors: Vec::new(),
            layers: Vec::new(),
            entry_point: 0,
            m,
            ef_construction,
        }
    }

    /// Build index from vectors
    fn build(vectors: &[Vec<f32>], m: usize, ef_construction: usize, seed: u64) -> Self {
        let mut index = Self::new(m, ef_construction);
        let mut rng = StdRng::seed_from_u64(seed);

        for (i, vec) in vectors.iter().enumerate() {
            index.insert(vec.clone(), &mut rng);
            if i == 0 {
                index.entry_point = 0;
            }
        }

        index
    }

    /// Insert a vector into the index
    fn insert(&mut self, vector: Vec<f32>, rng: &mut StdRng) {
        let idx = self.vectors.len();
        self.vectors.push(vector);

        // Determine the layer for this node (geometric distribution)
        let level = (-rng.gen::<f64>().ln() * (1.0 / (self.m as f64).ln())).floor() as usize;

        // Ensure we have enough layers
        while self.layers.len() <= level {
            self.layers.push(Vec::new());
        }

        // Initialize neighbors for all layers up to level
        for layer in self.layers.iter_mut().take(level + 1) {
            while layer.len() <= idx {
                layer.push(Vec::new());
            }
        }

        // If this is the first node, we're done
        if idx == 0 {
            return;
        }

        // Connect to nearest neighbors at each layer
        let mut ep = self.entry_point;

        for l in (0..=level.min(self.layers.len() - 1)).rev() {
            // Search for nearest neighbors at this layer
            let neighbors = self.search_layer(&self.vectors[idx], ep, self.ef_construction, l);

            // Select m best neighbors
            let selected: Vec<usize> = neighbors
                .into_iter()
                .take(self.m)
                .map(|(i, _)| i)
                .collect();

            // Add bidirectional connections
            for &neighbor in &selected {
                if self.layers[l].len() > idx {
                    self.layers[l][idx].push(neighbor);
                }
                if self.layers[l].len() > neighbor {
                    self.layers[l][neighbor].push(idx);
                    // Prune if too many connections
                    if self.layers[l][neighbor].len() > self.m * 2 {
                        self.layers[l][neighbor].truncate(self.m * 2);
                    }
                }
            }

            if !selected.is_empty() {
                ep = selected[0];
            }
        }

        // Update entry point if this node is at a higher level
        if level >= self.layers.len() - 1 {
            self.entry_point = idx;
        }
    }

    /// Search within a single layer
    fn search_layer(
        &self,
        query: &[f32],
        entry_point: usize,
        ef: usize,
        layer: usize,
    ) -> Vec<(usize, f32)> {
        if layer >= self.layers.len() || self.vectors.is_empty() {
            return Vec::new();
        }

        let mut visited = vec![false; self.vectors.len()];
        let mut candidates: BinaryHeap<SearchEntry> = BinaryHeap::new();
        let mut results: Vec<(usize, f32)> = Vec::new();

        let initial_score = dot_product(query, &self.vectors[entry_point]);
        candidates.push(SearchEntry {
            index: entry_point,
            score: -initial_score, // Negate for max-heap behavior
        });
        visited[entry_point] = true;
        results.push((entry_point, initial_score));

        while let Some(SearchEntry { index: current, .. }) = candidates.pop() {
            if layer < self.layers.len() && current < self.layers[layer].len() {
                for &neighbor in &self.layers[layer][current] {
                    if neighbor < self.vectors.len() && !visited[neighbor] {
                        visited[neighbor] = true;
                        let score = dot_product(query, &self.vectors[neighbor]);
                        candidates.push(SearchEntry {
                            index: neighbor,
                            score: -score,
                        });
                        results.push((neighbor, score));

                        if results.len() >= ef {
                            break;
                        }
                    }
                }
            }

            if results.len() >= ef {
                break;
            }
        }

        // Sort by score (descending)
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        results.truncate(ef);
        results
    }

    /// Search the index for k nearest neighbors
    fn search(&self, query: &[f32], k: usize, ef_search: usize) -> Vec<(usize, f32)> {
        if self.vectors.is_empty() {
            return Vec::new();
        }

        let mut ep = self.entry_point;
        let num_layers = self.layers.len();

        // Traverse from top layer to layer 1
        for l in (1..num_layers).rev() {
            let neighbors = self.search_layer(query, ep, 1, l);
            if let Some((best, _)) = neighbors.first() {
                ep = *best;
            }
        }

        // Search at layer 0 with full ef
        let mut results = self.search_layer(query, ep, ef_search.max(k), 0);
        results.truncate(k);
        results
    }
}

// ============================================================================
// Benchmark Functions
// ============================================================================

fn bench_dot_product(c: &mut Criterion) {
    let mut group = c.benchmark_group("dot_product");
    group.sample_size(100);

    let mut rng = StdRng::seed_from_u64(42);

    for dim in [128, 256, 384, 512, 768].iter() {
        let a = generate_random_embedding(*dim, &mut rng);
        let b = generate_random_embedding(*dim, &mut rng);

        group.throughput(Throughput::Elements(*dim as u64));

        group.bench_with_input(
            BenchmarkId::new("standard", dim),
            dim,
            |bench, _| {
                bench.iter(|| black_box(dot_product(&a, &b)))
            },
        );

        group.bench_with_input(
            BenchmarkId::new("chunked", dim),
            dim,
            |bench, _| {
                bench.iter(|| black_box(dot_product_chunked(&a, &b)))
            },
        );
    }

    group.finish();
}

fn bench_brute_force_search(c: &mut Criterion) {
    let dim = 128;
    let k = 10;

    let mut group = c.benchmark_group("brute_force_search");
    group.sample_size(30);

    // Performance target: 10K vectors < 100ms
    for &vector_count in &[1_000, 10_000, 100_000] {
        let vectors = generate_embedding_set(vector_count, dim, 42);
        let mut rng = StdRng::seed_from_u64(123);
        let query = generate_random_embedding(dim, &mut rng);

        group.throughput(Throughput::Elements(vector_count as u64));

        group.bench_with_input(
            BenchmarkId::new("knn", vector_count),
            &vector_count,
            |b, _| {
                b.iter(|| black_box(brute_force_knn(&vectors, &query, k)))
            },
        );
    }

    group.finish();
}

fn bench_hnsw_build(c: &mut Criterion) {
    let dim = 128;
    let m = 16;
    let ef_construction = 100;

    let mut group = c.benchmark_group("hnsw_index_build");
    group.sample_size(10); // Building is expensive

    for &vector_count in &[1_000, 5_000, 10_000] {
        let vectors = generate_embedding_set(vector_count, dim, 42);

        group.throughput(Throughput::Elements(vector_count as u64));

        group.bench_with_input(
            BenchmarkId::new("build", vector_count),
            &vector_count,
            |b, _| {
                b.iter(|| black_box(SimpleHnswIndex::build(&vectors, m, ef_construction, 42)))
            },
        );
    }

    group.finish();
}

fn bench_hnsw_query(c: &mut Criterion) {
    let dim = 128;
    let k = 10;
    let m = 16;
    let ef_construction = 100;
    let ef_search = 50;

    let mut group = c.benchmark_group("hnsw_query");
    group.sample_size(50);

    for &vector_count in &[1_000, 10_000] {
        let vectors = generate_embedding_set(vector_count, dim, 42);
        let index = SimpleHnswIndex::build(&vectors, m, ef_construction, 42);

        let mut rng = StdRng::seed_from_u64(123);
        let query = generate_random_embedding(dim, &mut rng);

        group.throughput(Throughput::Elements(1));

        group.bench_with_input(
            BenchmarkId::new("search", vector_count),
            &vector_count,
            |b, _| {
                b.iter(|| black_box(index.search(&query, k, ef_search)))
            },
        );
    }

    group.finish();
}

fn bench_query_throughput(c: &mut Criterion) {
    let dim = 128;
    let k = 10;
    let vector_count = 10_000;

    let vectors = generate_embedding_set(vector_count, dim, 42);

    // Generate multiple queries
    let queries: Vec<Vec<f32>> = generate_embedding_set(100, dim, 999);

    let mut group = c.benchmark_group("query_throughput");
    group.sample_size(20);
    group.throughput(Throughput::Elements(100)); // 100 queries per iteration

    group.bench_function("brute_force_100_queries", |b| {
        b.iter(|| {
            for query in &queries {
                black_box(brute_force_knn(&vectors, query, k));
            }
        })
    });

    // Build HNSW index for comparison
    let index = SimpleHnswIndex::build(&vectors, 16, 100, 42);

    group.bench_function("hnsw_100_queries", |b| {
        b.iter(|| {
            for query in &queries {
                black_box(index.search(query, k, 50));
            }
        })
    });

    group.finish();
}

fn bench_scaling(c: &mut Criterion) {
    let dim = 128;
    let k = 10;

    let mut group = c.benchmark_group("embedding_scaling");
    group.sample_size(10);

    // Test scaling characteristics
    for &vector_count in &[1_000, 2_500, 5_000, 10_000, 25_000, 50_000] {
        let vectors = generate_embedding_set(vector_count, dim, 42);
        let mut rng = StdRng::seed_from_u64(123);
        let query = generate_random_embedding(dim, &mut rng);

        group.throughput(Throughput::Elements(vector_count as u64));

        group.bench_with_input(
            BenchmarkId::new("brute_force", vector_count),
            &vector_count,
            |b, _| {
                b.iter(|| black_box(brute_force_knn(&vectors, &query, k)))
            },
        );
    }

    group.finish();
}

fn bench_dimension_impact(c: &mut Criterion) {
    let vector_count = 10_000;
    let k = 10;

    let mut group = c.benchmark_group("dimension_impact");
    group.sample_size(20);

    for &dim in &[64, 128, 256, 384, 512] {
        let vectors = generate_embedding_set(vector_count, dim, 42);
        let mut rng = StdRng::seed_from_u64(123);
        let query = generate_random_embedding(dim, &mut rng);

        group.throughput(Throughput::Elements(dim as u64 * vector_count as u64));

        group.bench_with_input(
            BenchmarkId::new("dim", dim),
            &dim,
            |b, _| {
                b.iter(|| black_box(brute_force_knn(&vectors, &query, k)))
            },
        );
    }

    group.finish();
}

fn bench_batch_search(c: &mut Criterion) {
    let dim = 128;
    let k = 10;
    let vector_count = 10_000;

    let vectors = generate_embedding_set(vector_count, dim, 42);

    let mut group = c.benchmark_group("batch_search");
    group.sample_size(20);

    // Test batch sizes
    for &batch_size in &[1, 10, 50, 100] {
        let queries = generate_embedding_set(batch_size, dim, 999);

        group.throughput(Throughput::Elements(batch_size as u64));

        group.bench_with_input(
            BenchmarkId::new("batch", batch_size),
            &batch_size,
            |b, _| {
                b.iter(|| {
                    let results: Vec<_> = queries
                        .iter()
                        .map(|q| brute_force_knn(&vectors, q, k))
                        .collect();
                    black_box(results)
                })
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_dot_product,
    bench_brute_force_search,
    bench_hnsw_build,
    bench_hnsw_query,
    bench_query_throughput,
    bench_scaling,
    bench_dimension_impact,
    bench_batch_search,
);

criterion_main!(benches);
