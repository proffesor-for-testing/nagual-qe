//! Comprehensive ProfDAG Benchmarks (PROFDAG-011)
//!
//! End-to-end performance benchmarks covering the full ProfDAG pipeline:
//!
//! - **profdag_end_to_end**: Full cycle - store, search, traverse, record
//! - **profdag_search_variants**: Compare ef_search values on profiler-backed search
//! - **profdag_traversal**: Graph traversal with and without wormhole shortcuts
//! - **profdag_routing**: Vendor routing decision latency
//! - **profdag_injection**: E_nagual computation + formatting
//! - **profdag_light_cone**: Light cone building at different depths
//!
//! Run with:
//! ```sh
//! cargo bench --bench comprehensive_benchmark
//! ```

use criterion::{
    black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput,
};
use instant_distance::{Builder as HnswBuilder, HnswMap, Search};
use rand::prelude::*;
use std::collections::HashMap;

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

/// Generate query embeddings related to the base set.
fn generate_related_queries(
    base_embeddings: &[Vec<f32>],
    num_queries: usize,
    noise_level: f32,
    seed: u64,
) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..num_queries)
        .map(|_| {
            let base_idx = rng.gen_range(0..base_embeddings.len());
            let base = &base_embeddings[base_idx];
            let mut query: Vec<f32> = base
                .iter()
                .map(|&x| x + rng.gen_range(-noise_level..noise_level))
                .collect();
            let norm: f32 = query.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > f32::EPSILON {
                query.iter_mut().for_each(|x| *x /= norm);
            }
            query
        })
        .collect()
}

// ============================================================================
// HNSW Helpers
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
        1.0 - self
            .embedding
            .iter()
            .zip(other.embedding.iter())
            .map(|(a, b)| a * b)
            .sum::<f32>()
    }
}

/// Build an HNSW index.
fn build_hnsw_index(
    embeddings: &[Vec<f32>],
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
        .ef_search(ef_search)
        .build(points, values)
}

/// Search the HNSW index.
fn hnsw_search(index: &HnswMap<NodePoint, usize>, query: &[f32], k: usize) -> Vec<(usize, f32)> {
    let query_point = NodePoint {
        id: usize::MAX,
        embedding: query.to_vec(),
    };
    let mut search = Search::default();
    index
        .search(&query_point, &mut search)
        .take(k)
        .map(|n| (*n.value, 1.0 - n.distance))
        .collect()
}

// ============================================================================
// Simulated ProfDAG Operations
// ============================================================================

/// Simulate a graph traversal by following edges from a starting node.
/// Returns the number of edges traversed and whether a wormhole was used.
fn simulate_traversal(
    adjacency: &[Vec<usize>],
    start: usize,
    max_depth: usize,
    wormholes: &HashMap<usize, usize>,
) -> (usize, bool) {
    let mut visited = 0;
    let mut current = start;
    let mut used_wormhole = false;

    for _depth in 0..max_depth {
        // Check wormhole shortcut
        if let Some(&target) = wormholes.get(&current) {
            current = target;
            used_wormhole = true;
            visited += 1;
            continue;
        }

        if let Some(neighbors) = adjacency.get(current) {
            if neighbors.is_empty() {
                break;
            }
            // Follow first neighbor (deterministic traversal for benchmarking)
            current = neighbors[0];
            visited += 1;
        } else {
            break;
        }
    }

    (visited, used_wormhole)
}

/// Build a random adjacency list for graph traversal benchmarks.
fn build_random_graph(node_count: usize, avg_degree: usize, seed: u64) -> Vec<Vec<usize>> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut adjacency: Vec<Vec<usize>> = vec![Vec::new(); node_count];

    for i in 0..node_count {
        let degree = rng.gen_range(1..=avg_degree * 2);
        for _ in 0..degree {
            let target = rng.gen_range(0..node_count);
            if target != i {
                adjacency[i].push(target);
            }
        }
    }

    adjacency
}

/// Build random wormhole shortcuts.
fn build_wormholes(node_count: usize, count: usize, seed: u64) -> HashMap<usize, usize> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut wormholes = HashMap::new();
    for _ in 0..count {
        let source = rng.gen_range(0..node_count);
        let target = rng.gen_range(0..node_count);
        if source != target {
            wormholes.insert(source, target);
        }
    }
    wormholes
}

/// Simulate vendor routing: compute a complexity score and select a vendor.
fn simulate_routing(features: &[f32]) -> &'static str {
    let complexity: f32 = features.iter().map(|f| f.abs()).sum::<f32>() / features.len() as f32;
    if complexity > 0.7 {
        "anthropic"
    } else if complexity > 0.4 {
        "openai"
    } else {
        "local"
    }
}

/// Simulate E_nagual injection: format patterns into a prompt block.
fn simulate_injection(patterns: &[Vec<f32>], _token_budget: usize) -> String {
    let mut output = String::with_capacity(patterns.len() * 256);
    output.push_str("<e_nagual confidence=\"0.85\">\n");

    for (i, pattern) in patterns.iter().enumerate() {
        output.push_str(&format!("  <pattern id=\"{}\" dim=\"{}\">\n", i, pattern.len()));
        // Simulate formatting a subset of the embedding
        let preview: Vec<String> = pattern.iter().take(8).map(|v| format!("{:.4}", v)).collect();
        output.push_str(&format!("    [{}]\n", preview.join(", ")));
        output.push_str("  </pattern>\n");
    }

    output.push_str("</e_nagual>");
    output
}

/// Simulate light cone building: BFS from a center node to a given depth.
fn simulate_light_cone(adjacency: &[Vec<usize>], center: usize, depth: usize) -> usize {
    let mut visited = vec![false; adjacency.len()];
    let mut frontier = vec![center];
    visited[center] = true;
    let mut total_visited = 1;

    for _d in 0..depth {
        let mut next_frontier = Vec::new();
        for &node in &frontier {
            if let Some(neighbors) = adjacency.get(node) {
                for &n in neighbors {
                    if !visited[n] {
                        visited[n] = true;
                        next_frontier.push(n);
                        total_visited += 1;
                    }
                }
            }
        }
        frontier = next_frontier;
        if frontier.is_empty() {
            break;
        }
    }

    total_visited
}

// ============================================================================
// Benchmark Groups
// ============================================================================

fn bench_profdag_end_to_end(c: &mut Criterion) {
    let dim = 128;
    let k = 10;
    let node_count = 10_000;
    let ef_construction = 200;
    let ef_search = 200;

    let embeddings = generate_embedding_set(node_count, dim, 42);
    let queries = generate_related_queries(&embeddings, 50, 0.1, 123);
    let index = build_hnsw_index(&embeddings, ef_construction, ef_search);
    let adjacency = build_random_graph(node_count, 4, 55);
    let wormholes = build_wormholes(node_count, 200, 66);

    let mut group = c.benchmark_group("profdag_end_to_end");
    group.sample_size(50);
    group.throughput(Throughput::Elements(1));

    group.bench_function("full_cycle", |b| {
        let mut qi = 0;
        b.iter(|| {
            let query = &queries[qi % queries.len()];
            qi += 1;

            // 1. Search
            let results = hnsw_search(&index, query, k);
            black_box(&results);

            // 2. Traverse from top result
            if let Some((top_id, _)) = results.first() {
                let (edges, _wh) = simulate_traversal(&adjacency, *top_id, 5, &wormholes);
                black_box(edges);
            }

            // 3. Route
            let vendor = simulate_routing(query);
            black_box(vendor);

            // 4. Inject
            let patterns: Vec<Vec<f32>> = results
                .iter()
                .take(3)
                .map(|(id, _)| embeddings[*id].clone())
                .collect();
            let block = simulate_injection(&patterns, 2048);
            black_box(&block);
        })
    });

    group.finish();
}

fn bench_profdag_search_variants(c: &mut Criterion) {
    let dim = 128;
    let k = 10;
    let node_count = 50_000;
    let ef_construction = 200;

    let embeddings = generate_embedding_set(node_count, dim, 42);
    let queries = generate_related_queries(&embeddings, 20, 0.1, 123);

    let mut group = c.benchmark_group("profdag_search_variants");
    group.sample_size(30);

    for &ef_search in &[50, 100, 150, 200, 300] {
        let index = build_hnsw_index(&embeddings, ef_construction, ef_search);

        group.throughput(Throughput::Elements(1));
        group.bench_with_input(
            BenchmarkId::new("ef_search", ef_search),
            &ef_search,
            |b, _| {
                let mut qi = 0;
                b.iter(|| {
                    let query = &queries[qi % queries.len()];
                    qi += 1;
                    black_box(hnsw_search(&index, query, k))
                })
            },
        );
    }

    group.finish();
}

fn bench_profdag_traversal(c: &mut Criterion) {
    let node_count = 50_000;
    let avg_degree = 5;

    let adjacency = build_random_graph(node_count, avg_degree, 42);
    let wormholes = build_wormholes(node_count, 500, 77);
    let empty_wormholes: HashMap<usize, usize> = HashMap::new();

    let mut rng = StdRng::seed_from_u64(99);
    let start_nodes: Vec<usize> = (0..100).map(|_| rng.gen_range(0..node_count)).collect();

    let mut group = c.benchmark_group("profdag_traversal");
    group.sample_size(100);

    for &depth in &[3, 5, 10, 20] {
        group.throughput(Throughput::Elements(1));

        group.bench_with_input(
            BenchmarkId::new("without_wormholes", depth),
            &depth,
            |b, &depth| {
                let mut si = 0;
                b.iter(|| {
                    let start = start_nodes[si % start_nodes.len()];
                    si += 1;
                    black_box(simulate_traversal(
                        &adjacency,
                        start,
                        depth,
                        &empty_wormholes,
                    ))
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("with_wormholes", depth),
            &depth,
            |b, &depth| {
                let mut si = 0;
                b.iter(|| {
                    let start = start_nodes[si % start_nodes.len()];
                    si += 1;
                    black_box(simulate_traversal(&adjacency, start, depth, &wormholes))
                })
            },
        );
    }

    group.finish();
}

fn bench_profdag_routing(c: &mut Criterion) {
    let dim = 128;
    let mut rng = StdRng::seed_from_u64(42);
    let feature_sets: Vec<Vec<f32>> = (0..1000)
        .map(|_| generate_random_embedding(dim, &mut rng))
        .collect();

    let mut group = c.benchmark_group("profdag_routing");
    group.sample_size(200);
    group.throughput(Throughput::Elements(1));

    group.bench_function("vendor_decision", |b| {
        let mut fi = 0;
        b.iter(|| {
            let features = &feature_sets[fi % feature_sets.len()];
            fi += 1;
            black_box(simulate_routing(features))
        })
    });

    // Batch routing
    for &batch_size in &[10, 50, 100] {
        group.throughput(Throughput::Elements(batch_size as u64));
        group.bench_with_input(
            BenchmarkId::new("batch", batch_size),
            &batch_size,
            |b, &bs| {
                b.iter(|| {
                    let results: Vec<&str> = feature_sets
                        .iter()
                        .take(bs)
                        .map(|f| simulate_routing(f))
                        .collect();
                    black_box(results)
                })
            },
        );
    }

    group.finish();
}

fn bench_profdag_injection(c: &mut Criterion) {
    let dim = 128;
    let mut rng = StdRng::seed_from_u64(42);
    let patterns: Vec<Vec<f32>> = (0..50)
        .map(|_| generate_random_embedding(dim, &mut rng))
        .collect();

    let mut group = c.benchmark_group("profdag_injection");
    group.sample_size(100);

    for &pattern_count in &[1, 3, 5, 10, 20] {
        let subset: Vec<Vec<f32>> = patterns.iter().take(pattern_count).cloned().collect();

        group.throughput(Throughput::Elements(1));
        group.bench_with_input(
            BenchmarkId::new("patterns", pattern_count),
            &pattern_count,
            |b, _| b.iter(|| black_box(simulate_injection(&subset, 4096))),
        );
    }

    group.finish();
}

fn bench_profdag_light_cone(c: &mut Criterion) {
    let node_count = 50_000;
    let avg_degree = 5;

    let adjacency = build_random_graph(node_count, avg_degree, 42);

    let mut rng = StdRng::seed_from_u64(99);
    let center_nodes: Vec<usize> = (0..50).map(|_| rng.gen_range(0..node_count)).collect();

    let mut group = c.benchmark_group("profdag_light_cone");
    group.sample_size(30);

    for &depth in &[1, 2, 3, 5, 8] {
        group.throughput(Throughput::Elements(1));
        group.bench_with_input(
            BenchmarkId::new("depth", depth),
            &depth,
            |b, &depth| {
                let mut ci = 0;
                b.iter(|| {
                    let center = center_nodes[ci % center_nodes.len()];
                    ci += 1;
                    black_box(simulate_light_cone(&adjacency, center, depth))
                })
            },
        );
    }

    group.finish();
}

// ============================================================================
// Criterion Groups
// ============================================================================

criterion_group!(
    benches,
    bench_profdag_end_to_end,
    bench_profdag_search_variants,
    bench_profdag_traversal,
    bench_profdag_routing,
    bench_profdag_injection,
    bench_profdag_light_cone,
);

criterion_main!(benches);
