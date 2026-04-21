//! Minimal test to verify ef_search impact on recall.
//!
//! This demonstrates the critical importance of setting ef_search at build time.
//! Run with: cargo run --example ef_search_test --release

use instant_distance::{Builder as HnswBuilder, HnswMap, Search};
use rand::prelude::*;
use std::collections::HashSet;
use std::time::Instant;

fn generate_random_embedding(dim: usize, rng: &mut StdRng) -> Vec<f32> {
    let mut embedding: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect();
    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > f32::EPSILON {
        embedding.iter_mut().for_each(|x| *x /= norm);
    }
    embedding
}

fn generate_embedding_set(count: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..count).map(|_| generate_random_embedding(dim, &mut rng)).collect()
}

fn generate_related_queries(base: &[Vec<f32>], num: usize, noise: f32, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..num).map(|_| {
        let base_idx = rng.gen_range(0..base.len());
        let mut query: Vec<f32> = base[base_idx].iter()
            .map(|&x| x + rng.gen_range(-noise..noise))
            .collect();
        let norm: f32 = query.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > f32::EPSILON { query.iter_mut().for_each(|x| *x /= norm); }
        query
    }).collect()
}

#[derive(Clone)]
struct NodePoint { embedding: Vec<f32> }

impl instant_distance::Point for NodePoint {
    fn distance(&self, other: &Self) -> f32 {
        1.0 - self.embedding.iter().zip(&other.embedding).map(|(a,b)| a*b).sum::<f32>()
    }
}

fn brute_force_knn(embeddings: &[Vec<f32>], query: &[f32], k: usize) -> Vec<usize> {
    let mut scores: Vec<(usize, f32)> = embeddings.iter().enumerate()
        .map(|(i, e)| (i, e.iter().zip(query).map(|(a,b)| a*b).sum::<f32>()))
        .collect();
    scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    scores.into_iter().take(k).map(|(i, _)| i).collect()
}

fn hnsw_search(index: &HnswMap<NodePoint, usize>, query: &[f32], k: usize) -> Vec<usize> {
    let q = NodePoint { embedding: query.to_vec() };
    let mut search = Search::default();
    index.search(&q, &mut search).take(k).map(|n| *n.value).collect()
}

fn test_recall(embeddings: &[Vec<f32>], queries: &[Vec<f32>], ef_construction: usize, ef_search: usize, k: usize) -> (f32, f64) {
    let start = Instant::now();

    let points: Vec<NodePoint> = embeddings.iter()
        .map(|e| NodePoint { embedding: e.clone() }).collect();
    let values: Vec<usize> = (0..embeddings.len()).collect();

    let index = HnswBuilder::default()
        .ef_construction(ef_construction)
        .ef_search(ef_search)  // THE KEY PARAMETER
        .build(points, values);

    let build_time = start.elapsed().as_secs_f64();

    let mut total_recall = 0.0;
    for query in queries {
        let truth: HashSet<usize> = brute_force_knn(embeddings, query, k).into_iter().collect();
        let approx: HashSet<usize> = hnsw_search(&index, query, k).into_iter().collect();
        total_recall += truth.intersection(&approx).count() as f32 / k as f32;
    }

    (total_recall / queries.len() as f32, build_time)
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║     ef_search Impact Test (instant-distance API verification)    ║");
    println!("╠══════════════════════════════════════════════════════════════════╣");
    println!("║ ef_search is set at BUILD TIME in instant-distance (not query)   ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    // Use smaller scale for faster testing
    let node_count = 5_000;
    let dim = 128;
    let k = 10;
    let num_queries = 50;
    let ef_construction = 200;

    println!("Test setup: {} nodes, dim={}, k={}, queries={}", node_count, dim, k, num_queries);
    println!("ef_construction={} (fixed)\n", ef_construction);

    let embeddings = generate_embedding_set(node_count, dim, 42);
    let queries = generate_related_queries(&embeddings, num_queries, 0.1, 123);

    println!("+-------------+------------+---------------+--------+");
    println!("| ef_search   | Build Time |    Recall     | Status |");
    println!("+-------------+------------+---------------+--------+");

    // Test different ef_search values
    for ef_search in [20, 50, 100, 150, 200, 300] {
        let (recall, build_time) = test_recall(&embeddings, &queries, ef_construction, ef_search, k);
        let status = if recall >= 0.95 { "✓ PASS" } else { "✗ FAIL" };
        println!("| ef_s={:4}   |   {:5.2}s   |    {:.4}     | {} |",
            ef_search, build_time, recall, status);
    }

    println!("+-------------+------------+---------------+--------+");
    println!("\nTarget: Recall > 0.95 (95%)");
    println!("\nConclusion: ef_search directly controls recall quality.");
    println!("Higher ef_search = higher recall = slightly slower search.");
}
