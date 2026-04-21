//! Quick HNSW recall test - runs in seconds, not minutes.
//!
//! Tests recall at different scales with tuned parameters.
//! Run with: cargo run --example quick_recall_test --release

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
struct NodePoint { id: usize, embedding: Vec<f32> }

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
    let q = NodePoint { id: usize::MAX, embedding: query.to_vec() };
    let mut search = Search::default();
    index.search(&q, &mut search).take(k).map(|n| *n.value).collect()
}

fn test_recall(node_count: usize, m: usize, ef_construction: usize, ef_search: usize, num_queries: usize) -> f32 {
    let dim = 128;
    let k = 10;

    println!("Building index: {} nodes, m={}, ef_construction={}, ef_search={}", node_count, m, ef_construction, ef_search);
    let start = Instant::now();

    let embeddings = generate_embedding_set(node_count, dim, 42);
    let queries = generate_related_queries(&embeddings, num_queries, 0.1, 123);

    let points: Vec<NodePoint> = embeddings.iter().enumerate()
        .map(|(id, e)| NodePoint { id, embedding: e.clone() }).collect();
    let values: Vec<usize> = (0..embeddings.len()).collect();
    // Critical: set ef_search at build time (instant-distance API requirement)
    let index = HnswBuilder::default()
        .ef_construction(ef_construction)
        .ef_search(ef_search)
        .build(points, values);

    println!("  Build time: {:?}", start.elapsed());

    let mut total_recall = 0.0;
    for query in &queries {
        let truth: HashSet<usize> = brute_force_knn(&embeddings, query, k).into_iter().collect();
        let approx: HashSet<usize> = hnsw_search(&index, query, k).into_iter().collect();
        total_recall += truth.intersection(&approx).count() as f32 / k as f32;
    }

    total_recall / num_queries as f32
}

fn main() {
    println!("======================================================================");
    println!("          Quick HNSW Recall Test (Tuned Parameters)                   ");
    println!("======================================================================\n");

    // Test configurations: (name, m, ef_construction, ef_search)
    // ef_search is the key parameter for recall at query time
    let configs = [
        ("Original (ef_s=100)", 16, 128, 100),
        ("Tuned (ef_s=200)", 24, 200, 200),
        ("High (ef_s=300)", 32, 300, 300),
    ];

    let scales = [10_000, 50_000, 100_000];
    let num_queries = 50;  // Fewer queries for speed

    println!("+-------------------------+------------+------------+------------+");
    println!("| Configuration           |    10K     |    50K     |   100K     |");
    println!("+-------------------------+------------+------------+------------+");

    for (name, m, ef_c, ef_s) in configs {
        let mut recalls = Vec::new();
        for &scale in &scales {
            let recall = test_recall(scale, m, ef_c, ef_s, num_queries);
            recalls.push(recall);
        }

        let status = |r: f32| if r >= 0.95 { "PASS" } else { "FAIL" };
        println!("| {:23} | {:.2} {:4} | {:.2} {:4} | {:.2} {:4} |",
            name,
            recalls[0], status(recalls[0]),
            recalls[1], status(recalls[1]),
            recalls[2], status(recalls[2]),
        );
    }

    println!("+-------------------------+------------+------------+------------+");
    println!("\nTarget: Recall > 0.95 (95%)");
}
