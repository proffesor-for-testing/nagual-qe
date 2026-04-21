//! Fine-tune command for LoRA adapter training.
//!
//! Trains lightweight Low-Rank Adaptation (LoRA) adapters using
//! contrastive learning on patterns from a specified domain.
//! The adapters improve retrieval accuracy for domain-specific queries.

use std::path::PathBuf;

use clap::Args;
use rand::Rng;

use crate::cli::common::init_storage;
use crate::error::Result;
use crate::ml::{to_array1, LoraAdapter, LoraConfig, LoraStorage, LoraTrainer, TrainingConfig};

/// Arguments for the finetune subcommand.
#[derive(Args, Debug)]
pub struct FinetuneArgs {
    /// Domain to fine-tune for.
    #[arg(long)]
    pub domain: String,

    /// Path to the database.
    #[arg(long, default_value = "./nagual.db")]
    pub db_path: PathBuf,

    /// LoRA rank (default: 4).
    #[arg(long, default_value = "4")]
    pub rank: usize,

    /// Maximum training epochs.
    #[arg(long, default_value = "50")]
    pub max_epochs: u32,

    /// Directory to save adapters.
    #[arg(long, default_value = "./models/lora")]
    pub adapter_dir: PathBuf,

    /// Use demo data for testing.
    #[arg(long)]
    pub demo: bool,
}

/// Run the finetune command: train a LoRA adapter for a domain.
pub async fn run(args: &FinetuneArgs) -> Result<()> {
    println!("\nNagual LoRA Fine-Tuning");
    println!("{:-<50}", "");
    println!("Domain: {}", args.domain);
    println!("Rank: {}", args.rank);
    println!("Max epochs: {}", args.max_epochs);
    println!("Adapter dir: {}", args.adapter_dir.display());
    println!();

    // Collect patterns with embeddings
    let pattern_data: Vec<(ndarray::Array1<f32>, String, f32)> = if args.demo {
        // Generate synthetic demo data for testing
        println!("Using demo data (synthetic embeddings)...");
        let mut rng = rand::thread_rng();
        let dim = 128;

        // Create a cluster center for the target domain
        let domain_center: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let domain_center = ndarray::Array1::from_vec(domain_center);
        let norm = domain_center.dot(&domain_center).sqrt();
        let domain_center = domain_center.mapv(|x| x / norm);

        // Create a different cluster center for other domains
        let other_center: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let other_center = ndarray::Array1::from_vec(other_center);
        let norm = other_center.dot(&other_center).sqrt();
        let other_center = other_center.mapv(|x| x / norm);

        let mut data = Vec::new();

        // 15 high-reward patterns in target domain
        for _ in 0..15 {
            let noise: ndarray::Array1<f32> =
                ndarray::Array1::from_vec((0..dim).map(|_| rng.gen_range(-0.1..0.1)).collect());
            let emb = &domain_center + &noise;
            let norm = emb.dot(&emb).sqrt();
            let emb = emb.mapv(|x| x / norm);
            let reward = rng.gen_range(0.7..1.0);
            data.push((emb, args.domain.clone(), reward));
        }

        // 10 low-reward patterns in other domains
        for i in 0..10 {
            let noise: ndarray::Array1<f32> =
                ndarray::Array1::from_vec((0..dim).map(|_| rng.gen_range(-0.1..0.1)).collect());
            let emb = &other_center + &noise;
            let norm = emb.dot(&emb).sqrt();
            let emb = emb.mapv(|x| x / norm);
            let reward = rng.gen_range(0.1..0.4);
            let domain = format!("other_{}", i % 3);
            data.push((emb, domain, reward));
        }

        data
    } else {
        // Load from database
        let storage = init_storage(&args.db_path, None).await?;
        let patterns = storage.get_recent(100_000).await?;

        let mut data = Vec::new();
        for pattern in &patterns {
            if let Some(emb) = pattern.embedding() {
                let emb_arr = to_array1(emb);
                let domain = pattern.category().to_string();
                let reward = pattern.reward();
                data.push((emb_arr, domain, reward));
            }
        }
        data
    };

    let total_patterns = pattern_data.len();
    let domain_count = pattern_data
        .iter()
        .filter(|(_, d, _)| *d == args.domain)
        .count();

    println!("Total patterns with embeddings: {}", total_patterns);
    println!(
        "Patterns in '{}' domain: {}",
        args.domain, domain_count
    );

    // Check minimum patterns threshold
    let training_config = TrainingConfig {
        max_epochs: args.max_epochs,
        patience: 5,
        margin: 0.5,
        min_patterns: 20,
        ..Default::default()
    };

    if total_patterns < training_config.min_patterns {
        println!(
            "\nInsufficient patterns: need at least {}, have {}.",
            training_config.min_patterns, total_patterns
        );
        println!("Run 'nagual learn embed' first to generate embeddings.");
        return Ok(());
    }

    // Generate training pairs
    println!("\nGenerating training pairs...");
    let pairs = LoraTrainer::generate_pairs(&pattern_data, 500);

    if pairs.is_empty() {
        println!("Could not generate training pairs.");
        println!("Need at least 2 high-reward patterns in the same domain and");
        println!("at least 1 low-reward or different-domain pattern.");
        return Ok(());
    }

    println!("Generated {} training pairs", pairs.len());

    // Create and train the adapter
    let lora_config = LoraConfig {
        base_dim: 128,
        rank: args.rank,
        learning_rate: 0.001,
        alpha: 1.0,
    };

    let mut adapter = LoraAdapter::new(&args.domain, lora_config);
    let trainer = LoraTrainer::new(training_config);

    println!("\nTraining LoRA adapter...");
    let start = std::time::Instant::now();
    let result = trainer.train(&mut adapter, &pairs).map_err(|e| {
        crate::error::NagualError::Internal { message: format!("Training failed: {}", e) }
    })?;
    let elapsed = start.elapsed();

    println!("\n=== Training Complete ===");
    println!("  Epochs: {}", result.epochs);
    println!("  Final loss: {:.4}", result.final_loss);
    println!("  Early stopped: {}", result.early_stopped);
    println!("  Training pairs: {}", result.num_pairs);
    println!("  Duration: {:.1}s", elapsed.as_secs_f64());
    println!(
        "  Adapter size: {} bytes ({:.1} KB)",
        adapter.size_bytes(),
        adapter.size_bytes() as f64 / 1024.0
    );

    // Save the adapter
    let lora_storage = LoraStorage::new(&args.adapter_dir);
    let saved_path = lora_storage.save(&adapter).map_err(|e| {
        crate::error::NagualError::Internal { message: format!("Failed to save adapter: {}", e) }
    })?;
    println!("\n  Saved to: {}", saved_path.display());

    // List existing adapters
    match lora_storage.list() {
        Ok(adapters) => {
            if adapters.len() > 1 {
                println!("\nAll stored adapters:");
                for a in &adapters {
                    println!(
                        "  - {} (loss: {:.4}, epochs: {}, {} bytes)",
                        a.domain, a.final_loss, a.iterations, a.size_bytes
                    );
                }
            }
        }
        Err(_) => {}
    }

    Ok(())
}
