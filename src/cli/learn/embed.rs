//! Embed command for generating pattern embeddings.
//!
//! Uses ONNX MiniLM model to generate 128-dimensional embeddings
//! for patterns. Enables similarity search, consolidation, and
//! the full learning loop.

use std::path::PathBuf;

use clap::Args;

use crate::cli::common::{init_storage, resolve_postgres_url};
use crate::error::Result;
use crate::ml::{cosine_similarity, to_array1, HashEmbedder};
#[cfg(feature = "onnx-embed")]
use crate::ml::{Embedder, EmbedderConfig};
use crate::reasoning_bank::pattern::Pattern;

/// Arguments for the embed subcommand.
#[derive(Args, Debug)]
pub struct EmbedArgs {
    /// Path to ONNX model file.
    #[arg(long, default_value = "models/all-MiniLM-L6-v2.onnx")]
    pub model_path: String,

    /// Path to tokenizer JSON file.
    #[arg(long, default_value = "models/tokenizer.json")]
    pub tokenizer_path: String,

    /// Batch size for embedding generation.
    #[arg(long, default_value = "32")]
    pub batch_size: usize,

    /// Re-generate embeddings even for patterns that already have them.
    #[arg(long)]
    pub force: bool,

    /// Path to SQLite database.
    #[arg(long, default_value = "./nagual.db")]
    pub db_path: PathBuf,

    /// PostgreSQL connection URL (overrides DATABASE_URL env and config.toml).
    #[arg(long, env = "DATABASE_URL")]
    pub postgres_url: Option<String>,

    /// Output results as JSON.
    #[arg(long)]
    pub json: bool,

    /// Also generate hyperbolic (Poincare ball) embeddings.
    ///
    /// Converts Euclidean embeddings into hyperbolic space using domain
    /// hierarchy depth to position points. Root domains are placed near
    /// the origin, while deeply nested domains are placed near the boundary.
    #[arg(long)]
    pub hyperbolic: bool,

    /// Show verbose output.
    #[arg(short, long)]
    pub verbose: bool,
}

/// Run the embed command: generate embeddings for patterns.
pub async fn run(args: &EmbedArgs) -> Result<()> {
    #[cfg(not(feature = "onnx-embed"))]
    {
        run_hash_embed(args).await
    }

    #[cfg(feature = "onnx-embed")]
    {
        run_onnx_embed(args).await
    }
}

/// Fallback embedding using HashEmbedder when ONNX is unavailable.
#[cfg(not(feature = "onnx-embed"))]
async fn run_hash_embed(args: &EmbedArgs) -> Result<()> {
    println!("\nNagual Embedding Generator (hash mode)");
    println!("{:-<50}", "");
    println!("Note: ONNX embedder unavailable. Using deterministic hash embedder.");
    println!("      Hash embeddings are fast but less accurate than ONNX.");
    println!("      Rebuild with: cargo build --features onnx-embed for ONNX support.\n");

    // Initialize storage with dual-write
    let storage = init_storage(&args.db_path, args.postgres_url.as_deref()).await?;

    // Get all patterns
    let all_patterns = storage.get_recent(100_000).await?;
    let total = all_patterns.len();

    if total == 0 {
        println!("No patterns found in database.");
        return Ok(());
    }

    // Filter to patterns needing embeddings
    let patterns_to_embed: Vec<&Pattern> = if args.force {
        all_patterns.iter().collect()
    } else {
        all_patterns.iter().filter(|p| p.embedding().is_none()).collect()
    };

    let to_embed_count = patterns_to_embed.len();
    let already_have = total - to_embed_count;

    println!("Total patterns: {}", total);
    println!("Already embedded: {}", already_have);
    println!("To embed: {}", to_embed_count);
    println!("Embedder: HashEmbedder (SHAKE-256, 128D)");
    println!();

    if to_embed_count == 0 {
        println!("All patterns already have embeddings. Use --force to regenerate.");
        return Ok(());
    }

    let hash_embedder = HashEmbedder::new();

    // Collect existing embeddings for surprise scoring
    let existing_embeddings: Vec<Vec<f32>> = all_patterns
        .iter()
        .filter_map(|p| p.embedding().map(|e| e.to_vec()))
        .collect();
    let existing_arrays: Vec<_> = existing_embeddings.iter().map(|e| to_array1(e)).collect();
    let has_existing = !existing_arrays.is_empty();

    let mut embedded_count = 0;
    let mut error_count = 0;
    let mut surprise_count = 0;
    let start = std::time::Instant::now();

    for (idx, pattern) in patterns_to_embed.iter().enumerate() {
        let text = format!("{} {}", pattern.problem(), pattern.solution());
        match hash_embedder.embed(&text) {
            Ok(result) => {
                let mut updated = (*pattern).clone();
                updated.set_embedding(result.embedding.clone());
                updated.set_embedding_method("hash");

                // Surprise scoring
                if has_existing {
                    let emb_arr = to_array1(&result.embedding);
                    let max_sim = existing_arrays
                        .iter()
                        .map(|e| cosine_similarity(&emb_arr.view(), &e.view()))
                        .fold(0.0f32, f32::max);
                    updated.set_surprise_score((1.0 - max_sim).clamp(0.0, 1.0));
                    surprise_count += 1;
                }

                match storage.update_pattern(&updated).await {
                    Ok(()) => { embedded_count += 1; }
                    Err(e) => {
                        error_count += 1;
                        if args.verbose {
                            eprintln!("  ERR updating {}: {}", pattern.id(), e);
                        }
                    }
                }
            }
            Err(e) => {
                error_count += 1;
                if args.verbose {
                    eprintln!("  ERR embedding {}: {}", pattern.id(), e);
                }
            }
        }

        // Progress every 100 patterns
        if (idx + 1) % 100 == 0 || idx + 1 == to_embed_count {
            let elapsed = start.elapsed();
            let rate = if elapsed.as_secs_f64() > 0.0 {
                (idx + 1) as f64 / elapsed.as_secs_f64()
            } else {
                0.0
            };
            print!("\r  Embedded {}/{} ({:.0}/s)", idx + 1, to_embed_count, rate);
        }
    }
    println!();

    // Wait for any background PostgreSQL writes
    if resolve_postgres_url(args.postgres_url.as_deref()).is_some() {
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
    }

    let elapsed = start.elapsed();
    println!("\n=== Embedding Complete (hash mode) ===");
    println!("  Embedded: {}", embedded_count);
    println!("  Surprise scored: {}", surprise_count);
    println!("  Errors: {}", error_count);
    println!("  Duration: {:.1}s", elapsed.as_secs_f64());
    if elapsed.as_secs_f64() > 0.0 {
        println!("  Rate: {:.0} patterns/s", embedded_count as f64 / elapsed.as_secs_f64());
    }
    println!("  Database: {}", args.db_path.display());

    Ok(())
}

/// Full ONNX-based embedding pipeline.
#[cfg(feature = "onnx-embed")]
async fn run_onnx_embed(args: &EmbedArgs) -> Result<()> {
    println!("\nNagual Embedding Generator");
    println!("{:-<50}", "");

    // Initialize storage with dual-write
    let storage = init_storage(&args.db_path, args.postgres_url.as_deref()).await?;

    // Get all patterns
    let all_patterns = storage.get_recent(100_000).await?;
    let total = all_patterns.len();

    if total == 0 {
        println!("No patterns found in database.");
        return Ok(());
    }

    // Filter to patterns needing embeddings
    let patterns_to_embed: Vec<&Pattern> = if args.force {
        all_patterns.iter().collect()
    } else {
        all_patterns.iter().filter(|p| p.embedding().is_none()).collect()
    };

    let to_embed_count = patterns_to_embed.len();
    let already_have = total - to_embed_count;

    println!("Total patterns: {}", total);
    println!("Already embedded: {}", already_have);
    println!("To embed: {}", to_embed_count);
    println!("Model: {}", args.model_path);
    println!("Batch size: {}", args.batch_size);
    println!();

    if to_embed_count == 0 {
        println!("All patterns already have embeddings. Use --force to regenerate.");
        return Ok(());
    }

    // Load the ONNX embedder
    let config = EmbedderConfig::dim_128(&args.model_path, &args.tokenizer_path);

    println!("Loading ONNX model...");
    let embedder = match Embedder::new(&config) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("Warning: Failed to load ONNX embedder: {}", e);
            eprintln!("Falling back to HashEmbedder (deterministic, no model files needed).");
            return run_hash_embed_fallback(args, &storage, &all_patterns, &patterns_to_embed).await;
        }
    };
    println!("Model loaded ({}D embeddings)", embedder.embedding_dim());

    // Collect existing embeddings for surprise scoring (as ndarray for cosine_similarity)
    let existing_embeddings: Vec<Vec<f32>> = all_patterns
        .iter()
        .filter_map(|p| p.embedding().map(|e| e.to_vec()))
        .collect();
    let existing_arrays: Vec<_> = existing_embeddings.iter().map(|e| to_array1(e)).collect();
    let has_existing = !existing_arrays.is_empty();

    // Process patterns in batches
    let batch_size = args.batch_size;
    let mut embedded_count = 0;
    let mut error_count = 0;
    let mut surprise_count = 0;
    let mut chunk_count = 0;
    let start = std::time::Instant::now();

    for (batch_idx, chunk) in patterns_to_embed.chunks(batch_size).enumerate() {
        // Prepare texts: combine problem + solution for richer embeddings
        let texts: Vec<String> = chunk.iter().map(|p| {
            format!("{} {}", p.problem(), p.solution())
        }).collect();
        let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();

        // Generate embeddings
        match embedder.embed_batch(&text_refs) {
            Ok(results) => {
                for (pattern, result) in chunk.iter().zip(results.iter()) {
                    // Clone pattern and set embedding
                    let mut updated = (*pattern).clone();
                    updated.set_embedding(result.embedding.clone());
                    updated.set_embedding_method("onnx");

                    // Surprise scoring: cosine distance to nearest existing embedding
                    if has_existing {
                        let emb_arr = to_array1(&result.embedding);
                        let max_sim = existing_arrays
                            .iter()
                            .map(|e| cosine_similarity(&emb_arr.view(), &e.view()))
                            .fold(0.0f32, f32::max);
                        // surprise = 1 - max_similarity (novel = far from existing)
                        let surprise = (1.0 - max_sim).clamp(0.0, 1.0);
                        updated.set_surprise_score(surprise);
                        surprise_count += 1;
                    }

                    // Chunk-level embeddings for long solutions (DESC method)
                    // Split solutions > 600 chars into ~300-char chunks and embed each
                    let solution = pattern.solution();
                    if solution.len() > 600 {
                        let chunks: Vec<&str> = solution
                            .as_bytes()
                            .chunks(300)
                            .map(|c| std::str::from_utf8(c).unwrap_or(""))
                            .filter(|s| s.len() > 50)
                            .collect();
                        if chunks.len() >= 2 {
                            let chunk_results: Vec<Vec<f32>> = chunks
                                .iter()
                                .filter_map(|text| embedder.embed(text).ok().map(|r| r.embedding))
                                .collect();
                            if !chunk_results.is_empty() {
                                updated.set_chunk_embeddings(chunk_results);
                                chunk_count += 1;
                            }
                        }
                    }

                    // Update in storage (dual-write to both SQLite and PostgreSQL)
                    match storage.update_pattern(&updated).await {
                        Ok(()) => {
                            embedded_count += 1;
                        }
                        Err(e) => {
                            error_count += 1;
                            if args.verbose {
                                eprintln!("  ERR updating {}: {}", pattern.id(), e);
                            }
                        }
                    }
                }
            }
            Err(e) => {
                // Fall back to individual embedding
                if args.verbose {
                    eprintln!("  Batch {} failed ({}), falling back to individual", batch_idx, e);
                }
                for pattern in chunk.iter() {
                    let text = format!("{} {}", pattern.problem(), pattern.solution());
                    match embedder.embed(&text) {
                        Ok(result) => {
                            let mut updated = (*pattern).clone();
                            updated.set_embedding(result.embedding.clone());
                            updated.set_embedding_method("onnx");

                            // Surprise scoring (individual fallback)
                            if has_existing {
                                let emb_arr = to_array1(&result.embedding);
                                let max_sim = existing_arrays
                                    .iter()
                                    .map(|e| cosine_similarity(&emb_arr.view(), &e.view()))
                                    .fold(0.0f32, f32::max);
                                updated.set_surprise_score((1.0 - max_sim).clamp(0.0, 1.0));
                                surprise_count += 1;
                            }

                            match storage.update_pattern(&updated).await {
                                Ok(()) => { embedded_count += 1; }
                                Err(e) => {
                                    error_count += 1;
                                    if args.verbose {
                                        eprintln!("  ERR updating {}: {}", pattern.id(), e);
                                    }
                                }
                            }
                        }
                        Err(e) => {
                            error_count += 1;
                            if args.verbose {
                                eprintln!("  ERR embedding {}: {}", pattern.id(), e);
                            }
                        }
                    }
                }
            }
        }

        // Progress
        let processed = (batch_idx + 1) * batch_size.min(to_embed_count);
        let processed = processed.min(to_embed_count);
        let elapsed = start.elapsed();
        let rate = if elapsed.as_secs_f64() > 0.0 {
            processed as f64 / elapsed.as_secs_f64()
        } else {
            0.0
        };
        print!("\r  Embedded {}/{} ({:.0}/s)", processed, to_embed_count, rate);
    }
    println!();

    // Wait for any background PostgreSQL writes
    if resolve_postgres_url(args.postgres_url.as_deref()).is_some() {
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
    }

    // Generate hyperbolic (Poincare ball) embeddings if requested
    if args.hyperbolic {
        println!("\nGenerating hyperbolic embeddings...");
        let hyper_config = crate::ml::HyperbolicConfig::default();
        let hyper_embedder = crate::ml::HyperbolicEmbedder::new(hyper_config);
        let mut hyper_count = 0;

        // Re-fetch patterns that now have embeddings
        let patterns_with_embeddings = storage.get_recent(100_000).await?;

        // For each pattern with an embedding, compute Poincare point
        // (Note: just log that we computed them; actual persistence
        // would require a new column -- for now this validates the pipeline)
        for pattern in &patterns_with_embeddings {
            if let Some(emb) = pattern.embedding() {
                let emb_arr = crate::ml::to_array1(emb);
                let domain_str = pattern.category().to_string();
                match hyper_embedder.embed_from_euclidean(&emb_arr.view(), &domain_str) {
                    Ok(point) => {
                        hyper_count += 1;
                        if hyper_count <= 3 {
                            println!(
                                "  Pattern {}: depth={:.3}, norm={:.3}",
                                &pattern.id().to_string()[..8],
                                point.depth,
                                point.norm()
                            );
                        }
                    }
                    Err(e) => {
                        eprintln!("  Warning: Failed to compute hyperbolic embedding: {}", e);
                    }
                }
            }
        }
        println!("Generated {} hyperbolic embeddings", hyper_count);
    }

    let elapsed = start.elapsed();
    let stats = embedder.stats();

    println!("\n=== Embedding Complete ===");
    println!("  Embedded: {}", embedded_count);
    println!("  Surprise scored: {}", surprise_count);
    println!("  Chunk-embedded: {}", chunk_count);
    println!("  Errors: {}", error_count);
    println!("  Duration: {:.1}s", elapsed.as_secs_f64());
    println!("  Rate: {:.0} patterns/s", embedded_count as f64 / elapsed.as_secs_f64());
    println!("  Tokens processed: {}", stats.tokens_processed);
    if stats.truncated_inputs > 0 {
        println!("  Truncated inputs: {}", stats.truncated_inputs);
    }
    println!("  Database: {}", args.db_path.display());

    Ok(())
}

/// HashEmbedder fallback when ONNX model fails to load at runtime.
/// This is used within the onnx-embed feature gate when the model file is missing or corrupt.
#[cfg(feature = "onnx-embed")]
async fn run_hash_embed_fallback(
    args: &EmbedArgs,
    storage: &crate::reasoning_bank::storage::PatternStorage,
    all_patterns: &[Pattern],
    patterns_to_embed: &[&Pattern],
) -> Result<()> {
    let hash_embedder = HashEmbedder::new();
    let to_embed_count = patterns_to_embed.len();

    println!("\nUsing HashEmbedder (SHAKE-256, 128D)");

    let existing_embeddings: Vec<Vec<f32>> = all_patterns
        .iter()
        .filter_map(|p| p.embedding().map(|e| e.to_vec()))
        .collect();
    let existing_arrays: Vec<_> = existing_embeddings.iter().map(|e| to_array1(e)).collect();
    let has_existing = !existing_arrays.is_empty();

    let mut embedded_count = 0;
    let mut error_count = 0;
    let mut surprise_count = 0;
    let start = std::time::Instant::now();

    for (idx, pattern) in patterns_to_embed.iter().enumerate() {
        let text = format!("{} {}", pattern.problem(), pattern.solution());
        match hash_embedder.embed(&text) {
            Ok(result) => {
                let mut updated = (*pattern).clone();
                updated.set_embedding(result.embedding.clone());
                updated.set_embedding_method("hash");

                if has_existing {
                    let emb_arr = to_array1(&result.embedding);
                    let max_sim = existing_arrays
                        .iter()
                        .map(|e| cosine_similarity(&emb_arr.view(), &e.view()))
                        .fold(0.0f32, f32::max);
                    updated.set_surprise_score((1.0 - max_sim).clamp(0.0, 1.0));
                    surprise_count += 1;
                }

                match storage.update_pattern(&updated).await {
                    Ok(()) => { embedded_count += 1; }
                    Err(e) => {
                        error_count += 1;
                        if args.verbose {
                            eprintln!("  ERR updating {}: {}", pattern.id(), e);
                        }
                    }
                }
            }
            Err(e) => {
                error_count += 1;
                if args.verbose {
                    eprintln!("  ERR embedding {}: {}", pattern.id(), e);
                }
            }
        }

        if (idx + 1) % 100 == 0 || idx + 1 == to_embed_count {
            let elapsed = start.elapsed();
            let rate = if elapsed.as_secs_f64() > 0.0 {
                (idx + 1) as f64 / elapsed.as_secs_f64()
            } else {
                0.0
            };
            print!("\r  Embedded {}/{} ({:.0}/s)", idx + 1, to_embed_count, rate);
        }
    }
    println!();

    if resolve_postgres_url(args.postgres_url.as_deref()).is_some() {
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
    }

    let elapsed = start.elapsed();
    println!("\n=== Embedding Complete (hash fallback) ===");
    println!("  Embedded: {}", embedded_count);
    println!("  Surprise scored: {}", surprise_count);
    println!("  Errors: {}", error_count);
    println!("  Duration: {:.1}s", elapsed.as_secs_f64());
    if elapsed.as_secs_f64() > 0.0 {
        println!("  Rate: {:.0} patterns/s", embedded_count as f64 / elapsed.as_secs_f64());
    }
    println!("  Database: {}", args.db_path.display());

    Ok(())
}
