//! Pattern consolidation command.
//!
//! Consolidates similar patterns using cosine similarity on embeddings,
//! archives low-performers, and detects cross-domain wormholes.

use std::path::PathBuf;
use std::sync::Arc;

use clap::Args;

use crate::cli::common::init_storage;
use crate::constitution::{Constitution, Operation, OperationContext};
use crate::error::Result;
use crate::events::{EventBus, NagualEvent};
use crate::learning::{consolidate_patterns, PatternConsolidationConfig};
use crate::profdag::wormhole::{WormholeConfig, WormholeManager};
use crate::profdag::wormhole_detector::{DetectorConfig, WormholeDetector};

/// Arguments for the consolidate subcommand.
#[derive(Args, Debug)]
pub struct ConsolidateArgs {
    /// Trigger type: manual, time, or count.
    #[arg(long, default_value = "manual")]
    pub trigger: String,

    /// Similarity threshold for consolidation (0.0 - 1.0).
    #[arg(long, default_value = "0.9")]
    pub similarity: f32,

    /// Auto-archive patterns below this reward threshold.
    #[arg(long)]
    pub archive_threshold: Option<f32>,

    /// Minimum age in days for auto-archiving.
    #[arg(long, default_value = "30")]
    pub archive_min_age: i64,

    /// Dry-run mode (show what would happen).
    #[arg(long)]
    pub dry_run: bool,

    /// Path to SQLite database.
    #[arg(long, default_value = "./nagual.db")]
    pub db_path: PathBuf,

    /// Output results as JSON.
    #[arg(long)]
    pub json: bool,

    /// Show verbose output.
    #[arg(short, long)]
    pub verbose: bool,
}

/// Run the consolidation command.
pub async fn run(args: &ConsolidateArgs) -> Result<()> {
    tracing::info!("Running pattern consolidation");

    // Initialize storage
    let storage = init_storage(&args.db_path, None).await?;

    // Create the real consolidation config
    let consolidation_config = PatternConsolidationConfig {
        similarity_threshold: args.similarity,
        dry_run: args.dry_run,
        max_patterns_to_process: 10000,
        ..Default::default()
    };

    // Constitution check before consolidation (F08)
    let constitution = Constitution::new();
    let ctx = OperationContext {
        operation: Operation::Consolidate,
        pattern_id: None,
        reward: None,
        tier: None,
        surprise_score: None,
        has_recent_backup: false,
        failure_mode: None,
        domain: None,
    };
    if !constitution.is_allowed(&ctx) {
        eprintln!("Constitution blocked consolidation. Ensure a recent backup exists or disable enforcement.");
        return Ok(());
    }

    // Run real consolidation (uses embeddings for cosine similarity)
    println!("\nRunning pattern consolidation...");
    println!("  Similarity threshold: {:.2}", args.similarity);
    println!("  Dry run: {}", args.dry_run);
    println!();

    let result = consolidate_patterns(&storage, &consolidation_config).await?;

    // Emit ConsolidationCompleted event (F16)
    if !result.dry_run && result.patterns_consolidated > 0 {
        let event_bus = EventBus::new();
        let merged_ids: Vec<String> = result
            .groups
            .iter()
            .flat_map(|g| g.merged_ids.iter().map(|id| id.to_string()))
            .collect();
        event_bus.publish_sync(NagualEvent::consolidation_completed(
            result.patterns_consolidated,
            0,
            merged_ids,
        ));
    }

    // Run cross-domain wormhole detection post-consolidation (F05)
    if !result.dry_run && result.patterns_consolidated > 0 {
        let adapter = storage.adapter().clone();
        match WormholeManager::new(adapter.clone(), WormholeConfig::default()).await {
            Ok(wm) => {
                let wm = Arc::new(wm);
                match WormholeDetector::new(adapter, wm, DetectorConfig::default()).await {
                    Ok(detector) => {
                        match detector.detect_cross_domain_wormholes().await {
                            Ok(candidates) => {
                                if !candidates.is_empty() {
                                    println!(
                                        "\n  Cross-domain wormholes detected: {}",
                                        candidates.len()
                                    );
                                    for c in candidates.iter().take(5) {
                                        println!(
                                            "    {} <-> {} (score: {:.2})",
                                            c.source_id, c.target_id, c.score
                                        );
                                    }
                                }
                            }
                            Err(e) => {
                                tracing::debug!("Wormhole detection skipped: {}", e);
                            }
                        }
                    }
                    Err(e) => tracing::debug!("Wormhole detector init skipped: {}", e),
                }
            }
            Err(e) => tracing::debug!("Wormhole manager init skipped: {}", e),
        }
    }

    // Display results
    println!("Consolidation Result");
    println!("{:-<60}", "");
    println!("  Patterns analyzed: {}", result.patterns_processed);
    println!("  Similar groups found: {}", result.groups_formed);
    println!("  Patterns consolidated: {}", result.patterns_consolidated);
    println!("  Duration: {}ms", result.duration_ms);

    if result.dry_run {
        println!("\n  DRY RUN: No changes were made.");
    } else if result.patterns_consolidated > 0 {
        println!("\n  Changes applied successfully.");
    } else {
        println!("\n  No similar patterns found above threshold.");
    }

    if !result.errors.is_empty() {
        println!("\n  Errors:");
        for err in &result.errors {
            println!("    - {}", err);
        }
    }

    if args.verbose && !result.groups.is_empty() {
        println!("\n  Groups:");
        for (i, group) in result.groups.iter().enumerate() {
            println!(
                "    {}. Primary: {} | Merged: {} | Similarity: {:.2}",
                i + 1,
                group.primary_id,
                group.merged_ids.len(),
                group.average_similarity
            );
        }
    }

    if args.json {
        let json = serde_json::to_string_pretty(&result)?;
        println!("{}", json);
    }

    Ok(())
}
