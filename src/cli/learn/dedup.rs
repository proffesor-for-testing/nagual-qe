//! Dedup command for finding and merging duplicate patterns.

use std::path::PathBuf;

use clap::Args;

use crate::cli::common::init_storage_arc;
use crate::error::Result;
use crate::reasoning_bank::dedup::{
    auto_merge, backfill_content_hashes, print_report, scan_duplicates, DedupConfig,
};

/// Arguments for the dedup subcommand.
#[derive(Args, Debug)]
pub struct DedupArgs {
    /// Scan for duplicates without making changes.
    ///
    /// Shows what duplicates exist but does not modify the database.
    #[arg(long)]
    pub scan: bool,

    /// Automatically merge exact duplicates.
    ///
    /// Keeps the pattern with highest reward as canonical and
    /// deletes duplicates. Aggregates reuse counts to the canonical.
    #[arg(long)]
    pub auto: bool,

    /// Generate detailed report.
    ///
    /// Shows duplicate groups with their canonical and duplicate IDs,
    /// similarity scores, and estimated space savings.
    #[arg(long)]
    pub report: bool,

    /// Similarity threshold for near-duplicates (0.0-1.0).
    ///
    /// Patterns with embedding cosine similarity at or above this
    /// threshold are considered near-duplicates. Default is 0.95.
    #[arg(long, default_value = "0.95")]
    pub threshold: f32,

    /// Path to SQLite database.
    #[arg(long, default_value = "./nagual.db")]
    pub db_path: PathBuf,

    /// Output results as JSON.
    #[arg(long)]
    pub json: bool,

    /// Show verbose output.
    #[arg(short, long)]
    pub verbose: bool,

    /// Backfill content hashes for patterns that don't have them.
    ///
    /// Computes BLAKE3 hash of problem+solution for all patterns
    /// with NULL content_hash. Use --scan to preview without changes.
    #[arg(long)]
    pub backfill_hashes: bool,
}

/// Run the dedup command: find and merge duplicate patterns.
pub async fn run(args: &DedupArgs) -> Result<()> {
    // Initialize storage
    let storage = init_storage_arc(&args.db_path, None).await?;

    // Handle backfill-hashes mode
    if args.backfill_hashes {
        println!("\nNagual Content Hash Backfill");
        println!("{:-<50}", "");
        println!(
            "Mode: {}",
            if args.scan {
                "Dry run (preview)"
            } else {
                "Update database"
            }
        );
        println!();

        let result = backfill_content_hashes(&storage, args.scan).await?;

        if args.json {
            let json = serde_json::to_string_pretty(&result)?;
            println!("{}", json);
        } else {
            println!("Content Hash Backfill Results");
            println!("{:-<50}", "");
            println!("  Total patterns: {}", result.total_patterns);
            println!("  Already hashed: {}", result.already_hashed);
            println!(
                "  {}: {}",
                if args.scan { "Would update" } else { "Updated" },
                result.updated
            );
            println!("  Errors: {}", result.errors.len());
            println!("  Duration: {}ms", result.duration_ms);

            if args.scan {
                println!("\n  DRY RUN: No changes made. Use without --scan to update.");
            }
        }

        return Ok(());
    }

    println!("\nNagual Pattern Deduplication");
    println!("{:-<50}", "");

    // Build configuration
    let config = DedupConfig::default()
        .with_threshold(args.threshold)
        .with_scan_only(!args.auto)
        .with_report(args.report);

    // Run deduplication
    let result = if args.auto {
        println!("Mode: Auto-merge exact duplicates");
        println!("Threshold: {:.2}", args.threshold);
        println!();
        auto_merge(&storage, &config).await?
    } else {
        // Default to scan mode
        println!("Mode: Scan only (use --auto to merge)");
        println!("Threshold: {:.2}", args.threshold);
        println!();
        scan_duplicates(&storage, &config).await?
    };

    // Output results
    if args.json {
        let json = serde_json::to_string_pretty(&result)?;
        println!("{}", json);
    } else {
        print_report(&result);
    }

    if args.verbose && !result.errors.is_empty() {
        println!("\nDetailed errors:");
        for (i, err) in result.errors.iter().enumerate() {
            println!("  {}. {}", i + 1, err);
        }
    }

    Ok(())
}
