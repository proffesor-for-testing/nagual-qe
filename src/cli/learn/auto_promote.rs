//! CLI command for auto-promotion of patterns meeting recurrence thresholds.

use clap::Args;

use crate::cli::common::init_storage;
use crate::error::Result;
use crate::learning::auto_promotion::run_auto_promotion;
use crate::reasoning_bank::AutoPromotionCriteria;

/// Auto-promote patterns that meet recurrence thresholds.
///
/// Patterns seen 3+ times across 2+ distinct sessions within 30 days
/// are automatically promoted one tier (booster→crystal, crystal→reflex).
#[derive(Args, Debug)]
pub struct AutoPromoteArgs {
    /// Database path.
    #[arg(long, default_value = "nagual.db")]
    pub db_path: std::path::PathBuf,

    /// Minimum number of times a pattern must be used.
    #[arg(long, default_value = "3")]
    pub min_occurrences: u32,

    /// Minimum distinct sessions/tasks where the pattern was used.
    #[arg(long, default_value = "2")]
    pub min_contexts: u32,

    /// Time window in days to consider.
    #[arg(long, default_value = "30")]
    pub window: u32,

    /// Preview only — do not actually promote.
    #[arg(long)]
    pub dry_run: bool,

    /// Output as JSON.
    #[arg(long)]
    pub json: bool,
}

pub async fn run(args: &AutoPromoteArgs) -> Result<()> {
    let storage = init_storage(&args.db_path, None).await?;

    let criteria = AutoPromotionCriteria {
        min_occurrences: args.min_occurrences,
        min_distinct_contexts: args.min_contexts,
        window_days: args.window,
    };

    if args.dry_run {
        println!("Auto-promotion dry run (no changes will be made)");
        println!(
            "Criteria: {}+ uses, {}+ sessions, {}-day window\n",
            criteria.min_occurrences, criteria.min_distinct_contexts, criteria.window_days
        );
    }

    let result = run_auto_promotion(&storage, &criteria).await?;

    if args.json {
        println!("{}", serde_json::to_string_pretty(&result).unwrap_or_default());
        return Ok(());
    }

    println!("Auto-Promotion Results");
    println!("======================");
    println!("Patterns scanned:  {}", result.patterns_scanned);
    println!("Patterns promoted: {}", result.patterns_promoted);

    if !result.promotions.is_empty() {
        println!("\nPromotions:");
        for p in &result.promotions {
            println!(
                "  {} : {} → {} ({} uses, {} sessions)",
                &p.pattern_id[..8.min(p.pattern_id.len())],
                p.old_tier,
                p.new_tier,
                p.occurrences,
                p.distinct_contexts
            );
        }
    } else {
        println!("\nNo patterns met the promotion criteria.");
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_auto_promote_args_defaults() {
        use clap::Parser;

        #[derive(Parser)]
        struct Cli {
            #[command(flatten)]
            args: AutoPromoteArgs,
        }

        let cli = Cli::parse_from(["test"]);
        assert_eq!(cli.args.min_occurrences, 3);
        assert_eq!(cli.args.min_contexts, 2);
        assert_eq!(cli.args.window, 30);
        assert!(!cli.args.dry_run);
        assert!(!cli.args.json);
    }
}
