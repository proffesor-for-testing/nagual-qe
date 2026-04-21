//! Strange-loop meta-cognitive status subcommand.
//!
//! Shows meta-cognitive evaluation statistics accumulated during SONA
//! outcome recording. Each outcome passes through the strange-loop
//! evaluator, and the results are tracked here.

use clap::Args;

use crate::error::Result;
use crate::learning::{get_meta_cognitive_stats, get_meta_cognitive_status};

/// Arguments for the strange-loop subcommand.
#[derive(Args, Debug)]
pub struct StrangeLoopArgs {
    /// Show detailed history.
    #[arg(short, long)]
    pub verbose: bool,

    /// Database path for loading persisted history.
    #[arg(long)]
    pub db_path: Option<std::path::PathBuf>,
}

/// Resolve the database path for SQLite fallback.
fn resolve_db_path(args: &StrangeLoopArgs) -> String {
    if let Some(ref p) = args.db_path {
        return p.to_string_lossy().to_string();
    }
    "nagual.db".to_string()
}

/// Run the strange-loop meta-cognitive status command.
pub async fn run(args: &StrangeLoopArgs) -> Result<()> {
    println!("Meta-Cognitive Status (Strange Loop)");
    println!("{:=<50}", "");

    let (mut avg_quality, mut health_rate, mut count) = get_meta_cognitive_stats();
    let mut from_history = false;

    // Fall back to SQLite persistence when in-memory tracker is empty
    if count == 0 {
        let db_path = resolve_db_path(args);
        if let Ok((db_avg, db_health, db_count)) =
            crate::learning::strange_loop::load_stats(&db_path)
        {
            if db_count > 0 {
                avg_quality = db_avg;
                health_rate = db_health;
                count = db_count;
                from_history = true;
            }
        }
    }

    if count == 0 {
        println!("\nNo meta-cognitive evaluations yet.");
        println!("Evaluations accumulate as SONA records pattern outcomes.");
        return Ok(());
    }

    if from_history {
        println!("\n  (loaded from database history)");
    }
    println!("\n  Evaluations: {}", count);
    println!("  Avg Quality: {:.3}", avg_quality);
    println!("  Health Rate: {:.1}%", health_rate * 100.0);

    // Try in-memory first, then SQLite for latest report
    let latest = get_meta_cognitive_status().or_else(|| {
        let db_path = resolve_db_path(args);
        crate::learning::strange_loop::load_latest(&db_path)
            .ok()
            .flatten()
    });

    if let Some(latest) = latest {
        println!("\n  Latest Assessment:");
        println!("    Score:  {:.3}", latest.quality_score);
        println!("    Bonus:  {:.4}", latest.bonus);
        println!(
            "    Status: {}",
            if latest.is_healthy {
                "healthy"
            } else {
                "degraded"
            }
        );
        println!("    Note:   {}", latest.assessment);

        if args.verbose {
            println!("    Iterations: {}", latest.iterations);

            #[cfg(feature = "strange-loop-meta")]
            println!("    Engine: strange-loop (Lipschitz fixed-point)");
            #[cfg(not(feature = "strange-loop-meta"))]
            println!("    Engine: fallback (confidence-adjusted)");
        }
    }

    Ok(())
}
