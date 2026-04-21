//! Recommendations command for viewing improvement suggestions.

use std::path::PathBuf;

use clap::Args;

use super::common::load_patterns_from_db;
use crate::error::Result;
use crate::learning::{ImprovementConfig, Recommendation, RecommendationType, SelfImprover};

/// Arguments for the recommendations subcommand.
#[derive(Args, Debug)]
pub struct RecommendationsArgs {
    /// Filter by domain.
    #[arg(long)]
    pub domain: Option<String>,

    /// Filter by recommendation type (consolidate, archive, improve, split, review, promote).
    #[arg(long, value_name = "TYPE")]
    pub filter_type: Option<String>,

    /// Maximum number of recommendations to show.
    #[arg(long, default_value = "20")]
    pub max: usize,

    /// Minimum priority level (1-10).
    #[arg(long, default_value = "1")]
    pub min_priority: u8,

    /// Path to SQLite database.
    #[arg(long, default_value = "./nagual.db")]
    pub db_path: PathBuf,

    /// Output results as JSON.
    #[arg(long)]
    pub json: bool,

    /// Show verbose output.
    #[arg(short, long)]
    pub verbose: bool,

    /// Use demo data for testing.
    #[arg(long)]
    pub demo: bool,
}

/// Run the recommendations command.
pub async fn run(args: &RecommendationsArgs) -> Result<()> {
    tracing::info!("Fetching recommendations");

    // Get patterns from database or demo
    let patterns = load_patterns_from_db(&args.db_path, args.demo, 10000).await?;

    if patterns.is_empty() && !args.json {
        println!(
            "\nNo patterns found in database at: {}",
            args.db_path.display()
        );
        println!("Use 'nagual patterns store' to add patterns, or --demo for sample data.\n");
        return Ok(());
    }

    // Run improvement to get recommendations
    let config = ImprovementConfig::default();
    let improver = SelfImprover::new(config);
    let plan = improver.self_improve(&patterns, args.domain.as_deref());

    // Filter recommendations
    let mut recommendations: Vec<_> = plan
        .recommendations
        .into_iter()
        .filter(|r| r.priority >= args.min_priority)
        .filter(|r| {
            if let Some(ref filter) = args.filter_type {
                r.recommendation_type.to_string() == *filter
            } else {
                true
            }
        })
        .take(args.max)
        .collect();

    // Sort by priority descending
    recommendations.sort_by(|a, b| b.priority.cmp(&a.priority));

    // Output
    if args.json {
        let json = serde_json::to_string_pretty(&recommendations)?;
        println!("{}", json);
    } else {
        display_recommendations(&recommendations, args.verbose);
    }

    Ok(())
}

/// Display recommendations in human-readable format.
fn display_recommendations(recommendations: &[Recommendation], verbose: bool) {
    println!("\nRecommendations ({} total)", recommendations.len());
    println!("{:-<60}", "");

    if recommendations.is_empty() {
        println!("\nNo recommendations found.");
        return;
    }

    for (i, rec) in recommendations.iter().enumerate() {
        let icon = match rec.recommendation_type {
            RecommendationType::Consolidate => "[CONSOLIDATE]",
            RecommendationType::Archive => "[ARCHIVE]",
            RecommendationType::Improve => "[IMPROVE]",
            RecommendationType::Split => "[SPLIT]",
            RecommendationType::Review => "[REVIEW]",
            RecommendationType::Promote => "[PROMOTE]",
        };

        println!(
            "\n{}. {} Priority: {} | Impact: {:.0}%",
            i + 1,
            icon,
            rec.priority,
            rec.expected_impact * 100.0
        );
        println!("   Domain: {}", rec.domain);
        println!("   {}", rec.rationale);
        println!("   Affects {} patterns", rec.target_patterns.len());

        if verbose {
            if let Some(ref details) = rec.details {
                println!("   Details: {}", details);
            }
            println!(
                "   Generated: {}",
                rec.generated_at.format("%Y-%m-%d %H:%M UTC")
            );
        }
    }
}
