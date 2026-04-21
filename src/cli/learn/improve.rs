//! Improve command for self-improvement cycles.
//!
//! Analyzes patterns in the specified domain and generates
//! recommendations for consolidation, archiving, or improvement.

use std::path::PathBuf;

use clap::Args;

use super::common::{apply_recommendation_impl, load_patterns_from_db};
use crate::cli::common::init_storage_arc;
use crate::error::Result;
use crate::learning::{ImprovementConfig, ImprovementPlan, RecommendationType, SelfImprover};

/// Arguments for the improve subcommand.
#[derive(Args, Debug)]
pub struct ImproveArgs {
    /// Domain to analyze (e.g., "rust.async", "database").
    /// Use "all" or omit for global analysis.
    #[arg(value_name = "DOMAIN")]
    pub domain: Option<String>,

    /// Minimum reward threshold for high-performing patterns.
    #[arg(long, default_value = "0.8")]
    pub high_threshold: f32,

    /// Maximum reward threshold for low-performing patterns.
    #[arg(long, default_value = "0.4")]
    pub low_threshold: f32,

    /// Maximum number of recommendations to generate.
    #[arg(long, default_value = "20")]
    pub max_recommendations: usize,

    /// Apply recommendations automatically (dry-run by default).
    #[arg(long)]
    pub apply: bool,

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

/// Run the improvement cycle.
pub async fn run(args: &ImproveArgs) -> Result<()> {
    tracing::info!("Running improvement cycle");

    // Get patterns from database or demo
    let patterns = load_patterns_from_db(&args.db_path, args.demo, 10000).await?;

    if patterns.is_empty() && !args.json {
        println!("\nNo patterns found in database at: {}", args.db_path.display());
        println!("Use 'nagual patterns store' to add patterns, or --demo for sample data.\n");
        return Ok(());
    }

    // Configure the improver
    let config = ImprovementConfig::default()
        .with_high_reward_threshold(args.high_threshold)
        .with_low_reward_threshold(args.low_threshold)
        .with_max_recommendations(args.max_recommendations);

    let improver = SelfImprover::new(config);

    // Run improvement
    let domain = args.domain.as_deref().filter(|d| *d != "all");
    let plan = improver.self_improve(&patterns, domain);

    // Output results
    if args.json {
        let json = serde_json::to_string_pretty(&plan)?;
        println!("{}", json);
    } else {
        display_improvement_plan(&plan, args.verbose);
    }

    // Apply recommendations if requested
    if args.apply && !plan.recommendations.is_empty() {
        println!("\nApplying recommendations...");

        let storage = init_storage_arc(&args.db_path, None).await?;
        let mut applied_count = 0;
        let mut failed_count = 0;

        for recommendation in &plan.recommendations {
            let result = apply_recommendation_impl(&storage, recommendation, args.verbose).await;
            match result {
                Ok(count) => {
                    applied_count += count;
                    if args.verbose {
                        println!(
                            "  [OK] Applied: {} ({} patterns)",
                            recommendation.recommendation_type,
                            count
                        );
                    }
                }
                Err(e) => {
                    failed_count += 1;
                    if args.verbose {
                        println!(
                            "  [FAIL] {}: {}",
                            recommendation.recommendation_type,
                            e
                        );
                    }
                }
            }
        }

        println!(
            "\nApplied {} pattern changes ({} recommendations failed)",
            applied_count, failed_count
        );
    }

    Ok(())
}

/// Display the improvement plan.
fn display_improvement_plan(plan: &ImprovementPlan, verbose: bool) {
    println!("\nImprovement Plan: {}", plan.domain);
    println!("{:-<60}", "");

    // Summary
    println!("\nSummary:");
    println!("  Total patterns analyzed: {}", plan.summary.total_patterns);
    println!("  High performers: {}", plan.summary.high_performers);
    println!("  Low performers: {}", plan.summary.low_performers);
    println!(
        "  Average quality: {:.2}",
        plan.summary.average_quality
    );
    println!(
        "  Expected impact: {:.2}",
        plan.summary.total_expected_impact
    );

    // Opportunities
    if !plan.opportunities.is_empty() {
        println!("\nOpportunities Found:");
        for opp in &plan.opportunities {
            println!(
                "  [{:?}] {} (confidence: {:.0}%, value: {:.0}%)",
                opp.opportunity_type,
                opp.description,
                opp.confidence * 100.0,
                opp.potential_value * 100.0
            );
            if verbose {
                println!("    Patterns: {}", opp.pattern_ids.len());
                for (k, v) in &opp.metrics {
                    println!("    {}: {:.2}", k, v);
                }
            }
        }
    }

    // Recommendations
    if !plan.recommendations.is_empty() {
        println!("\nRecommendations ({}):", plan.recommendations.len());
        for rec in &plan.recommendations {
            let icon = match rec.recommendation_type {
                RecommendationType::Consolidate => "[C]",
                RecommendationType::Archive => "[A]",
                RecommendationType::Improve => "[I]",
                RecommendationType::Split => "[S]",
                RecommendationType::Review => "[R]",
                RecommendationType::Promote => "[P]",
            };
            println!(
                "  {} {} [Priority: {}]",
                icon,
                rec.recommendation_type,
                rec.priority
            );
            println!("     {}", rec.rationale);
            if verbose {
                println!("     Patterns: {}", rec.target_patterns.len());
                println!("     Expected impact: {:.2}", rec.expected_impact);
                if let Some(ref details) = rec.details {
                    println!("     Details: {}", details);
                }
            }
        }
    }

    println!("\nGenerated at: {}", plan.generated_at.format("%Y-%m-%d %H:%M:%S UTC"));
}
