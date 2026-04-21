//! Insights subcommand for domain analytics.
//!
//! Displays aggregated metrics, trends, and top patterns for a given domain
//! with time-windowed analysis.

use std::path::PathBuf;

use clap::Args;

use super::common::{load_patterns_from_db, parse_time_windows};
use crate::error::Result;
use crate::learning::{aggregate_insights, DomainInsights, InsightsConfig};

/// Arguments for the insights subcommand.
#[derive(Args, Debug)]
pub struct InsightsArgs {
    /// Domain to analyze (e.g., "rust", "database.postgres").
    /// Use "all" or omit for global insights.
    #[arg(value_name = "DOMAIN")]
    pub domain: Option<String>,

    /// Time windows for analysis (comma-separated: 7d,30d,90d).
    #[arg(long, default_value = "7d,30d,90d")]
    pub windows: String,

    /// Number of top patterns to show.
    #[arg(long, default_value = "10")]
    pub top: usize,

    /// Include child domain breakdown.
    #[arg(long)]
    pub children: bool,

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

/// Run the insights command.
pub async fn run(args: &InsightsArgs) -> Result<()> {
    tracing::info!("Generating domain insights");

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

    // Parse time windows
    let windows = parse_time_windows(&args.windows);

    // Configure insights
    let config = InsightsConfig::default()
        .with_time_windows(windows)
        .with_top_patterns_count(args.top)
        .with_child_domains(args.children);

    // Generate insights
    let domain = args.domain.as_deref().unwrap_or("");
    let insights = aggregate_insights(&patterns, domain, &config);

    // Output results
    if args.json {
        let json = serde_json::to_string_pretty(&insights)?;
        println!("{}", json);
    } else {
        display_insights(&insights, args.verbose);
    }

    Ok(())
}

/// Display domain insights.
fn display_insights(insights: &DomainInsights, verbose: bool) {
    let domain_name = if insights.domain.is_empty() {
        "All Domains"
    } else {
        &insights.domain
    };

    println!("\nDomain Insights: {}", domain_name);
    println!("{:-<60}", "");

    // Overview
    println!("\nOverview:");
    println!("  Total patterns: {}", insights.total_patterns);
    println!(
        "  Patterns with reliable metrics: {}",
        insights.patterns_with_reliable_metrics
    );
    println!("  Total usage: {}", insights.total_usage);

    // Metrics
    println!("\nMetrics:");
    println!("  Success rate: {:.1}%", insights.success_rate * 100.0);
    println!("  Average reward: {:.3}", insights.avg_reward);
    println!("  Average effectiveness: {:.3}", insights.avg_effectiveness);
    println!("  Average confidence: {:.3}", insights.avg_confidence);

    // Trend
    println!("\nTrend: {:?}", insights.trend);
    if verbose {
        if let Some(ref trend) = insights.trend_details {
            println!("  Reward change: {:+.1}%", trend.reward_change_pct);
            println!(
                "  Success rate change: {:+.1}%",
                trend.success_rate_change_pct
            );
            println!("  Usage growth: {:+.1}%", trend.usage_growth_pct);
            println!("  New patterns: {}", trend.new_patterns);
        }
    }

    // Time Window Analysis
    if !insights.window_analysis.is_empty() {
        println!("\nTime Window Analysis:");
        let mut windows: Vec<_> = insights.window_analysis.iter().collect();
        windows.sort_by_key(|(k, _)| k.as_str());

        for (label, analysis) in windows {
            println!(
                "  {}: {} patterns, avg reward {:.3}, success rate {:.1}%",
                label,
                analysis.pattern_count,
                analysis.avg_reward,
                analysis.avg_success_rate * 100.0
            );
            if verbose {
                println!(
                    "    New: {}, Updated: {}, Usage: {}",
                    analysis.new_patterns, analysis.updated_patterns, analysis.total_usage
                );
            }
        }
    }

    // Top Patterns
    if !insights.top_patterns.is_empty() {
        println!("\nTop Patterns:");
        for (i, pattern) in insights.top_patterns.iter().enumerate().take(5) {
            println!(
                "  {}. {} (reward: {:.3}, quality: {:.3})",
                i + 1,
                pattern.problem_summary,
                pattern.reward,
                pattern.quality_score
            );
            if verbose {
                println!(
                    "     Usage: {}, Effectiveness: {:.3}",
                    pattern.usage_count, pattern.effectiveness
                );
            }
        }
    }

    // Child domains
    if !insights.child_domains.is_empty() {
        println!("\nChild Domains:");
        for child in &insights.child_domains {
            let trend_icon = match child.trend {
                crate::learning::Trend::Improving => "+",
                crate::learning::Trend::Declining => "-",
                crate::learning::Trend::Stable => "=",
                crate::learning::Trend::Unknown => "?",
            };
            println!(
                "  [{}] {}: {} patterns, avg reward {:.3}",
                trend_icon, child.domain, child.pattern_count, child.avg_reward
            );
        }
    }

    println!(
        "\nGenerated at: {}",
        insights.generated_at.format("%Y-%m-%d %H:%M:%S UTC")
    );
}
