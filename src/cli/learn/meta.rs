//! Meta-learning subcommand for ADR-035 analysis.
//!
//! Provides meta-learning capabilities including:
//! - EWC++ pattern importance analysis
//! - Transfer learning suggestions between domains
//! - Optimization cycles with Fisher decay
//! - Meta-learning statistics

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

use clap::Args;
use tracing::info;

use crate::cli::common::init_storage;
use crate::error::Result;
use crate::learning::{MetaLearningConfig, MetaLearningEngine};
use crate::reasoning_bank::pattern::Pattern;

/// Arguments for the meta subcommand.
#[derive(Args, Debug)]
pub struct MetaArgs {
    /// Run EWC analysis on patterns (compute importance).
    #[arg(long)]
    pub analyze: bool,

    /// Run transfer learning suggestions.
    #[arg(long)]
    pub transfer: bool,

    /// Run optimization cycle (decay Fisher info, update stats).
    #[arg(long)]
    pub optimize: bool,

    /// Show current meta-learning statistics.
    #[arg(long)]
    pub stats: bool,

    /// Target domain for analysis (optional).
    #[arg(long)]
    pub domain: Option<String>,

    /// Minimum pattern usage count to analyze.
    #[arg(long, default_value = "5")]
    pub min_usage: u32,

    /// Top N patterns to analyze per domain.
    #[arg(long, default_value = "100")]
    pub limit: usize,

    /// Path to SQLite database.
    #[arg(long, default_value = "./nagual.db")]
    pub db_path: PathBuf,

    /// Output as JSON.
    #[arg(long)]
    pub json: bool,
}

/// Run meta-learning analysis (ADR-035).
pub async fn run(args: &MetaArgs) -> Result<()> {
    info!("Running meta-learning analysis");

    // Initialize storage using existing helper
    let storage = init_storage(&args.db_path, None).await?;

    // Get the SQLite database from storage for meta-learning engine
    let sqlite = storage.adapter().sqlite().clone();

    // Initialize meta-learning engine with database
    let meta_config = MetaLearningConfig::default();
    let meta_engine = MetaLearningEngine::with_db(meta_config, sqlite);
    meta_engine.initialize().await?;

    // Get patterns - use FTS search if domain specified, otherwise get top effective
    let patterns = if let Some(domain) = &args.domain {
        // Search by domain using FTS
        storage.fts_search(domain, args.limit).await?
    } else {
        // Get patterns with highest effectiveness (these tend to have more usage data)
        storage.get_top_effective(args.limit).await?
    };

    println!("\nMeta-Learning Analysis (ADR-035)");
    println!("{:=<70}", "");

    // Debug: show what we got
    println!("\nLoaded {} patterns from database", patterns.len());

    // Filter patterns with sufficient usage (using reuse_count as usage proxy)
    let eligible_patterns: Vec<_> = patterns
        .iter()
        .filter(|p| p.reuse_count() >= args.min_usage)
        .collect();

    println!(
        "Filtered to {} patterns (min {} reuse_count, limit {})",
        eligible_patterns.len(),
        args.min_usage,
        args.limit
    );

    if args.analyze || (!args.transfer && !args.optimize && !args.stats) {
        run_ewc_analysis(&meta_engine, &eligible_patterns);
    }

    if args.transfer {
        run_transfer_analysis(&meta_engine, &eligible_patterns);
    }

    if args.optimize {
        run_optimization(&meta_engine).await?;
    }

    if args.stats {
        display_stats(&meta_engine);
    }

    if args.json {
        let stats = meta_engine.stats();
        println!("\n{}", serde_json::to_string_pretty(&stats).unwrap_or_default());
    }

    println!("\nMeta-learning analysis complete.");
    Ok(())
}

/// Run EWC++ pattern importance analysis.
fn run_ewc_analysis(meta_engine: &MetaLearningEngine, eligible_patterns: &[&Pattern]) {
    println!("\n--- EWC++ Pattern Importance Analysis ---\n");

    // Group patterns by domain (category)
    let mut by_domain: HashMap<String, Vec<&Pattern>> = HashMap::new();
    for p in eligible_patterns {
        by_domain
            .entry(p.category().to_string())
            .or_default()
            .push(p);
    }

    let mut total_protected = 0;
    let mut total_analyzed = 0;

    for (domain, domain_patterns) in &by_domain {
        println!("Domain: {} ({} patterns)", domain, domain_patterns.len());
        println!("{:-<60}", "");

        for pattern in domain_patterns.iter().take(10) {
            let reuse = pattern.reuse_count();
            if reuse == 0 {
                continue;
            }

            // Estimate success count from effectiveness and reuse count
            let success_rate = pattern.effectiveness();
            let success_count = (success_rate * reuse as f32).round() as u32;

            // Build outcomes array
            let outcomes: Vec<bool> = (0..reuse).map(|i| i < success_count).collect();

            // Update importance in EWC engine
            let importance = meta_engine.ewc.update_importance(
                pattern.id().as_str(),
                success_count,
                reuse,
                &outcomes,
            );

            total_analyzed += 1;
            let protected = meta_engine.ewc.is_protected(pattern.id().as_str());
            if protected {
                total_protected += 1;
            }

            let problem = pattern.problem();
            let problem_short = if problem.len() > 40 {
                format!("{}...", &problem[..40])
            } else {
                problem.to_string()
            };

            println!(
                "  {} | imp={:.3} fisher={:.3} eff={:.1}% reuse={} {}",
                problem_short,
                importance.importance,
                importance.fisher_info,
                importance.success_rate() * 100.0,
                reuse,
                if protected { "[PROTECTED]" } else { "" }
            );
        }
        println!();
    }

    println!(
        "Summary: {} patterns analyzed, {} protected by EWC++",
        total_analyzed, total_protected
    );
}

/// Run transfer learning suggestions.
fn run_transfer_analysis(meta_engine: &MetaLearningEngine, eligible_patterns: &[&Pattern]) {
    println!("\n--- Transfer Learning Suggestions ---\n");

    // Get unique domains
    let domains: HashSet<_> = eligible_patterns
        .iter()
        .map(|p| p.category().to_string())
        .collect();

    for domain in &domains {
        let suggestions = meta_engine.suggest_transfers(domain);
        if !suggestions.is_empty() {
            println!("Domain '{}' can transfer from:", domain);
            for (source, coef) in suggestions.iter().take(5) {
                println!("  - {} (coefficient: {:.3})", source, coef);
            }
            println!();
        }
    }

    // Show related domains
    println!("Related Domain Pairs (from transfer engine):");
    for transfer in meta_engine.transfer.all_transfers().iter().take(15) {
        println!(
            "  {} <-> {} (coef: {:.3}, success: {}, fail: {})",
            transfer.source_domain,
            transfer.target_domain,
            transfer.transfer_coefficient,
            transfer.successful_transfers,
            transfer.failed_transfers
        );
    }
}

/// Run optimization cycle.
async fn run_optimization(meta_engine: &MetaLearningEngine) -> Result<()> {
    println!("\n--- Running Optimization Cycle ---\n");

    let result = meta_engine.optimize().await?;

    println!("Optimization complete:");
    println!("  Fisher decay applied: {}", result.fisher_decayed);
    println!("  Data persisted: {}", result.persisted);
    println!("  Protected patterns: {}", result.stats.protected_patterns);
    println!("  Forgetting prevented: {}", result.stats.forgetting_prevented);
    println!("  Successful transfers: {}", result.stats.successful_transfers);
    println!("  Failed transfers: {}", result.stats.failed_transfers);

    Ok(())
}

/// Display meta-learning statistics.
fn display_stats(meta_engine: &MetaLearningEngine) {
    println!("\n--- Meta-Learning Statistics ---\n");

    let stats = meta_engine.stats();

    println!("EWC++ Statistics:");
    println!("  Protected patterns: {}", stats.protected_patterns);
    println!("  Forgetting events prevented: {}", stats.forgetting_prevented);

    println!("\nTransfer Learning Statistics:");
    println!("  Successful transfers: {}", stats.successful_transfers);
    println!("  Failed transfers: {}", stats.failed_transfers);

    println!("\nLearning Rate Adaptation:");
    println!("  Rate adjustments: {}", stats.rate_adjustments);

    println!("\nPattern Generalization:");
    println!("  Patterns generalized: {}", stats.patterns_generalized);
    println!("  Templates created: {}", stats.templates_created);

    if let Some(last) = stats.last_optimization {
        println!("\nLast optimization: {}", last);
    }
}
