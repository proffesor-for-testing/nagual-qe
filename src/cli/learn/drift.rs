//! Embedding drift analysis CLI command.
//!
//! Monitors how pattern embeddings change over time within each domain.
//! High drift may indicate data quality issues or concept evolution.
//! Stagnation may indicate the domain needs fresh contributions.

use clap::Args;
use std::path::PathBuf;

use crate::error::Result;
use crate::learning::{get_domain_drift, get_drift_reports};

/// Arguments for the drift analysis command.
#[derive(Args, Debug)]
pub struct DriftArgs {
    /// Filter to a specific domain
    #[arg(short, long)]
    pub domain: Option<String>,

    /// Database path
    #[arg(long)]
    pub db_path: Option<PathBuf>,
}

/// Resolve the database path for SQLite fallback.
fn resolve_db_path(args: &DriftArgs) -> String {
    if let Some(ref p) = args.db_path {
        return p.to_string_lossy().to_string();
    }
    "nagual.db".to_string()
}

/// Execute the drift analysis command.
pub async fn run(args: &DriftArgs) -> Result<()> {
    if let Some(domain) = &args.domain {
        // Single domain report -- try in-memory first, then SQLite
        match get_domain_drift(domain) {
            Some(report) => {
                println!("\nDrift Analysis");
                println!("{}", "=".repeat(40));
                print_drift_report(&report);
            }
            None => {
                // Fall back to SQLite persistence
                let db_path = resolve_db_path(args);
                match crate::drift::load_drift_reports_for_domain(&db_path, domain) {
                    Ok(reports) if !reports.is_empty() => {
                        println!("\nDrift Analysis (from history)");
                        println!("{}", "=".repeat(40));
                        for report in &reports {
                            print_drift_report(report);
                        }
                    }
                    _ => {
                        println!(
                            "No drift data available for domain '{}'.",
                            domain
                        );
                        println!("Run `nagual learn embed` first to generate embeddings.");
                    }
                }
            }
        }
    } else {
        // All domain reports -- try in-memory first, then SQLite
        let mut reports = get_drift_reports();

        if reports.is_empty() {
            // Fall back to SQLite persistence
            let db_path = resolve_db_path(args);
            if let Ok(persisted) = crate::drift::load_drift_reports(&db_path) {
                reports = persisted;
            }
        }

        if reports.is_empty() {
            println!("No drift data available. Run `nagual learn embed` first to generate embeddings.");
            return Ok(());
        }

        let source = if get_drift_reports().is_empty() {
            " (from history)"
        } else {
            ""
        };
        println!("\nDrift Analysis{source}");
        println!("{}", "=".repeat(40));

        for report in &reports {
            print_drift_report(report);
        }
    }

    Ok(())
}

/// Display a single drift report.
fn print_drift_report(report: &crate::drift::DriftReport) {
    println!("\nDomain: {}", report.domain);

    println!(
        "  Coefficient of Variation: {:.3}",
        report.coefficient_of_variation
    );
    println!("  Trend: {}", report.trend);

    if report.is_drifting {
        println!("  Status: Investigating potential embedding drift");
    } else {
        println!("  Status: Normal operation");
    }

    println!("  Window: {} embeddings", report.window_size);
}
