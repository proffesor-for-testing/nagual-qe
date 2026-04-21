//! CLI command for cross-domain transfer learning.
//!
//! Provides `nagual learn transfer {status,domains,apply,plateau}` subcommands
//! that interact with the domain expansion engine powered by
//! ruvector-domain-expansion.

use clap::{Args, Subcommand};

use crate::error::Result;

/// Resolve the SQLite database path for reading persisted domain state.
#[cfg(feature = "domain-expansion")]
fn resolve_db_path_for_transfer() -> String {
    if let Ok(path) = std::env::var("NAGUAL_DB_PATH") {
        if !path.is_empty() {
            return path;
        }
    }
    if let Ok(home) = std::env::var("HOME") {
        let config_path = std::path::Path::new(&home)
            .join(".nagual")
            .join("config.toml");
        if let Ok(content) = std::fs::read_to_string(config_path) {
            for line in content.lines() {
                let trimmed = line.trim();
                if trimmed.starts_with("sqlite_path") {
                    if let Some(value) = trimmed.split('=').nth(1) {
                        let path = value.trim().trim_matches('"').trim_matches('\'');
                        if !path.is_empty() {
                            return path.to_string();
                        }
                    }
                }
            }
        }
    }
    "nagual.db".to_string()
}

/// Cross-domain transfer learning commands.
#[derive(Args, Debug)]
pub struct TransferArgs {
    #[command(subcommand)]
    pub action: TransferAction,
}

/// Transfer subcommands.
#[derive(Subcommand, Debug)]
pub enum TransferAction {
    /// Show domain expansion status.
    Status,

    /// List registered domains.
    Domains,

    /// Transfer priors from source to target domain.
    Apply {
        /// Source domain name.
        source: String,
        /// Target domain name.
        target: String,
    },

    /// Check if a domain has plateaued.
    Plateau {
        /// Domain to check.
        domain: String,
    },
}

pub async fn run(args: &TransferArgs) -> Result<()> {
    match &args.action {
        TransferAction::Status => {
            println!("Domain Expansion Status");
            println!("{:=<50}", "");

            #[cfg(feature = "domain-expansion")]
            {
                if let Some(health) = crate::learning::get_expansion_health() {
                    println!(
                        "  Learning:     {}",
                        if health.is_learning {
                            "active"
                        } else {
                            "stalled"
                        }
                    );
                    println!(
                        "  Diverse:      {}",
                        if health.is_diverse { "yes" } else { "no" }
                    );
                    println!(
                        "  Exploring:    {}",
                        if health.is_exploring { "yes" } else { "no" }
                    );
                    println!("  Pareto front: {} solutions", health.pareto_size);
                    println!("  Curiosity:    {} visits", health.curiosity_total_visits);
                    println!("  Plateaus:     {} total", health.total_plateaus);
                } else {
                    println!("  No domain expansion data yet.");
                }
            }
            #[cfg(not(feature = "domain-expansion"))]
            {
                println!("  domain-expansion feature not enabled");
            }

            // Show persisted prior state from SQLite (if available).
            #[cfg(feature = "domain-expansion")]
            {
                let db_path = resolve_db_path_for_transfer();
                match crate::learning::domain_expansion::load_domain_state(&db_path) {
                    Ok(states) if !states.is_empty() => {
                        println!();
                        println!("  Persisted priors ({} domains):", states.len());
                        for (domain, json) in &states {
                            let cycles = json
                                .get("training_cycles")
                                .and_then(|v| v.as_u64())
                                .unwrap_or(0);
                            let buckets = json
                                .get("bucket_priors")
                                .and_then(|v| v.as_object())
                                .map(|o| o.len())
                                .unwrap_or(0);
                            println!(
                                "    - {} ({} cycles, {} buckets)",
                                domain, cycles, buckets
                            );
                        }
                    }
                    _ => {}
                }
            }

            Ok(())
        }
        TransferAction::Domains => {
            let domains = crate::learning::get_expansion_domains();
            if domains.is_empty() {
                println!(
                    "No domains registered yet. Domains are auto-registered as SONA records outcomes."
                );
            } else {
                println!("Registered domains ({}):", domains.len());
                for d in &domains {
                    println!("  - {}", d);
                }
            }
            Ok(())
        }
        TransferAction::Apply { source, target } => {
            match crate::learning::initiate_transfer(source, target) {
                Ok(msg) => println!("{}", msg),
                Err(e) => println!("Transfer failed: {}", e),
            }
            Ok(())
        }
        TransferAction::Plateau { domain } => {
            let status = crate::learning::check_domain_plateau(domain);
            println!("Domain '{}': {}", domain, status);
            Ok(())
        }
    }
}
