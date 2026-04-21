//! Constitution command for displaying Nagual principles and rules.
//!
//! Provides the `nagual constitution` command to view the philosophical
//! principles and operational rules that guide the system.

use std::path::PathBuf;

use clap::Args;
use serde::Serialize;

use crate::constitution::{AdherenceTracker, Constitution, Principle};
use crate::error::Result;

/// Constitution command for viewing principles and rules.
///
/// Displays the 8 philosophical principles (rooted in Castaneda's Tonal/Nagual
/// teachings) and the 5 operational rules that are runtime-enforced.
#[derive(Args, Debug)]
pub struct ConstitutionCommand {
    /// Show a specific principle by number (0-7).
    #[arg(short, long, value_name = "NUMBER")]
    pub principle: Option<u8>,

    /// Show only operational rules (no principles).
    #[arg(long)]
    pub rules: bool,

    /// Show only principles (no rules).
    #[arg(long)]
    pub principles: bool,

    /// Output as JSON.
    #[arg(long)]
    pub json: bool,

    /// Show short summaries instead of full descriptions.
    #[arg(short, long)]
    pub short: bool,

    /// Show a random principle (for startup greeting).
    #[arg(long)]
    pub random: bool,

    /// Show principle adherence metrics.
    #[arg(long)]
    pub adherence: bool,

    /// Time window for adherence metrics in hours (default: 24).
    #[arg(long, default_value = "24")]
    pub window: u32,

    /// Path to SQLite database for adherence metrics.
    #[arg(long, default_value = "./nagual.db")]
    pub db_path: PathBuf,
}

/// JSON output structure for a single principle.
#[derive(Debug, Serialize)]
struct PrincipleOutput {
    number: u8,
    name: String,
    summary: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    quote: Option<String>,
}

/// JSON output structure for operational rules.
#[derive(Debug, Serialize)]
struct RuleOutput {
    number: u8,
    name: String,
    description: String,
}

/// JSON output structure for the full constitution.
#[derive(Debug, Serialize)]
struct ConstitutionOutput {
    version: String,
    enforcement_mode: String,
    principles: Vec<PrincipleOutput>,
    rules: Vec<RuleOutput>,
}

impl ConstitutionCommand {
    /// Execute the constitution command.
    pub async fn run(&self) -> Result<()> {
        // Handle adherence metrics request
        if self.adherence {
            return self.show_adherence_metrics();
        }

        // Handle random principle request (for startup hook)
        if self.random {
            return self.show_random_principle();
        }

        // Handle specific principle request
        if let Some(n) = self.principle {
            return self.show_principle(n);
        }

        // Handle rules-only request
        if self.rules {
            return self.show_rules();
        }

        // Handle principles-only request
        if self.principles {
            return self.show_all_principles();
        }

        // Default: show full constitution
        self.show_full_constitution()
    }

    /// Show a random principle.
    fn show_random_principle(&self) -> Result<()> {
        let p = Principle::random();

        if self.json {
            let output = self.principle_to_output(&p, !self.short);
            println!("{}", serde_json::to_string_pretty(&output)?);
        } else {
            println!("{}", Constitution::startup_greeting());
        }

        Ok(())
    }

    /// Show principle adherence metrics.
    fn show_adherence_metrics(&self) -> Result<()> {
        let tracker = AdherenceTracker::new(&self.db_path);

        // Try to initialize schema (idempotent)
        if let Err(e) = tracker.init_schema() {
            if self.json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&serde_json::json!({
                        "error": format!("Failed to initialize adherence tracking: {}", e)
                    }))?
                );
            } else {
                eprintln!("Warning: Could not initialize adherence tracking: {}", e);
                eprintln!("Adherence metrics may be unavailable.");
            }
        }

        let window = if self.window == 0 {
            None
        } else {
            Some(self.window)
        };

        match tracker.overall_stats(window) {
            Ok(stats) => {
                if self.json {
                    println!("{}", serde_json::to_string_pretty(&stats)?);
                } else {
                    self.print_adherence_stats(&stats);
                }
            }
            Err(e) => {
                if self.json {
                    println!(
                        "{}",
                        serde_json::to_string_pretty(&serde_json::json!({
                            "error": format!("Failed to get adherence stats: {}", e),
                            "total_events": 0,
                            "overall_adherence_rate": 1.0
                        }))?
                    );
                } else {
                    println!();
                    println!("PRINCIPLE ADHERENCE METRICS");
                    println!("----------------------------------------");
                    println!("  No adherence data recorded yet.");
                    println!("  Events will be tracked as principles are invoked.");
                    println!();
                }
            }
        }

        Ok(())
    }

    /// Print adherence statistics in human-readable format.
    fn print_adherence_stats(&self, stats: &crate::constitution::OverallAdherenceStats) {
        println!();
        println!("╔══════════════════════════════════════════════════════════════╗");
        println!("║              PRINCIPLE ADHERENCE METRICS                     ║");
        println!("╚══════════════════════════════════════════════════════════════╝");
        println!();

        let window_str = stats
            .window_hours
            .map(|h| format!("Last {} hours", h))
            .unwrap_or_else(|| "All time".to_string());

        println!("Time window: {}", window_str);
        println!();
        println!(
            "Overall: {:.1}% adherence ({} adhered / {} total)",
            stats.overall_adherence_rate * 100.0,
            stats.total_adhered,
            stats.total_events
        );

        if stats.total_violations > 0 {
            println!(
                "         {} violations recorded",
                stats.total_violations
            );
        }

        println!();
        println!("By Principle:");
        println!("─────────────────────────────────────────────────────────────────");

        for ps in &stats.by_principle {
            if ps.total_events > 0 {
                let bar = self.adherence_bar(ps.adherence_rate, 20);
                println!(
                    "  {} {}: {:.0}% {} ({}/{})",
                    self.principle_icon(&ps.principle),
                    ps.principle.name(),
                    ps.adherence_rate * 100.0,
                    bar,
                    ps.adhered_count,
                    ps.total_events
                );
            }
        }

        // Check if any principles have no data
        let principles_with_data: usize = stats
            .by_principle
            .iter()
            .filter(|ps| ps.total_events > 0)
            .count();

        if principles_with_data == 0 {
            println!("  (No adherence events recorded yet)");
        } else if principles_with_data < 8 {
            println!();
            println!(
                "  ({} of 8 principles have recorded events)",
                principles_with_data
            );
        }

        println!();
    }

    /// Create a visual bar for adherence rate.
    fn adherence_bar(&self, rate: f64, width: usize) -> String {
        let filled = ((rate * width as f64).round() as usize).min(width);
        let empty = width - filled;
        format!("[{}{}]", "█".repeat(filled), "░".repeat(empty))
    }

    /// Show a specific principle by number.
    fn show_principle(&self, n: u8) -> Result<()> {
        match Principle::from_number(n) {
            Some(p) => {
                if self.json {
                    let output = self.principle_to_output(&p, !self.short);
                    println!("{}", serde_json::to_string_pretty(&output)?);
                } else if self.short {
                    println!();
                    println!("{}", p.format_short());
                    println!();
                } else {
                    println!();
                    println!("{}", p.format_full());
                    println!();
                }
                Ok(())
            }
            None => {
                eprintln!("Error: Principle {} does not exist. Valid range: 0-7", n);
                std::process::exit(1);
            }
        }
    }

    /// Show all principles.
    fn show_all_principles(&self) -> Result<()> {
        if self.json {
            let principles: Vec<PrincipleOutput> = Principle::ALL
                .iter()
                .map(|p| self.principle_to_output(p, !self.short))
                .collect();
            println!("{}", serde_json::to_string_pretty(&principles)?);
        } else {
            println!();
            println!("# NAGUAL CONSTITUTION — Philosophical Principles");
            println!();
            println!("Rooted in Carlos Castaneda's Tonal/Nagual teachings from");
            println!("\"Tales of Power\" and \"The Eagle's Gift\".");
            println!();
            println!("---");
            println!();

            for p in &Principle::ALL {
                if self.short {
                    println!("{}", p.format_short());
                    println!();
                } else {
                    println!("{}", p.format_full());
                    println!();
                    println!("---");
                    println!();
                }
            }
        }

        Ok(())
    }

    /// Show operational rules.
    fn show_rules(&self) -> Result<()> {
        let constitution = Constitution::new();

        if self.json {
            let rules = self.rules_to_output();
            let output = serde_json::json!({
                "enforcement_mode": constitution.mode().to_string(),
                "rules": rules
            });
            println!("{}", serde_json::to_string_pretty(&output)?);
        } else {
            println!();
            println!("{}", constitution.format_rules());
            println!();
        }

        Ok(())
    }

    /// Show the full constitution.
    fn show_full_constitution(&self) -> Result<()> {
        let constitution = Constitution::new();

        if self.json {
            let output = ConstitutionOutput {
                version: "1.0.0".to_string(),
                enforcement_mode: constitution.mode().to_string(),
                principles: Principle::ALL
                    .iter()
                    .map(|p| self.principle_to_output(p, !self.short))
                    .collect(),
                rules: self.rules_to_output(),
            };
            println!("{}", serde_json::to_string_pretty(&output)?);
        } else {
            println!();
            println!("╔══════════════════════════════════════════════════════════════╗");
            println!("║                    NAGUAL CONSTITUTION                       ║");
            println!("║                        Version 1.0.0                         ║");
            println!("╚══════════════════════════════════════════════════════════════╝");
            println!();
            println!("\"The warrior's way is harmony — the harmony between tonal and nagual.\"");
            println!();
            println!("════════════════════════════════════════════════════════════════");
            println!("                    PHILOSOPHICAL PRINCIPLES");
            println!("════════════════════════════════════════════════════════════════");
            println!();

            for p in &Principle::ALL {
                if self.short {
                    println!("  {} {}: {}", self.principle_icon(p), p.number(), p.name());
                    println!("    {}", p.summary());
                    println!();
                } else {
                    println!("{}", p.format_full());
                    println!();
                    println!("---");
                    println!();
                }
            }

            println!("════════════════════════════════════════════════════════════════");
            println!("                     OPERATIONAL RULES");
            println!("════════════════════════════════════════════════════════════════");
            println!();
            println!("Enforcement mode: {}", constitution.mode());
            println!();
            println!("  1. NeverDeleteWithoutBackup");
            println!("     Block pattern deletion without a backup within 24 hours.");
            println!();
            println!("  2. AlwaysRecordMAST");
            println!("     Require MAST failure mode classification for failure outcomes.");
            println!();
            println!("  3. SurpriseReview");
            println!("     Flag patterns with surprise > 0.8 for human review before consolidation.");
            println!();
            println!("  4. ConflictEscalation");
            println!("     Create conflict record instead of silently overwriting patterns.");
            println!();
            println!("  5. MinimumRewardForReflex");
            println!("     Require reward >= 0.9 for promotion to reflex tier.");
            println!();
            println!("════════════════════════════════════════════════════════════════");
            println!();
            println!("See NAGUAL_CONSTITUTION.md for the full document.");
            println!("\"The system that learns from itself, improves itself, and never forgets.\"");
            println!();
        }

        Ok(())
    }

    /// Convert a principle to JSON output.
    fn principle_to_output(&self, p: &Principle, full: bool) -> PrincipleOutput {
        PrincipleOutput {
            number: p.number(),
            name: p.name().to_string(),
            summary: p.summary().to_string(),
            description: if full {
                Some(p.description().to_string())
            } else {
                None
            },
            quote: if full {
                Some(p.quote().to_string())
            } else {
                None
            },
        }
    }

    /// Convert rules to JSON output.
    fn rules_to_output(&self) -> Vec<RuleOutput> {
        vec![
            RuleOutput {
                number: 1,
                name: "NeverDeleteWithoutBackup".to_string(),
                description: "Block pattern deletion without a backup within 24 hours".to_string(),
            },
            RuleOutput {
                number: 2,
                name: "AlwaysRecordMAST".to_string(),
                description: "Require MAST failure mode classification for failure outcomes"
                    .to_string(),
            },
            RuleOutput {
                number: 3,
                name: "SurpriseReview".to_string(),
                description:
                    "Flag patterns with surprise > 0.8 for human review before consolidation"
                        .to_string(),
            },
            RuleOutput {
                number: 4,
                name: "ConflictEscalation".to_string(),
                description: "Create conflict record instead of silently overwriting patterns"
                    .to_string(),
            },
            RuleOutput {
                number: 5,
                name: "MinimumRewardForReflex".to_string(),
                description: "Require reward >= 0.9 for promotion to reflex tier".to_string(),
            },
        ]
    }

    /// Get an icon for a principle (for short display).
    fn principle_icon(&self, p: &Principle) -> &'static str {
        match p {
            Principle::SeekTruth => "◉",
            Principle::Partnership => "◎",
            Principle::PartnerCreator => "◈",
            Principle::Impeccability => "◆",
            Principle::EpistemicHumility => "◇",
            Principle::DoNoHarm => "○",
            Principle::Transparency => "◌",
            Principle::WarriorOptimization => "●",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper to create a default test command.
    fn test_cmd() -> ConstitutionCommand {
        ConstitutionCommand {
            principle: None,
            rules: false,
            principles: false,
            json: false,
            short: false,
            random: false,
            adherence: false,
            window: 24,
            db_path: std::env::temp_dir().join("nagual_test_constitution.db"),
        }
    }

    #[tokio::test]
    async fn test_constitution_command_default() {
        let cmd = test_cmd();
        // Should not panic
        let result = cmd.run().await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_constitution_command_specific_principle() {
        let mut cmd = test_cmd();
        cmd.principle = Some(0);
        let result = cmd.run().await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_constitution_command_rules_only() {
        let mut cmd = test_cmd();
        cmd.rules = true;
        let result = cmd.run().await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_constitution_command_principles_only() {
        let mut cmd = test_cmd();
        cmd.principles = true;
        cmd.short = true;
        let result = cmd.run().await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_constitution_command_json() {
        let mut cmd = test_cmd();
        cmd.json = true;
        cmd.short = true;
        let result = cmd.run().await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_constitution_command_random() {
        let mut cmd = test_cmd();
        cmd.random = true;
        let result = cmd.run().await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_constitution_command_adherence() {
        let mut cmd = test_cmd();
        cmd.adherence = true;
        let result = cmd.run().await;
        assert!(result.is_ok());
    }

    #[test]
    fn test_principle_to_output_short() {
        let mut cmd = test_cmd();
        cmd.short = true;
        let output = cmd.principle_to_output(&Principle::SeekTruth, false);
        assert_eq!(output.number, 0);
        assert!(output.description.is_none());
        assert!(output.quote.is_none());
    }

    #[test]
    fn test_principle_to_output_full() {
        let cmd = test_cmd();
        let output = cmd.principle_to_output(&Principle::SeekTruth, true);
        assert_eq!(output.number, 0);
        assert!(output.description.is_some());
        assert!(output.quote.is_some());
    }

    #[test]
    fn test_rules_to_output() {
        let cmd = test_cmd();
        let rules = cmd.rules_to_output();
        assert_eq!(rules.len(), 5);
        assert_eq!(rules[0].name, "NeverDeleteWithoutBackup");
    }

    #[test]
    fn test_adherence_bar() {
        let cmd = test_cmd();
        assert_eq!(cmd.adherence_bar(1.0, 10), "[██████████]");
        assert_eq!(cmd.adherence_bar(0.5, 10), "[█████░░░░░]");
        assert_eq!(cmd.adherence_bar(0.0, 10), "[░░░░░░░░░░]");
    }
}
