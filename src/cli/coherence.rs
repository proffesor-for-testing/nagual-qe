//! Coherence command for belief consistency verification

use clap::{Args, Subcommand};
use std::path::PathBuf;
use std::sync::Arc;
use tracing::info;

use crate::coherence::{
    CoherenceAction, CoherenceConfigUpdate, CoherenceGate, CoherenceResult,
    ConflictSeverity, GlobalCoherenceReport,
};
use crate::db::SqliteDb;
use crate::error::NagualError;

/// Coherence command for belief consistency verification
#[derive(Args, Debug)]
pub struct CoherenceCommand {
    #[command(subcommand)]
    pub subcommand: CoherenceSubcommand,

    /// Path to the SQLite database
    #[arg(long, default_value = "nagual.db")]
    pub db_path: PathBuf,
}

#[derive(Subcommand, Debug)]
pub enum CoherenceSubcommand {
    /// Check coherence of a pattern by ID
    Check {
        /// Pattern ID to check
        pattern_id: String,
    },

    /// Test coherence of new content before storing
    Test {
        /// Problem description
        #[arg(long, short = 'p')]
        problem: String,

        /// Solution description
        #[arg(long, short = 's')]
        solution: String,

        /// Domain/category
        #[arg(long, short = 'd')]
        domain: String,
    },

    /// Analyze global coherence across the knowledge base
    Analyze {
        /// Show detailed conflict information
        #[arg(long)]
        detailed: bool,
    },

    /// Configure coherence gate settings (changes are persisted)
    Config {
        /// Energy threshold (0.0-1.0)
        #[arg(long)]
        energy_threshold: Option<f64>,

        /// Similarity threshold for contradiction detection
        #[arg(long)]
        similarity_threshold: Option<f64>,

        /// Maximum conflicts before rejection
        #[arg(long)]
        max_conflicts: Option<usize>,

        /// Enable or disable coherence checking
        #[arg(long)]
        enabled: Option<bool>,

        /// Show current configuration
        #[arg(long)]
        show: bool,
    },
}

impl CoherenceCommand {
    pub async fn execute(&self, json_output: bool) -> Result<(), NagualError> {
        let db = Arc::new(SqliteDb::open(&self.db_path)?);

        match &self.subcommand {
            CoherenceSubcommand::Check { pattern_id } => {
                let gate = CoherenceGate::with_persisted_config(db).await?;
                self.check_pattern(&gate, pattern_id, json_output).await
            }
            CoherenceSubcommand::Test { problem, solution, domain } => {
                let gate = CoherenceGate::with_persisted_config(db).await?;
                self.test_content(&gate, problem, solution, domain, json_output).await
            }
            CoherenceSubcommand::Analyze { detailed } => {
                let gate = CoherenceGate::with_persisted_config(db).await?;
                self.analyze_global(&gate, *detailed, json_output).await
            }
            CoherenceSubcommand::Config {
                energy_threshold,
                similarity_threshold,
                max_conflicts,
                enabled,
                show,
            } => {
                self.handle_config(
                    db,
                    *energy_threshold,
                    *similarity_threshold,
                    *max_conflicts,
                    *enabled,
                    *show,
                    json_output,
                ).await
            }
        }
    }

    async fn check_pattern(
        &self,
        gate: &CoherenceGate,
        pattern_id: &str,
        json_output: bool,
    ) -> Result<(), NagualError> {
        info!("Checking coherence for pattern: {}", pattern_id);

        let result = gate.check_pattern(pattern_id).await?;

        if json_output {
            println!("{}", serde_json::to_string_pretty(&result).unwrap_or_default());
        } else {
            self.print_coherence_result(&result);
        }

        Ok(())
    }

    async fn test_content(
        &self,
        gate: &CoherenceGate,
        problem: &str,
        solution: &str,
        domain: &str,
        json_output: bool,
    ) -> Result<(), NagualError> {
        info!("Testing coherence for new content in domain: {}", domain);

        let result = gate.check(problem, solution, domain).await?;

        if json_output {
            println!("{}", serde_json::to_string_pretty(&result).unwrap_or_default());
        } else {
            println!("Coherence Test Results");
            println!("======================");
            println!();
            println!("Domain: {}", domain);
            println!("Problem: {}", &problem[..problem.len().min(80)]);
            println!();
            self.print_coherence_result(&result);
        }

        Ok(())
    }

    async fn analyze_global(
        &self,
        gate: &CoherenceGate,
        detailed: bool,
        json_output: bool,
    ) -> Result<(), NagualError> {
        info!("Analyzing global coherence");

        let report = gate.analyze_global_coherence().await?;

        if json_output {
            println!("{}", serde_json::to_string_pretty(&report).unwrap_or_default());
        } else {
            self.print_global_report(&report, detailed);
        }

        Ok(())
    }

    async fn handle_config(
        &self,
        db: Arc<SqliteDb>,
        energy_threshold: Option<f64>,
        similarity_threshold: Option<f64>,
        max_conflicts: Option<usize>,
        enabled: Option<bool>,
        show: bool,
        json_output: bool,
    ) -> Result<(), NagualError> {
        // Load existing config
        let mut gate = CoherenceGate::with_persisted_config(db).await?;

        // Check if any updates were provided
        let has_updates = energy_threshold.is_some()
            || similarity_threshold.is_some()
            || max_conflicts.is_some()
            || enabled.is_some();

        if has_updates {
            // Apply updates
            gate.update_config(CoherenceConfigUpdate {
                energy_threshold,
                similarity_threshold,
                max_conflicts,
                check_enabled: enabled,
            }).await?;

            if json_output {
                println!("{}", serde_json::to_string_pretty(gate.config()).unwrap_or_default());
            } else {
                println!("Configuration updated and saved:");
                println!("  Energy Threshold:     {:.2}", gate.config().energy_threshold);
                println!("  Similarity Threshold: {:.2}", gate.config().similarity_threshold);
                println!("  Max Conflicts:        {}", gate.config().max_conflicts);
                println!("  Check Enabled:        {}", gate.config().check_enabled);
            }
        } else if show || !has_updates {
            // Just show current config
            let config = gate.config();
            if json_output {
                println!("{}", serde_json::to_string_pretty(config).unwrap_or_default());
            } else {
                println!("Coherence Gate Configuration");
                println!("============================");
                println!();
                println!("  Energy Threshold:     {:.2} (patterns need >= this to pass)", config.energy_threshold);
                println!("  Similarity Threshold: {:.2} (for contradiction detection)", config.similarity_threshold);
                println!("  Max Conflicts:        {}   (before auto-reject)", config.max_conflicts);
                println!("  Check Enabled:        {}", config.check_enabled);
                println!();
                println!("Configuration is persisted to database.");
                println!("Use --energy-threshold, --similarity-threshold, --max-conflicts, --enabled to modify.");
            }
        }

        Ok(())
    }

    fn print_coherence_result(&self, result: &CoherenceResult) {
        let icon = if result.is_coherent { "●" } else { "○" };
        let status = if result.is_coherent { "COHERENT" } else { "INCOHERENT" };

        println!("{} Status: {}", icon, status);
        println!();
        println!("Energy: {:.3} (threshold: {:.3})", result.energy, result.threshold);
        println!("Supporting Patterns: {}", result.supporting_patterns);
        println!();

        if result.conflicts.is_empty() {
            println!("Conflicts: None detected");
        } else {
            println!("Conflicts ({}):", result.conflicts.len());
            for (i, conflict) in result.conflicts.iter().enumerate() {
                let severity_icon = match conflict.severity {
                    ConflictSeverity::Major => "!!",
                    ConflictSeverity::Moderate => "! ",
                    ConflictSeverity::Minor => "- ",
                };
                println!(
                    "  {}. [{}] {} (similarity: {:.2})",
                    i + 1,
                    severity_icon,
                    conflict.description,
                    conflict.similarity
                );
                println!(
                    "       Patterns: {} vs {}",
                    &conflict.pattern_a_id[..conflict.pattern_a_id.len().min(8)],
                    &conflict.pattern_b_id[..conflict.pattern_b_id.len().min(8)]
                );
            }
        }

        println!();
        println!("Recommendation: {}", result.recommendation);

        match &result.recommendation {
            CoherenceAction::Accept => {
                println!("  Pattern can be safely stored.");
            }
            CoherenceAction::AcceptWithWarning { warnings } => {
                println!("  Pattern can be stored with {} warning(s):", warnings.len());
                for (i, w) in warnings.iter().take(3).enumerate() {
                    println!("    {}. {}", i + 1, w);
                }
            }
            CoherenceAction::RequireReview { conflicts } => {
                println!("  Manual review required for {} conflict(s).", conflicts.len());
            }
            CoherenceAction::Reject { reason } => {
                println!("  Pattern should NOT be stored: {}", reason);
            }
            CoherenceAction::Merge { merge_with } => {
                println!("  Consider merging with pattern: {}", merge_with);
            }
        }
    }

    fn print_global_report(&self, report: &GlobalCoherenceReport, detailed: bool) {
        println!("Global Coherence Report");
        println!("=======================");
        println!();

        let coherence_icon = if report.overall_coherence > 0.8 {
            "●"
        } else if report.overall_coherence > 0.6 {
            "◐"
        } else {
            "○"
        };

        println!(
            "{} Overall Coherence: {:.1}%",
            coherence_icon,
            report.overall_coherence * 100.0
        );
        println!();
        println!("Statistics:");
        println!("  Total Patterns:    {}", report.total_patterns);
        println!("  Sampled Patterns:  {}", report.sampled_patterns);
        println!("  Comparisons Made:  {}", report.comparisons_made);
        println!("  Conflicts Found:   {}", report.conflicts_detected);
        println!();

        if !report.top_domains.is_empty() {
            println!("Top Domains:");
            for (i, (domain, count)) in report.top_domains.iter().take(10).enumerate() {
                println!("  {}. {} ({} patterns)", i + 1, domain, count);
            }
            println!();
        }

        if detailed && !report.sample_conflicts.is_empty() {
            println!("Sample Conflicts:");
            for (i, conflict) in report.sample_conflicts.iter().enumerate() {
                let severity_icon = match conflict.severity {
                    ConflictSeverity::Major => "!!",
                    ConflictSeverity::Moderate => "! ",
                    ConflictSeverity::Minor => "- ",
                };
                println!(
                    "  {}. [{}] {} (similarity: {:.2})",
                    i + 1,
                    severity_icon,
                    &conflict.description[..conflict.description.len().min(60)],
                    conflict.similarity
                );
            }
            println!();
        }

        // Recommendations
        println!("Recommendations:");
        if report.overall_coherence > 0.9 {
            println!("  Knowledge base is highly coherent. No action needed.");
        } else if report.overall_coherence > 0.7 {
            println!("  Good coherence. Consider reviewing flagged conflicts.");
        } else if report.overall_coherence > 0.5 {
            println!("  Moderate coherence. Run 'nagual learn consolidate' to merge similar patterns.");
        } else {
            println!("  Low coherence detected. Manual review recommended.");
            println!("  Use 'nagual coherence check <pattern-id>' to inspect specific conflicts.");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_command_parse() {
        use clap::Parser;

        #[derive(Parser)]
        struct TestCli {
            #[command(subcommand)]
            cmd: TestCmd,
        }

        #[derive(clap::Subcommand)]
        enum TestCmd {
            Coherence(CoherenceCommand),
        }

        let args = vec!["test", "coherence", "analyze"];
        let result = TestCli::try_parse_from(args);
        assert!(result.is_ok());
    }

    #[test]
    fn test_config_command_parse() {
        use clap::Parser;

        #[derive(Parser)]
        struct TestCli {
            #[command(subcommand)]
            cmd: TestCmd,
        }

        #[derive(clap::Subcommand)]
        enum TestCmd {
            Coherence(CoherenceCommand),
        }

        let args = vec!["test", "coherence", "config", "--energy-threshold", "0.6"];
        let result = TestCli::try_parse_from(args);
        assert!(result.is_ok());
    }
}
