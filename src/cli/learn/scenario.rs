//! Scenario management CLI commands.
//!
//! Provides commands for managing validation scenarios (holdout set).
//! Scenarios are test cases for pattern validation that prevent overfitting
//! by evaluating patterns against scenarios they haven't "seen" during training.
//!
//! # Usage Examples
//!
//! ```bash
//! # Create a validation scenario
//! nagual learn scenario create \
//!   --domain rust.async \
//!   --description "Test timeout handling" \
//!   --context "Long-running async operation" \
//!   --expected "Should timeout gracefully" \
//!   --difficulty hard
//!
//! # List scenarios
//! nagual learn scenario list --domain rust
//! nagual learn scenario list --holdout-only
//!
//! # Evaluate a pattern against holdout scenarios
//! nagual learn scenario evaluate <pattern-id> --domain rust
//!
//! # View statistics
//! nagual learn scenario stats
//! ```

use std::path::PathBuf;

use clap::{Args, Subcommand};

use crate::cli::common::init_storage;
use crate::error::Result;
use crate::learning::{Difficulty, Scenario, ScenarioEvaluator, ScenarioId, ScenarioStorage};
use crate::reasoning_bank::pattern::PatternId;

/// Arguments for the scenario subcommand.
#[derive(Args, Debug)]
pub struct ScenarioArgs {
    #[command(subcommand)]
    pub action: ScenarioAction,
}

/// Scenario management actions.
#[derive(Subcommand, Debug)]
pub enum ScenarioAction {
    /// Create a new validation scenario.
    Create {
        /// Domain for the scenario (e.g., "rust.async", "database").
        #[arg(long)]
        domain: String,

        /// Short description of the scenario.
        #[arg(long)]
        description: String,

        /// The problem/situation context.
        #[arg(long)]
        context: String,

        /// What a good solution should do/achieve.
        #[arg(long)]
        expected: Option<String>,

        /// Difficulty level: easy, medium, hard.
        #[arg(long, default_value = "medium")]
        difficulty: String,

        /// Mark as non-holdout (patterns can see it).
        #[arg(long)]
        no_holdout: bool,

        /// Tags for filtering (comma-separated).
        #[arg(long)]
        tags: Option<String>,

        /// Path to SQLite database.
        #[arg(long, default_value = "./nagual.db")]
        db_path: PathBuf,
    },

    /// List scenarios.
    List {
        /// Filter by domain.
        #[arg(long)]
        domain: Option<String>,

        /// Show only holdout scenarios.
        #[arg(long)]
        holdout_only: bool,

        /// Maximum number of scenarios to show.
        #[arg(long, default_value = "20")]
        limit: usize,

        /// Path to SQLite database.
        #[arg(long, default_value = "./nagual.db")]
        db_path: PathBuf,

        /// Output as JSON.
        #[arg(long)]
        json: bool,
    },

    /// Evaluate a pattern against scenarios.
    Evaluate {
        /// Pattern ID to evaluate.
        pattern_id: String,

        /// Filter scenarios by domain.
        #[arg(long)]
        domain: Option<String>,

        /// Only evaluate against holdout scenarios.
        #[arg(long)]
        holdout_only: bool,

        /// Path to SQLite database.
        #[arg(long, default_value = "./nagual.db")]
        db_path: PathBuf,

        /// Output as JSON.
        #[arg(long)]
        json: bool,

        /// Show verbose output.
        #[arg(short, long)]
        verbose: bool,
    },

    /// Show scenario statistics.
    Stats {
        /// Path to SQLite database.
        #[arg(long, default_value = "./nagual.db")]
        db_path: PathBuf,

        /// Output as JSON.
        #[arg(long)]
        json: bool,
    },

    /// Delete a scenario.
    Delete {
        /// Scenario ID to delete.
        scenario_id: String,

        /// Skip confirmation.
        #[arg(long)]
        force: bool,

        /// Path to SQLite database.
        #[arg(long, default_value = "./nagual.db")]
        db_path: PathBuf,
    },
}

/// Execute the scenario command.
pub async fn run(args: &ScenarioArgs) -> Result<()> {
    match &args.action {
        ScenarioAction::Create {
            domain,
            description,
            context,
            expected,
            difficulty,
            no_holdout,
            tags,
            db_path,
        } => {
            let storage = init_storage(db_path, None).await?;
            let scenario_storage = ScenarioStorage::new(storage.adapter());

            // Initialize schema if needed
            scenario_storage.init_schema().await?;

            // Parse tags
            let tag_list: Vec<String> = tags
                .as_ref()
                .map(|t| t.split(',').map(|s| s.trim().to_string()).collect())
                .unwrap_or_default();

            // Build scenario
            let mut builder = Scenario::new(domain)
                .with_description(description)
                .with_input_context(context)
                .with_difficulty(Difficulty::from_str(difficulty))
                .as_holdout(!*no_holdout);

            if let Some(exp) = expected {
                builder = builder.with_expected_behavior(exp);
            }

            if !tag_list.is_empty() {
                builder = builder.with_tags(tag_list);
            }

            let scenario = builder.build();
            let id = scenario.id.clone();

            scenario_storage.create_scenario(&scenario).await?;

            println!("\nScenario Created");
            println!("{:-<50}", "");
            println!("  ID: {}", id);
            println!("  Domain: {}", domain);
            println!("  Description: {}", description);
            println!("  Difficulty: {}", scenario.difficulty);
            println!("  Holdout: {}", scenario.is_holdout);
            if !scenario.tags.is_empty() {
                println!("  Tags: {}", scenario.tags.join(", "));
            }
        }

        ScenarioAction::List {
            domain,
            holdout_only,
            limit,
            db_path,
            json,
        } => {
            let storage = init_storage(db_path, None).await?;
            let scenario_storage = ScenarioStorage::new(storage.adapter());
            scenario_storage.init_schema().await?;

            let scenarios = if let Some(dom) = domain {
                if *holdout_only {
                    scenario_storage.get_holdout_scenarios(dom).await?
                } else {
                    scenario_storage.get_scenarios_for_domain(dom).await?
                }
            } else {
                scenario_storage.list_scenarios(*limit).await?
            };

            if *json {
                println!("{}", serde_json::to_string_pretty(&scenarios)?);
            } else {
                println!("\nValidation Scenarios ({} total)", scenarios.len());
                println!("{:-<70}", "");

                if scenarios.is_empty() {
                    println!("  No scenarios found.");
                } else {
                    for (i, s) in scenarios.iter().take(*limit).enumerate() {
                        let holdout_marker = if s.is_holdout { "[H]" } else { "   " };
                        let pass_rate = s.pass_rate();
                        println!(
                            "{}. {} [{}] {} (pass rate: {:.0}%)",
                            i + 1,
                            holdout_marker,
                            s.difficulty,
                            s.description,
                            pass_rate * 100.0
                        );
                        println!("      Domain: {} | ID: {}", s.domain, &s.id.as_str()[..12]);
                    }
                }
            }
        }

        ScenarioAction::Evaluate {
            pattern_id,
            domain,
            holdout_only,
            db_path,
            json,
            verbose,
        } => {
            let storage = init_storage(db_path, None).await?;
            let scenario_storage = ScenarioStorage::new(storage.adapter());
            scenario_storage.init_schema().await?;

            // Get the pattern
            let pid = PatternId::from_string(pattern_id);
            let pattern = match storage.get_pattern(&pid).await? {
                Some(p) => p,
                None => {
                    eprintln!("Error: Pattern '{}' not found", pattern_id);
                    return Ok(());
                }
            };

            // Get scenarios
            let scenarios = if let Some(dom) = domain {
                if *holdout_only {
                    scenario_storage.get_holdout_scenarios(dom).await?
                } else {
                    scenario_storage.get_scenarios_for_domain(dom).await?
                }
            } else if *holdout_only {
                scenario_storage.get_holdout_scenarios("").await?
            } else {
                scenario_storage.list_scenarios(100).await?
            };

            if scenarios.is_empty() {
                println!("No scenarios found to evaluate against.");
                return Ok(());
            }

            // Evaluate
            let evaluator = ScenarioEvaluator::new();
            let (evaluations, stats) = evaluator.evaluate_batch(&pattern, &scenarios);

            // Record evaluations
            for eval in &evaluations {
                scenario_storage.record_evaluation(eval).await?;
            }

            if *json {
                println!("{}", serde_json::to_string_pretty(&stats)?);
            } else {
                println!("\nPattern Evaluation Results");
                println!("{:-<60}", "");
                println!("  Pattern: {}", pattern_id);
                println!("  Scenarios evaluated: {}", stats.scenarios_evaluated);
                println!("  Scenarios passed: {}", stats.scenarios_passed);
                println!("  Pass rate: {:.1}%", stats.pass_rate() * 100.0);
                println!("  Average score: {:.2}", stats.avg_score);
                println!("  Holdout pass rate: {:.1}%", stats.holdout_pass_rate * 100.0);
                println!("  Holdout count: {}", stats.holdout_count);

                if *verbose {
                    println!("\nDetailed Results:");
                    for (eval, scenario) in evaluations.iter().zip(scenarios.iter()) {
                        let status = if eval.passed { "PASS" } else { "FAIL" };
                        println!(
                            "  [{}] {} - score: {:.2}",
                            status, scenario.description, eval.score
                        );
                        if let Some(feedback) = &eval.feedback {
                            println!("        {}", feedback);
                        }
                    }
                }
            }
        }

        ScenarioAction::Stats { db_path, json } => {
            let storage = init_storage(db_path, None).await?;
            let scenario_storage = ScenarioStorage::new(storage.adapter());
            scenario_storage.init_schema().await?;

            let stats = scenario_storage.get_stats().await?;

            if *json {
                println!("{}", serde_json::to_string_pretty(&stats)?);
            } else {
                println!("\nScenario Statistics");
                println!("{:-<50}", "");
                println!("  Total scenarios: {}", stats.total_scenarios);
                println!("  Holdout scenarios: {}", stats.holdout_scenarios);
                println!("  Total evaluations: {}", stats.total_evaluations);
                println!("  Overall pass rate: {:.1}%", stats.pass_rate * 100.0);
                println!("  Domain count: {}", stats.domain_count);
            }
        }

        ScenarioAction::Delete {
            scenario_id,
            force,
            db_path,
        } => {
            let storage = init_storage(db_path, None).await?;
            let scenario_storage = ScenarioStorage::new(storage.adapter());
            scenario_storage.init_schema().await?;

            let id = ScenarioId::from_string(scenario_id);

            // Get scenario first to show info
            if let Some(scenario) = scenario_storage.get_scenario(&id).await? {
                if !*force {
                    println!("About to delete scenario:");
                    println!("  ID: {}", scenario.id);
                    println!("  Description: {}", scenario.description);
                    println!("  Domain: {}", scenario.domain);
                    println!("  Evaluations: {}", scenario.total_evaluations());
                    println!("\nUse --force to confirm deletion.");
                    return Ok(());
                }

                if scenario_storage.delete_scenario(&id).await? {
                    println!("Scenario '{}' deleted.", scenario_id);
                } else {
                    println!("Failed to delete scenario.");
                }
            } else {
                println!("Scenario '{}' not found.", scenario_id);
            }
        }
    }

    Ok(())
}
