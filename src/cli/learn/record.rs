//! Record command for pattern outcome tracking.

use std::path::PathBuf;

use clap::Args;

use crate::cli::common::init_storage_arc;
use crate::error::Result;
use crate::events::{EventBus, NagualEvent};
use crate::learning::{Outcome, SonaLearner};
use crate::reasoning_bank::pattern::{FailureMode, PatternId};

/// Arguments for the record subcommand.
#[derive(Args, Debug)]
pub struct RecordArgs {
    /// Pattern ID that was applied.
    #[arg(value_name = "PATTERN_ID")]
    pub pattern_id: String,

    /// Outcome: success, failure, partial, or skip.
    #[arg(value_name = "OUTCOME")]
    pub outcome: String,

    /// Optional feedback or notes about the outcome.
    #[arg(short, long)]
    pub feedback: Option<String>,

    /// Latency of the operation in milliseconds.
    #[arg(long)]
    pub latency_ms: Option<u64>,

    /// Session ID for tracking.
    #[arg(long)]
    pub session_id: Option<String>,

    /// Failure mode classification (MAST taxonomy).
    /// Options: specification, misalignment, verification, resource, unknown.
    #[arg(long)]
    pub failure_mode: Option<String>,

    /// Path to SQLite database.
    #[arg(long, default_value = "./nagual.db")]
    pub db_path: PathBuf,

    /// Output results as JSON.
    #[arg(long)]
    pub json: bool,

    /// Show verbose output.
    #[arg(short, long)]
    pub verbose: bool,
}

/// Output structure for record command.
#[derive(serde::Serialize)]
struct RecordOutput {
    pattern_id: String,
    outcome: String,
    reward: f32,
    feedback: Option<String>,
    latency_ms: Option<u64>,
    session_id: Option<String>,
    recorded_at: String,
    success: bool,
    message: String,
}

/// Run the record command.
pub async fn run(args: &RecordArgs) -> Result<()> {
    tracing::info!("Recording outcome for pattern: {}", args.pattern_id);

    // Parse outcome to Outcome enum
    let outcome = match args.outcome.to_lowercase().as_str() {
        "success" | "true" | "1" => Outcome::Success,
        "failure" | "fail" | "false" | "0" => Outcome::Failure,
        "partial" => Outcome::PartialSuccess,
        "skip" | "skipped" | "neutral" => Outcome::Neutral,
        _ => {
            let msg = format!(
                "Invalid outcome: '{}'. Use: success, failure, partial, or skip",
                args.outcome
            );
            if args.json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&serde_json::json!({
                        "error": msg,
                        "error_type": "invalid_outcome"
                    }))?
                );
            } else {
                eprintln!("Error: {}", msg);
            }
            return Ok(());
        }
    };

    // Initialize storage and learner for database persistence
    let storage = match init_storage_arc(&args.db_path, None).await {
        Ok(s) => s,
        Err(e) => {
            let msg = format!("Failed to open database: {}", e);
            if args.json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&serde_json::json!({
                        "error": msg,
                        "error_type": "database_error"
                    }))?
                );
            } else {
                eprintln!("Error: {}", msg);
            }
            return Ok(());
        }
    };

    // Set failure mode on the pattern if outcome is failure and --failure-mode is provided
    if let Some(ref mode_str) = args.failure_mode {
        let mode = FailureMode::from(mode_str.as_str());
        let pattern_id_for_mode = PatternId::from_string(&args.pattern_id);
        if let Ok(Some(mut pattern)) = storage.get_pattern(&pattern_id_for_mode).await {
            pattern.set_failure_mode(mode);
            let _ = storage.update_pattern(&pattern).await;
            if args.verbose {
                println!("Failure mode set: {}", mode_str);
            }
        }
    }

    let learner = SonaLearner::new(storage);
    let pattern_id = PatternId::from_string(&args.pattern_id);

    // Record the outcome using SonaLearner which persists to database
    let reward = match learner
        .record_outcome(&pattern_id, outcome, args.feedback.clone())
        .await
    {
        Ok(r) => r,
        Err(e) => {
            let msg = format!("Failed to record outcome: {}", e);
            if args.json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&serde_json::json!({
                        "error": msg,
                        "error_type": "record_error",
                        "pattern_id": args.pattern_id
                    }))?
                );
            } else {
                eprintln!("Error: {}", msg);
                eprintln!(
                    "Pattern ID '{}' may not exist in the database.",
                    args.pattern_id
                );
            }
            return Ok(());
        }
    };

    // Emit OutcomeRecorded event (F16)
    let event_bus = EventBus::new();
    event_bus.publish_sync(NagualEvent::outcome_recorded(
        &args.pattern_id,
        &args.outcome,
        reward,
        args.feedback.clone(),
    ));

    let now = chrono::Utc::now();
    let record_output = RecordOutput {
        pattern_id: args.pattern_id.clone(),
        outcome: outcome.to_string(),
        reward,
        feedback: args.feedback.clone(),
        latency_ms: args.latency_ms,
        session_id: args.session_id.clone(),
        recorded_at: now.to_rfc3339(),
        success: true,
        message: "Outcome recorded and persisted to database".to_string(),
    };

    if args.json {
        println!("{}", serde_json::to_string_pretty(&record_output)?);
    } else {
        println!("\nOutcome Recorded");
        println!("{:-<50}", "");
        println!("Pattern ID: {}", record_output.pattern_id);
        println!("Outcome: {}", record_output.outcome);
        println!("Reward: {:.2}", record_output.reward);
        if let Some(ref feedback) = record_output.feedback {
            println!("Feedback: {}", feedback);
        }
        if let Some(latency) = record_output.latency_ms {
            println!("Latency: {}ms", latency);
        }
        if args.verbose {
            if let Some(ref session) = record_output.session_id {
                println!("Session: {}", session);
            }
            println!("Recorded at: {}", record_output.recorded_at);
            println!("Database: {}", args.db_path.display());
        }
        println!("{:-<50}\n", "");
    }

    Ok(())
}
