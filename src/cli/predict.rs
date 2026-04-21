//! Prediction CLI commands.
//!
//! Provides commands for:
//! - Creating new predictions
//! - Resolving predictions with actual outcomes
//! - Listing predictions with filters
//! - Viewing calibration reports
//!
//! # Usage Examples
//!
//! ```bash
//! # Create a new prediction
//! nagual predict create "Deployment will succeed" --probability 0.85
//!
//! # Resolve a prediction
//! nagual predict resolve <prediction-id> true
//!
//! # List predictions
//! nagual predict list --status pending --limit 20
//!
//! # View calibration report
//! nagual predict calibration
//! ```

use std::path::PathBuf;
use std::sync::Arc;

use clap::{Args, Subcommand};
use serde::Serialize;

use crate::db::SqliteDb;
use crate::error::{NagualError, Result};
use crate::prediction::{
    CalibrationBucket, Prediction, PredictionFilter, PredictionId, PredictionStatus,
    PredictionStorage,
};

/// Prediction management commands.
///
/// Provides tools for creating, resolving, listing predictions,
/// and viewing calibration statistics.
#[derive(Args, Debug)]
pub struct PredictCommand {
    #[command(subcommand)]
    pub subcommand: PredictSubcommand,
}

/// Prediction subcommands.
#[derive(Subcommand, Debug)]
pub enum PredictSubcommand {
    /// Create a new prediction.
    ///
    /// Creates a prediction with specified probability and optional
    /// parameters like confidence, timeline, and domain.
    Create(CreateArgs),

    /// Resolve a prediction with actual outcome.
    ///
    /// Records the actual outcome of a prediction and calculates
    /// the Brier score for calibration tracking.
    Resolve(ResolveArgs),

    /// List predictions with optional filters.
    ///
    /// Lists predictions filtered by status, domain, probability range,
    /// and other criteria.
    List(ListArgs),

    /// View calibration report.
    ///
    /// Shows calibration buckets, Brier scores, and reliability
    /// information for prediction accuracy analysis.
    Calibration(CalibrationArgs),

    /// Show prediction statistics.
    ///
    /// Displays overall statistics about stored predictions.
    Stats(StatsArgs),

    /// Cancel a pending prediction.
    ///
    /// Marks a prediction as cancelled without resolving it.
    Cancel(CancelArgs),

    /// Delete a prediction.
    ///
    /// Permanently removes a prediction from storage.
    Delete(DeleteArgs),
}

/// Arguments for the create subcommand.
#[derive(Args, Debug)]
pub struct CreateArgs {
    /// Description of what is being predicted.
    #[arg(value_name = "DESCRIPTION")]
    pub description: String,

    /// Predicted probability (0.0 to 1.0).
    #[arg(short, long, default_value = "0.5")]
    pub probability: f64,

    /// Confidence in the probability estimate (0.0 to 1.0).
    #[arg(short = 'C', long, default_value = "0.5")]
    pub confidence: f64,

    /// Minimum expected days until resolution.
    #[arg(long, default_value = "1")]
    pub timeline_min: u32,

    /// Maximum expected days until resolution.
    #[arg(long, default_value = "30")]
    pub timeline_max: u32,

    /// Domain for categorization (e.g., "devops.deployment").
    #[arg(short, long, default_value = "general")]
    pub domain: String,

    /// Additional context about the prediction.
    #[arg(long)]
    pub context: Option<String>,

    /// Tags for categorization (comma-separated).
    #[arg(short, long)]
    pub tags: Option<String>,

    /// Session ID.
    #[arg(long)]
    pub session_id: Option<String>,

    /// Agent ID.
    #[arg(long)]
    pub agent_id: Option<String>,

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

/// Arguments for the resolve subcommand.
#[derive(Args, Debug)]
pub struct ResolveArgs {
    /// Prediction ID to resolve.
    #[arg(value_name = "ID")]
    pub id: String,

    /// Actual outcome (true, false, 1, 0, yes, no).
    #[arg(value_name = "OUTCOME")]
    pub outcome: String,

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

impl ResolveArgs {
    /// Parse the outcome string to a boolean.
    pub fn get_outcome(&self) -> bool {
        matches!(
            self.outcome.to_lowercase().as_str(),
            "true" | "1" | "yes" | "y"
        )
    }
}

/// Arguments for the list subcommand.
#[derive(Args, Debug)]
pub struct ListArgs {
    /// Filter by status (pending, resolved, expired, cancelled).
    #[arg(short, long)]
    pub status: Option<String>,

    /// Filter by domain.
    #[arg(short, long)]
    pub domain: Option<String>,

    /// Filter by minimum probability.
    #[arg(long)]
    pub min_prob: Option<f64>,

    /// Filter by maximum probability.
    #[arg(long)]
    pub max_prob: Option<f64>,

    /// Limit number of results.
    #[arg(short, long, default_value = "20")]
    pub limit: usize,

    /// Offset for pagination.
    #[arg(long, default_value = "0")]
    pub offset: usize,

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

/// Arguments for the calibration subcommand.
#[derive(Args, Debug)]
pub struct CalibrationArgs {
    /// Filter by domain.
    #[arg(short, long)]
    pub domain: Option<String>,

    /// Show detailed bucket information.
    #[arg(long)]
    pub detailed: bool,

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

/// Arguments for the stats subcommand.
#[derive(Args, Debug)]
pub struct StatsArgs {
    /// Path to SQLite database.
    #[arg(long, default_value = "./nagual.db")]
    pub db_path: PathBuf,

    /// Output results as JSON.
    #[arg(long)]
    pub json: bool,
}

/// Arguments for the cancel subcommand.
#[derive(Args, Debug)]
pub struct CancelArgs {
    /// Prediction ID to cancel.
    #[arg(value_name = "ID")]
    pub id: String,

    /// Path to SQLite database.
    #[arg(long, default_value = "./nagual.db")]
    pub db_path: PathBuf,

    /// Output results as JSON.
    #[arg(long)]
    pub json: bool,
}

/// Arguments for the delete subcommand.
#[derive(Args, Debug)]
pub struct DeleteArgs {
    /// Prediction ID to delete.
    #[arg(value_name = "ID")]
    pub id: String,

    /// Skip confirmation prompt.
    #[arg(long)]
    pub force: bool,

    /// Path to SQLite database.
    #[arg(long, default_value = "./nagual.db")]
    pub db_path: PathBuf,

    /// Output results as JSON.
    #[arg(long)]
    pub json: bool,
}

/// Calibration report for JSON output.
#[derive(Debug, Serialize)]
pub struct CalibrationReport {
    /// Domain being reported on
    pub domain: String,
    /// Overall Brier score (lower is better)
    pub overall_brier_score: Option<f64>,
    /// Total resolved predictions
    pub resolved_count: usize,
    /// Total pending predictions
    pub pending_count: usize,
    /// Calibration buckets
    pub buckets: Vec<BucketSummary>,
    /// Expected Calibration Error (ECE)
    pub expected_calibration_error: f64,
    /// Maximum Calibration Error (MCE)
    pub max_calibration_error: f64,
}

/// Summary of a calibration bucket.
#[derive(Debug, Serialize)]
pub struct BucketSummary {
    /// Range label (e.g., "0.0-0.1")
    pub range: String,
    /// Expected probability (midpoint)
    pub expected: f64,
    /// Actual positive rate
    pub actual: f64,
    /// Number of predictions in bucket
    pub count: u32,
    /// Calibration error for this bucket
    pub calibration_error: f64,
    /// Average Brier score for bucket
    pub avg_brier: f64,
}

impl PredictCommand {
    /// Execute the predict command.
    pub async fn run(&self) -> Result<()> {
        match &self.subcommand {
            PredictSubcommand::Create(args) => run_create(args).await,
            PredictSubcommand::Resolve(args) => run_resolve(args).await,
            PredictSubcommand::List(args) => run_list(args).await,
            PredictSubcommand::Calibration(args) => run_calibration(args).await,
            PredictSubcommand::Stats(args) => run_stats(args).await,
            PredictSubcommand::Cancel(args) => run_cancel(args).await,
            PredictSubcommand::Delete(args) => run_delete(args).await,
        }
    }
}

/// Run the create command.
async fn run_create(args: &CreateArgs) -> Result<()> {
    tracing::info!("Creating prediction");

    // Open database
    let db = open_db(&args.db_path)?;
    let storage = PredictionStorage::with_domain(db, &args.domain)
        .await
        .map_err(|e| NagualError::internal(e.to_string()))?;

    // Parse tags
    let tags: Vec<String> = args
        .tags
        .as_ref()
        .map(|t| t.split(',').map(|s| s.trim().to_string()).collect())
        .unwrap_or_default();

    // Build prediction
    let mut builder = Prediction::builder()
        .description(&args.description)
        .probability(args.probability)
        .confidence(args.confidence)
        .timeline(args.timeline_min, args.timeline_max)
        .domain(&args.domain)
        .tags(tags);

    if let Some(ref ctx) = args.context {
        builder = builder.context(ctx);
    }

    if let Some(ref sid) = args.session_id {
        builder = builder.session_id(sid);
    }

    if let Some(ref aid) = args.agent_id {
        builder = builder.agent_id(aid);
    }

    let prediction = builder
        .build()
        .map_err(|e| NagualError::internal(e.to_string()))?;

    // Store prediction
    let id = storage
        .store_prediction(&prediction)
        .await
        .map_err(|e| NagualError::internal(e.to_string()))?;

    // Output result
    if args.json {
        let output = serde_json::json!({
            "id": id.to_string(),
            "description": args.description,
            "probability": args.probability,
            "confidence": args.confidence,
            "domain": args.domain,
            "timeline": {
                "min_days": args.timeline_min,
                "max_days": args.timeline_max
            },
            "status": "pending"
        });
        println!("{}", serde_json::to_string_pretty(&output)?);
    } else {
        println!("\nPrediction Created");
        println!("{:-<50}", "");
        println!("  ID: {}", id);
        println!("  Description: {}", args.description);
        println!("  Probability: {:.1}%", args.probability * 100.0);
        println!("  Confidence: {:.1}%", args.confidence * 100.0);
        println!("  Timeline: {}-{} days", args.timeline_min, args.timeline_max);
        println!("  Domain: {}", args.domain);
        println!("  Status: pending");
    }

    Ok(())
}

/// Run the resolve command.
async fn run_resolve(args: &ResolveArgs) -> Result<()> {
    let outcome = args.get_outcome();
    tracing::info!(prediction_id = %args.id, outcome = %outcome, "Resolving prediction");

    // Open database
    let db = open_db(&args.db_path)?;
    let storage = PredictionStorage::new(db)
        .await
        .map_err(|e| NagualError::internal(e.to_string()))?;

    // Resolve prediction
    let prediction_id = PredictionId::from_string(&args.id);
    let prediction = storage
        .resolve_prediction(&prediction_id, outcome)
        .await
        .map_err(|e| NagualError::internal(e.to_string()))?;

    let brier_score = prediction.brier_score().unwrap_or(0.0);

    // Output result
    if args.json {
        let output = serde_json::json!({
            "id": args.id,
            "outcome": outcome,
            "probability": prediction.probability(),
            "brier_score": brier_score,
            "status": "resolved",
            "is_correct": prediction.is_correct()
        });
        println!("{}", serde_json::to_string_pretty(&output)?);
    } else {
        println!("\nPrediction Resolved");
        println!("{:-<50}", "");
        println!("  ID: {}", args.id);
        println!("  Predicted: {:.1}%", prediction.probability() * 100.0);
        println!("  Outcome: {}", if outcome { "TRUE" } else { "FALSE" });
        println!("  Brier Score: {:.4}", brier_score);
        println!(
            "  Correct: {}",
            prediction.is_correct().map_or("N/A".to_string(), |c| if c {
                "YES"
            } else {
                "NO"
            }
            .to_string())
        );

        // Quality assessment
        let quality = if brier_score < 0.1 {
            "Excellent"
        } else if brier_score < 0.25 {
            "Good"
        } else if brier_score < 0.5 {
            "Fair"
        } else {
            "Poor"
        };
        println!("  Quality: {}", quality);
    }

    Ok(())
}

/// Run the list command.
async fn run_list(args: &ListArgs) -> Result<()> {
    tracing::info!("Listing predictions");

    // Open database
    let db = open_db(&args.db_path)?;
    let storage = PredictionStorage::new(db)
        .await
        .map_err(|e| NagualError::internal(e.to_string()))?;

    // Build filter
    let mut filter = PredictionFilter::new()
        .with_limit(args.limit)
        .with_offset(args.offset);

    if let Some(ref status_str) = args.status {
        if let Ok(status) = status_str.parse::<PredictionStatus>() {
            filter = filter.with_status(status);
        }
    }

    if let Some(ref domain) = args.domain {
        filter = filter.with_domain(domain);
    }

    if let (Some(min), Some(max)) = (args.min_prob, args.max_prob) {
        filter = filter.with_probability_range(min, max);
    }

    // Get predictions
    let predictions = storage
        .list_predictions(&filter)
        .await
        .map_err(|e| NagualError::internal(e.to_string()))?;

    // Output results
    if args.json {
        let output: Vec<serde_json::Value> = predictions
            .iter()
            .map(|p| {
                serde_json::json!({
                    "id": p.id().to_string(),
                    "description": p.description(),
                    "probability": p.probability(),
                    "confidence": p.confidence(),
                    "status": p.status().to_string(),
                    "domain": p.domain(),
                    "brier_score": p.brier_score(),
                    "actual_outcome": p.actual_outcome(),
                    "created_at": p.created_at().to_rfc3339()
                })
            })
            .collect();
        println!("{}", serde_json::to_string_pretty(&output)?);
    } else {
        println!("\nPredictions ({} found)", predictions.len());
        println!("{:-<80}", "");

        if predictions.is_empty() {
            println!("  No predictions found.");
        } else {
            for p in &predictions {
                let status_icon = match p.status() {
                    PredictionStatus::Pending => "[P]",
                    PredictionStatus::Resolved => "[R]",
                    PredictionStatus::Expired => "[E]",
                    PredictionStatus::Cancelled => "[X]",
                };

                let brier_str = p
                    .brier_score()
                    .map(|b| format!(" BS:{:.3}", b))
                    .unwrap_or_default();

                // Truncate description if too long
                let desc = if p.description().len() > 40 {
                    format!("{}...", &p.description()[..37])
                } else {
                    p.description().to_string()
                };

                println!(
                    "  {} {} | {:.0}% | {}{}",
                    status_icon,
                    &p.id().to_string()[..8],
                    p.probability() * 100.0,
                    desc,
                    brier_str
                );

                if args.verbose {
                    println!(
                        "       Domain: {} | Confidence: {:.0}% | Created: {}",
                        p.domain(),
                        p.confidence() * 100.0,
                        p.created_at().format("%Y-%m-%d %H:%M")
                    );
                }
            }
        }
    }

    Ok(())
}

/// Run the calibration command.
async fn run_calibration(args: &CalibrationArgs) -> Result<()> {
    tracing::info!("Generating calibration report");

    // Open database
    let db = open_db(&args.db_path)?;
    let domain = args.domain.as_deref().unwrap_or("general");
    let storage = PredictionStorage::with_domain(db, domain)
        .await
        .map_err(|e| NagualError::internal(e.to_string()))?;

    // Get stats and buckets
    let stats = storage
        .get_stats()
        .await
        .map_err(|e| NagualError::internal(e.to_string()))?;

    let buckets = storage
        .get_calibration_buckets()
        .await
        .map_err(|e| NagualError::internal(e.to_string()))?;

    // Calculate ECE and MCE
    let (ece, mce) = calculate_calibration_errors(&buckets);

    // Build report
    let report = CalibrationReport {
        domain: domain.to_string(),
        overall_brier_score: stats.avg_brier_score,
        resolved_count: stats.resolved_count,
        pending_count: stats.pending_count,
        buckets: buckets
            .iter()
            .map(|b| BucketSummary {
                range: format!("{:.1}-{:.1}", b.lower_bound, b.upper_bound),
                expected: b.expected_probability(),
                actual: b.actual_rate(),
                count: b.prediction_count,
                calibration_error: b.calibration_error(),
                avg_brier: b.avg_brier_score(),
            })
            .collect(),
        expected_calibration_error: ece,
        max_calibration_error: mce,
    };

    // Output results
    if args.json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        display_calibration_report(&report, args.detailed);
    }

    Ok(())
}

/// Run the stats command.
async fn run_stats(args: &StatsArgs) -> Result<()> {
    tracing::info!("Getting prediction statistics");

    // Open database
    let db = open_db(&args.db_path)?;
    let storage = PredictionStorage::new(db)
        .await
        .map_err(|e| NagualError::internal(e.to_string()))?;

    let stats = storage
        .get_stats()
        .await
        .map_err(|e| NagualError::internal(e.to_string()))?;

    if args.json {
        println!("{}", serde_json::to_string_pretty(&stats)?);
    } else {
        println!("\nPrediction Statistics");
        println!("{:-<50}", "");
        println!("  Total predictions: {}", stats.total_predictions);
        println!("  Pending: {}", stats.pending_count);
        println!("  Resolved: {}", stats.resolved_count);
        println!("  Expired: {}", stats.expired_count);
        println!("  Cancelled: {}", stats.cancelled_count);
        if let Some(brier) = stats.avg_brier_score {
            println!("  Avg Brier Score: {:.4}", brier);
        }
        println!("  Calibration Buckets: {}", stats.bucket_count);
    }

    Ok(())
}

/// Run the cancel command.
async fn run_cancel(args: &CancelArgs) -> Result<()> {
    tracing::info!(prediction_id = %args.id, "Cancelling prediction");

    // Open database
    let db = open_db(&args.db_path)?;
    let storage = PredictionStorage::new(db)
        .await
        .map_err(|e| NagualError::internal(e.to_string()))?;

    let prediction_id = PredictionId::from_string(&args.id);
    storage
        .cancel_prediction(&prediction_id)
        .await
        .map_err(|e| NagualError::internal(e.to_string()))?;

    if args.json {
        let output = serde_json::json!({
            "id": args.id,
            "status": "cancelled"
        });
        println!("{}", serde_json::to_string_pretty(&output)?);
    } else {
        println!("Prediction {} cancelled.", args.id);
    }

    Ok(())
}

/// Run the delete command.
async fn run_delete(args: &DeleteArgs) -> Result<()> {
    tracing::info!(prediction_id = %args.id, "Deleting prediction");

    if !args.force {
        println!("Warning: This will permanently delete prediction {}.", args.id);
        println!("Use --force to confirm deletion.");
        return Ok(());
    }

    // Open database
    let db = open_db(&args.db_path)?;
    let storage = PredictionStorage::new(db)
        .await
        .map_err(|e| NagualError::internal(e.to_string()))?;

    let prediction_id = PredictionId::from_string(&args.id);
    storage
        .delete_prediction(&prediction_id)
        .await
        .map_err(|e| NagualError::internal(e.to_string()))?;

    if args.json {
        let output = serde_json::json!({
            "id": args.id,
            "deleted": true
        });
        println!("{}", serde_json::to_string_pretty(&output)?);
    } else {
        println!("Prediction {} deleted.", args.id);
    }

    Ok(())
}

/// Open database connection.
fn open_db(path: &PathBuf) -> Result<Arc<SqliteDb>> {
    let db = SqliteDb::open(path).map_err(|e| NagualError::internal(e.to_string()))?;
    Ok(Arc::new(db))
}

/// Calculate Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).
fn calculate_calibration_errors(buckets: &[CalibrationBucket]) -> (f64, f64) {
    let total_count: u32 = buckets.iter().map(|b| b.prediction_count).sum();

    if total_count == 0 {
        return (0.0, 0.0);
    }

    let mut ece: f64 = 0.0;
    let mut mce: f64 = 0.0;

    for bucket in buckets {
        if bucket.prediction_count > 0 {
            let weight = bucket.prediction_count as f64 / total_count as f64;
            let error = bucket.calibration_error();
            ece += weight * error;
            if error > mce {
                mce = error;
            }
        }
    }

    (ece, mce)
}

/// Display the calibration report.
fn display_calibration_report(report: &CalibrationReport, detailed: bool) {
    println!("\nCalibration Report: {}", report.domain);
    println!("{:-<60}", "");

    // Overview
    println!("\nOverview:");
    println!("  Resolved predictions: {}", report.resolved_count);
    println!("  Pending predictions: {}", report.pending_count);
    if let Some(brier) = report.overall_brier_score {
        println!("  Overall Brier Score: {:.4}", brier);

        // Interpret Brier score
        let quality = if brier < 0.1 {
            "Excellent - highly calibrated"
        } else if brier < 0.2 {
            "Good - well calibrated"
        } else if brier < 0.25 {
            "Fair - reasonably calibrated"
        } else {
            "Needs improvement"
        };
        println!("  Quality: {}", quality);
    }

    // Calibration metrics
    println!("\nCalibration Metrics:");
    println!(
        "  Expected Calibration Error (ECE): {:.4}",
        report.expected_calibration_error
    );
    println!(
        "  Maximum Calibration Error (MCE): {:.4}",
        report.max_calibration_error
    );

    // Reliability diagram (ASCII)
    println!("\nReliability Diagram:");
    println!("  Expected | Actual | Count | Error");
    println!("  {:-<45}", "");

    for bucket in &report.buckets {
        let bar_len = ((bucket.actual * 20.0).round() as usize).min(20);
        let expected_bar_len = ((bucket.expected * 20.0).round() as usize).min(20);
        let bar = "#".repeat(bar_len);
        let _expected_marker = if bucket.count > 0 && expected_bar_len < 20 {
            format!("{:>width$}|", "", width = expected_bar_len)
        } else {
            String::new()
        };

        if bucket.count > 0 {
            println!(
                "  {:>5.1}%   | {:>5.1}% | {:>5} | {:.4}",
                bucket.expected * 100.0,
                bucket.actual * 100.0,
                bucket.count,
                bucket.calibration_error
            );
            if detailed {
                println!("            [{:<20}] avg BS: {:.4}", bar, bucket.avg_brier);
            }
        } else {
            println!(
                "  {:>5.1}%   | {:>5}  | {:>5} | -",
                bucket.expected * 100.0,
                "-",
                0
            );
        }
    }

    // Interpretation
    println!("\nInterpretation:");
    if report.expected_calibration_error < 0.05 {
        println!("  Predictions are very well calibrated.");
    } else if report.expected_calibration_error < 0.1 {
        println!("  Predictions are reasonably well calibrated.");
    } else if report.expected_calibration_error < 0.2 {
        println!("  Some calibration adjustment may be beneficial.");
    } else {
        println!("  Predictions show significant calibration issues.");
        println!("  Consider using calibrated probabilities.");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    // Helper struct for testing CLI parsing
    #[derive(Parser, Debug)]
    struct TestCli {
        #[command(subcommand)]
        command: TestCommand,
    }

    #[derive(Subcommand, Debug)]
    enum TestCommand {
        Predict(PredictCommand),
    }

    #[test]
    fn test_cli_parse_create() {
        let args = vec![
            "test",
            "predict",
            "create",
            "Deployment will succeed",
            "-p",
            "0.85",
        ];
        let cli = TestCli::try_parse_from(args);
        assert!(cli.is_ok());
    }

    #[test]
    fn test_cli_parse_create_with_options() {
        let args = vec![
            "test",
            "predict",
            "create",
            "Test prediction",
            "-p",
            "0.75",
            "-C",
            "0.9",
            "-d",
            "devops.deployment",
            "--timeline-min",
            "7",
            "--timeline-max",
            "14",
            "--tags",
            "critical,deployment",
        ];
        let cli = TestCli::try_parse_from(args);
        assert!(cli.is_ok());
    }

    #[test]
    fn test_cli_parse_resolve() {
        let args = vec!["test", "predict", "resolve", "pred-123", "true"];
        let cli = TestCli::try_parse_from(args);
        assert!(cli.is_ok());
    }

    #[test]
    fn test_cli_parse_list() {
        let args = vec![
            "test",
            "predict",
            "list",
            "--status",
            "pending",
            "--limit",
            "10",
        ];
        let cli = TestCli::try_parse_from(args);
        assert!(cli.is_ok());
    }

    #[test]
    fn test_cli_parse_calibration() {
        let args = vec!["test", "predict", "calibration", "--detailed"];
        let cli = TestCli::try_parse_from(args);
        assert!(cli.is_ok());
    }

    #[test]
    fn test_cli_parse_stats() {
        let args = vec!["test", "predict", "stats", "--json"];
        let cli = TestCli::try_parse_from(args);
        assert!(cli.is_ok());
    }

    #[test]
    fn test_cli_parse_cancel() {
        let args = vec!["test", "predict", "cancel", "pred-456"];
        let cli = TestCli::try_parse_from(args);
        assert!(cli.is_ok());
    }

    #[test]
    fn test_cli_parse_delete() {
        let args = vec!["test", "predict", "delete", "pred-789", "--force"];
        let cli = TestCli::try_parse_from(args);
        assert!(cli.is_ok());
    }

    #[test]
    fn test_calculate_calibration_errors_empty() {
        let buckets: Vec<CalibrationBucket> = vec![];
        let (ece, mce) = calculate_calibration_errors(&buckets);
        assert!((ece - 0.0).abs() < f64::EPSILON);
        assert!((mce - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_calculate_calibration_errors() {
        use chrono::Utc;

        let buckets = vec![
            CalibrationBucket {
                id: "test_0".to_string(),
                lower_bound: 0.0,
                upper_bound: 0.1,
                prediction_count: 10,
                actual_positive_count: 1, // 10% actual, 5% expected -> 5% error
                total_brier_score: 0.5,
                domain: "test".to_string(),
                updated_at: Utc::now(),
            },
            CalibrationBucket {
                id: "test_5".to_string(),
                lower_bound: 0.5,
                upper_bound: 0.6,
                prediction_count: 10,
                actual_positive_count: 5, // 50% actual, 55% expected -> 5% error
                total_brier_score: 2.5,
                domain: "test".to_string(),
                updated_at: Utc::now(),
            },
        ];

        let (ece, mce) = calculate_calibration_errors(&buckets);

        // ECE should be weighted average of errors
        assert!(ece > 0.0);
        // MCE should be max of individual errors
        assert!(mce > 0.0);
    }
}
