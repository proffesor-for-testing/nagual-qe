//! Sync command implementation
//!
//! Provides CLI commands for sync operations including status,
//! backup, restore, and drill management.
//!
//! Usage:
//! - `nagual sync status` - Show sync status and health
//! - `nagual sync backup [--full|--incremental]` - Create backup
//! - `nagual sync restore <path>` - Restore from backup
//! - `nagual sync drill` - Run restore drill
//! - `nagual sync scheduler [start|stop|status]` - Manage scheduler

use std::path::PathBuf;

use clap::{Args, Subcommand};

use crate::error::Result;
use crate::sync::{
    BackupConfig, BackupManager, BackupType, DrillReport, DrillResult, RecoveryPlan,
    RestoreConfig, RestoreDrill, RestoreDrillConfig, RestoreManager, RestoreResult,
    SyncHealth, SyncScheduler, SyncSchedulerConfig, SyncStatusReport,
};

#[cfg(feature = "brain-sync")]
use crate::reasoning_bank::pattern::PatternId;

/// Sync command for backup, restore, and sync operations
#[derive(Args, Debug)]
pub struct SyncCommand {
    #[command(subcommand)]
    pub action: SyncAction,
}

/// Sync subcommands
#[derive(Subcommand, Debug)]
pub enum SyncAction {
    /// Show sync status and health
    Status(StatusArgs),

    /// Create a backup
    Backup(BackupArgs),

    /// Restore from a backup
    Restore(RestoreArgs),

    /// Run a restore drill
    Drill(DrillArgs),

    /// List available backups
    List(ListArgs),

    /// Point-in-time recovery
    Pitr(PitrArgs),

    /// Brain collective knowledge operations
    ///
    /// Share local patterns with the collective brain, search shared knowledge,
    /// and check brain system status. Requires the `brain-sync` feature.
    #[cfg(feature = "brain-sync")]
    Brain(BrainArgs),
}

/// Arguments for brain subcommand
#[cfg(feature = "brain-sync")]
#[derive(Args, Debug)]
pub struct BrainArgs {
    #[command(subcommand)]
    pub action: BrainAction,
}

/// Brain subcommands
#[cfg(feature = "brain-sync")]
#[derive(Subcommand, Debug)]
pub enum BrainAction {
    /// Share a local pattern with the collective brain
    ///
    /// Loads a pattern from the local SQLite database by ID and shares
    /// it with the collective brain API. PII is automatically stripped
    /// before transmission.
    Share(BrainShareArgs),

    /// Search the collective brain for relevant knowledge
    ///
    /// Queries the brain.ruv.io API for matching memories. Results are
    /// displayed similarly to `nagual knowledge search`.
    Search(BrainSearchArgs),

    /// Show brain system status
    ///
    /// Retrieves and displays the current status from the brain API,
    /// including connectivity and available memory counts.
    Status,
}

/// Arguments for brain share command
#[cfg(feature = "brain-sync")]
#[derive(Args, Debug)]
pub struct BrainShareArgs {
    /// Pattern ID to share
    pub pattern_id: String,

    /// Database path
    #[arg(long, default_value = "nagual.db")]
    pub db_path: PathBuf,
}

/// Arguments for brain search command
#[cfg(feature = "brain-sync")]
#[derive(Args, Debug)]
pub struct BrainSearchArgs {
    /// Search query
    pub query: String,

    /// Filter by category
    #[arg(long)]
    pub category: Option<String>,

    /// Maximum results
    #[arg(long, default_value = "10")]
    pub limit: usize,
}

/// Arguments for status command
#[derive(Args, Debug)]
pub struct StatusArgs {
    /// Show detailed sync history
    #[arg(long)]
    pub history: bool,

    /// Number of history entries to show
    #[arg(long, default_value = "10")]
    pub limit: usize,

    /// Output as JSON
    #[arg(long)]
    pub json: bool,
}

/// Arguments for backup command
#[derive(Args, Debug)]
pub struct BackupArgs {
    /// Create a full backup
    #[arg(long, conflicts_with = "incremental")]
    pub full: bool,

    /// Create an incremental backup
    #[arg(long, conflicts_with = "full")]
    pub incremental: bool,

    /// Path to the source database
    #[arg(long, default_value = "./nagual.db")]
    pub source: PathBuf,

    /// Directory to store backups
    #[arg(long, default_value = "./backups")]
    pub backup_dir: PathBuf,

    /// Compression level (0-9)
    #[arg(long, default_value = "6")]
    pub compression: u32,

    /// Output as JSON
    #[arg(long)]
    pub json: bool,
}

/// Arguments for restore command
#[derive(Args, Debug)]
pub struct RestoreArgs {
    /// Path or URL to backup file
    pub backup_path: String,

    /// Target database path
    #[arg(long, default_value = "./nagual.db")]
    pub target: PathBuf,

    /// Skip creating pre-restore backup
    #[arg(long)]
    pub no_backup: bool,

    /// Skip integrity verification
    #[arg(long)]
    pub no_verify: bool,

    /// Force restore even if target exists
    #[arg(long)]
    pub force: bool,

    /// Output as JSON
    #[arg(long)]
    pub json: bool,
}

/// Arguments for drill command
#[derive(Args, Debug)]
pub struct DrillArgs {
    /// Run the drill now
    #[arg(long)]
    pub run: bool,

    /// Show next scheduled drill date
    #[arg(long)]
    pub next: bool,

    /// List previous drill reports
    #[arg(long)]
    pub list: bool,

    /// Number of reports to show
    #[arg(long, default_value = "10")]
    pub limit: usize,

    /// Production database path
    #[arg(long, default_value = "./nagual.db")]
    pub db_path: PathBuf,

    /// Backup directory
    #[arg(long, default_value = "./backups")]
    pub backup_dir: PathBuf,

    /// Output as JSON
    #[arg(long)]
    pub json: bool,
}

/// Arguments for list command
#[derive(Args, Debug)]
pub struct ListArgs {
    /// Filter by backup type (full, incremental)
    #[arg(long, value_name = "TYPE")]
    pub backup_type: Option<String>,

    /// Maximum number of backups to show
    #[arg(long, default_value = "20")]
    pub limit: usize,

    /// Path to backup directory
    #[arg(long, default_value = "./backups")]
    pub backup_dir: PathBuf,

    /// Database path (for metadata)
    #[arg(long, default_value = "./nagual.db")]
    pub db_path: PathBuf,

    /// Output as JSON
    #[arg(long)]
    pub json: bool,
}

/// Arguments for point-in-time recovery
#[derive(Args, Debug)]
pub struct PitrArgs {
    /// Target timestamp (ISO 8601 format)
    pub timestamp: String,

    /// Target database path
    #[arg(long, default_value = "./nagual.db")]
    pub target: PathBuf,

    /// Backup directory
    #[arg(long, default_value = "./backups")]
    pub backup_dir: PathBuf,

    /// Show recovery plan without executing
    #[arg(long)]
    pub dry_run: bool,

    /// Output as JSON
    #[arg(long)]
    pub json: bool,
}

impl SyncCommand {
    /// Execute the sync command
    pub async fn run(&self) -> Result<()> {
        match &self.action {
            SyncAction::Status(args) => self.run_status(args).await,
            SyncAction::Backup(args) => self.run_backup(args).await,
            SyncAction::Restore(args) => self.run_restore(args).await,
            SyncAction::Drill(args) => self.run_drill(args).await,
            SyncAction::List(args) => self.run_list(args).await,
            SyncAction::Pitr(args) => self.run_pitr(args).await,
            #[cfg(feature = "brain-sync")]
            SyncAction::Brain(args) => self.run_brain(args).await,
        }
    }

    /// Run the status command
    async fn run_status(&self, args: &StatusArgs) -> Result<()> {
        // Create a scheduler to get status (or load from persistent storage)
        let scheduler = SyncScheduler::new(SyncSchedulerConfig::default())?;
        let report = scheduler.status_report().await;

        if args.json {
            println!("{}", serde_json::to_string_pretty(&report)?);
        } else {
            self.display_status_report(&report, args.history, args.limit);
        }

        Ok(())
    }

    /// Run the backup command
    async fn run_backup(&self, args: &BackupArgs) -> Result<()> {
        let config = BackupConfig::new(&args.source, &args.backup_dir)
            .with_compression_level(args.compression);

        let mut manager = BackupManager::new(config)?;

        let metadata = if args.incremental {
            manager.create_incremental_backup().await?
        } else {
            // Default to full backup
            manager.create_full_backup().await?
        };

        if args.json {
            println!("{}", serde_json::to_string_pretty(&metadata)?);
        } else {
            println!("\nBackup Created Successfully");
            println!("{:-<50}", "");
            println!("  ID: {}", metadata.id);
            println!("  Type: {}", metadata.backup_type);
            println!("  Path: {}", metadata.path);
            println!("  Size: {} bytes (compressed)", metadata.compressed_size);
            println!("  Compression: {:.1}%", metadata.compression_ratio * 100.0);
            println!("  Records: ~{}", metadata.record_count);
            println!("  Created: {}", metadata.created_at);
            println!("{:-<50}", "");
        }

        Ok(())
    }

    /// Run the restore command
    async fn run_restore(&self, args: &RestoreArgs) -> Result<()> {
        // Check if target exists and force not specified
        if args.target.exists() && !args.force {
            if !args.json {
                eprintln!(
                    "Target database exists: {}. Use --force to overwrite.",
                    args.target.display()
                );
            }
            return Ok(());
        }

        let config = RestoreConfig::new(&args.target, args.target.parent().unwrap_or(&PathBuf::from(".")))
            .with_backup_before_restore(!args.no_backup);

        let manager = RestoreManager::with_config(config)?;
        let result = manager.restore_from_backup(&args.backup_path).await?;

        if args.json {
            println!("{}", serde_json::to_string_pretty(&result)?);
        } else {
            self.display_restore_result(&result);
        }

        Ok(())
    }

    /// Run the drill command
    async fn run_drill(&self, args: &DrillArgs) -> Result<()> {
        let config = RestoreDrillConfig::new(&args.db_path, &args.backup_dir);
        let drill = RestoreDrill::new(config)?;

        if args.next {
            let next_date = drill.next_drill_date();
            if args.json {
                println!(r#"{{"next_drill_date": "{}"}}"#, next_date.to_rfc3339());
            } else {
                println!("Next scheduled drill: {}", next_date);
            }
            return Ok(());
        }

        if args.list {
            let reports = drill.list_reports()?;
            let limited: Vec<_> = reports.into_iter().take(args.limit).collect();

            if args.json {
                println!("{}", serde_json::to_string_pretty(&limited)?);
            } else {
                self.display_drill_reports(&limited);
            }
            return Ok(());
        }

        if args.run {
            println!("Running restore drill...\n");
            let report = drill.run_drill().await?;

            if args.json {
                println!("{}", serde_json::to_string_pretty(&report)?);
            } else {
                self.display_drill_report(&report);
            }
            return Ok(());
        }

        // Default: show status
        let next_date = drill.next_drill_date();
        let is_time = drill.is_drill_time();

        if args.json {
            println!(
                r#"{{"next_drill_date": "{}", "is_drill_time": {}}}"#,
                next_date.to_rfc3339(),
                is_time
            );
        } else {
            println!("Restore Drill Status");
            println!("{:-<50}", "");
            println!("  Next scheduled: {}", next_date);
            println!("  Is drill time: {}", if is_time { "Yes" } else { "No" });
            println!("{:-<50}", "");
            println!("\nCommands:");
            println!("  nagual sync drill --run     Run drill now");
            println!("  nagual sync drill --list    Show previous reports");
            println!("  nagual sync drill --next    Show next drill date");
        }

        Ok(())
    }

    /// Run the list command
    async fn run_list(&self, args: &ListArgs) -> Result<()> {
        let config = BackupConfig::new(&args.db_path, &args.backup_dir);
        let manager = BackupManager::new(config)?;

        let mut backups = manager.list_backups()?;

        // Filter by type if specified
        if let Some(ref type_filter) = args.backup_type {
            let backup_type = match type_filter.to_lowercase().as_str() {
                "full" => Some(BackupType::Full),
                "incremental" | "incr" => Some(BackupType::Incremental),
                _ => None,
            };

            if let Some(bt) = backup_type {
                backups.retain(|b| b.backup_type == bt);
            }
        }

        // Limit results
        backups.truncate(args.limit);

        if args.json {
            println!("{}", serde_json::to_string_pretty(&backups)?);
        } else {
            self.display_backup_list(&backups);
        }

        Ok(())
    }

    /// Run the PITR command
    async fn run_pitr(&self, args: &PitrArgs) -> Result<()> {
        // Parse timestamp
        let target_timestamp = chrono::DateTime::parse_from_rfc3339(&args.timestamp)
            .map(|dt| dt.with_timezone(&chrono::Utc))
            .map_err(|e| crate::error::NagualError::config(format!("Invalid timestamp: {}", e)))?;

        let config = RestoreConfig::new(&args.target, &args.backup_dir);
        let manager = RestoreManager::with_config(config)?;

        // Build recovery plan
        let plan = manager.build_recovery_plan(target_timestamp)?;

        if args.dry_run {
            if args.json {
                println!("{}", serde_json::to_string_pretty(&plan)?);
            } else {
                self.display_recovery_plan(&plan);
            }
            return Ok(());
        }

        // Execute recovery
        println!("Executing point-in-time recovery to {}...\n", target_timestamp);
        let result = manager.point_in_time_recovery(target_timestamp).await?;

        if args.json {
            println!("{}", serde_json::to_string_pretty(&result)?);
        } else {
            self.display_restore_result(&result);
        }

        Ok(())
    }

    /// Run the brain command
    #[cfg(feature = "brain-sync")]
    async fn run_brain(&self, args: &BrainArgs) -> Result<()> {
        match &args.action {
            BrainAction::Share(share_args) => self.run_brain_share(share_args).await,
            BrainAction::Search(search_args) => self.run_brain_search(search_args).await,
            BrainAction::Status => self.run_brain_status().await,
        }
    }

    /// Share a local pattern with the collective brain
    #[cfg(feature = "brain-sync")]
    async fn run_brain_share(&self, args: &BrainShareArgs) -> Result<()> {
        use crate::cli::common::init_storage_sqlite_only;
        use crate::sync::brain::BrainClient;

        // Load pattern from local SQLite
        let storage = init_storage_sqlite_only(&args.db_path).await?;
        let pattern_id = PatternId::from_string(&args.pattern_id);

        let pattern = storage.get_pattern(&pattern_id).await?;
        let pattern = match pattern {
            Some(p) => p,
            None => {
                eprintln!("Pattern not found: {}", args.pattern_id);
                return Ok(());
            }
        };

        // Extract fields for the brain API
        let category = format!("{}", pattern.category());
        let title = pattern.problem().chars().take(120).collect::<String>();
        let content = format!(
            "Problem: {}\n\nSolution: {}",
            pattern.problem(),
            pattern.solution()
        );
        let tags: Vec<String> = pattern.tags().to_vec();

        // Share with the collective brain
        let client = BrainClient::new();
        println!("Sharing pattern {} with collective brain...", args.pattern_id);
        println!("  Endpoint: {}", client.base_url());
        println!("  Auth: {}", if client.has_api_key() { "API key configured" } else { "No API key" });

        match client.share(&category, &title, &content, tags).await {
            Ok(memory_id) => {
                println!("\nShared successfully!");
                println!("  Memory ID: {}", memory_id);
                println!("  Category:  {}", category);
                println!("  Title:     {}", title);
            }
            Err(e) => {
                eprintln!("\nFailed to share pattern: {}", e);
                eprintln!("  Check BRAIN_URL and BRAIN_API_KEY environment variables.");
            }
        }

        Ok(())
    }

    /// Search the collective brain
    #[cfg(feature = "brain-sync")]
    async fn run_brain_search(&self, args: &BrainSearchArgs) -> Result<()> {
        use crate::sync::brain::BrainClient;

        let client = BrainClient::new();
        println!("Searching collective brain for: \"{}\"", args.query);
        if let Some(ref cat) = args.category {
            println!("  Category filter: {}", cat);
        }
        println!("  Limit: {}", args.limit);
        println!();

        match client
            .search(&args.query, args.category.as_deref(), args.limit)
            .await
        {
            Ok(memories) => {
                if memories.is_empty() {
                    println!("No results found.");
                    return Ok(());
                }

                println!("Found {} result(s):\n", memories.len());
                println!("{:-<70}", "");

                for (i, memory) in memories.iter().enumerate() {
                    println!(
                        "  {}. [{}] {} (quality: {:.2})",
                        i + 1,
                        memory.category,
                        memory.title,
                        memory.quality_score.mean()
                    );
                    // Show first 200 chars of content
                    let preview: String = memory.content.chars().take(200).collect();
                    println!("     {}", preview);
                    if !memory.tags.is_empty() {
                        println!("     Tags: {}", memory.tags.join(", "));
                    }
                    println!("     ID: {} | Created: {}", memory.id, memory.created_at);
                    println!("{:-<70}", "");
                }
            }
            Err(e) => {
                eprintln!("Search failed: {}", e);
                eprintln!("  Check BRAIN_URL and BRAIN_API_KEY environment variables.");
            }
        }

        Ok(())
    }

    /// Show brain system status
    #[cfg(feature = "brain-sync")]
    async fn run_brain_status(&self) -> Result<()> {
        use crate::sync::brain::BrainClient;

        let client = BrainClient::new();
        println!("Brain System Status");
        println!("{:-<50}", "");
        println!("  Endpoint: {}", client.base_url());
        println!("  Auth:     {}", if client.has_api_key() { "API key configured" } else { "No API key" });
        println!();

        match client.status().await {
            Ok(status) => {
                println!("{}", serde_json::to_string_pretty(&status)?);
            }
            Err(e) => {
                eprintln!("Failed to get status: {}", e);
                eprintln!("  Check BRAIN_URL and BRAIN_API_KEY environment variables.");
            }
        }

        Ok(())
    }

    // Display helpers

    fn display_status_report(&self, report: &SyncStatusReport, show_history: bool, limit: usize) {
        println!("\nSync Status Report");
        println!("{:-<50}", "");

        let health_icon = match report.current.sync_health {
            SyncHealth::Healthy => "[OK]",
            SyncHealth::Degraded => "[WARN]",
            SyncHealth::Unhealthy => "[FAIL]",
        };

        println!("{} Health: {}", health_icon, report.current.sync_health);
        println!("  Pending records: {}", report.current.pending_records);
        println!("  Consecutive failures: {}", report.current.consecutive_failures);

        if let Some(ref error) = report.current.last_error {
            println!("  Last error: {}", error);
        }

        println!("\nLast Sync Times:");
        for (task, time) in &report.current.last_sync_times {
            println!("  {}: {}", task, time);
        }

        if show_history && !report.history.is_empty() {
            println!("\nRecent History:");
            println!("{:-<50}", "");

            for entry in report.history.iter().take(limit) {
                let status = if entry.success { "[OK]" } else { "[FAIL]" };
                println!(
                    "  {} {} - {} ({}ms, {} records)",
                    status, entry.task, entry.executed_at, entry.duration_ms, entry.records_synced
                );
                if let Some(ref error) = entry.error {
                    println!("       Error: {}", error);
                }
            }
        }

        println!("{:-<50}", "");
        println!("Generated: {}", report.generated_at);
    }

    fn display_restore_result(&self, result: &RestoreResult) {
        let status = if result.success { "[OK]" } else { "[FAIL]" };

        println!("\nRestore Result");
        println!("{:-<50}", "");
        println!("{} Status: {}", status, if result.success { "Success" } else { "Failed" });
        println!("  Backup ID: {}", result.backup_id);
        println!("  Backup Type: {}", result.backup_type);
        println!("  Restored Path: {}", result.restored_path);
        println!("  Records: ~{}", result.record_count);
        println!("  Duration: {}ms", result.restore_duration_ms);

        if let Some(ref pre_backup) = result.pre_restore_backup_id {
            println!("  Pre-restore backup: {}", pre_backup);
        }

        if !result.warnings.is_empty() {
            println!("\nWarnings:");
            for warning in &result.warnings {
                println!("  - {}", warning);
            }
        }

        println!("{:-<50}", "");
    }

    fn display_drill_report(&self, report: &DrillReport) {
        let result_icon = match report.result {
            DrillResult::Success => "[OK]",
            DrillResult::Warning => "[WARN]",
            DrillResult::Failed => "[FAIL]",
        };

        println!("\nRestore Drill Report");
        println!("{:-<50}", "");
        println!("{} Result: {}", result_icon, report.result);
        println!("  Drill ID: {}", report.drill_id);
        println!("  Duration: {}ms", report.duration_ms);
        println!("  Backup used: {}", report.backup_used.id);
        println!("  Integrity check: {}", if report.integrity_check_passed { "Passed" } else { "Failed" });
        println!("  Production records: {}", report.production_record_count);
        println!("  Restored records: {}", report.restored_record_count);
        println!("  Difference: {}", report.record_count_difference);

        if !report.issues.is_empty() {
            println!("\nIssues:");
            for issue in &report.issues {
                println!("  [{:?}] {}: {}", issue.severity, issue.category, issue.message);
            }
        }

        if !report.warnings.is_empty() {
            println!("\nWarnings:");
            for warning in &report.warnings {
                println!("  - {}", warning);
            }
        }

        println!("{:-<50}", "");
    }

    fn display_drill_reports(&self, reports: &[DrillReport]) {
        println!("\nPrevious Drill Reports");
        println!("{:-<50}", "");

        if reports.is_empty() {
            println!("  No drill reports found.");
        } else {
            for report in reports {
                let icon = match report.result {
                    DrillResult::Success => "[OK]",
                    DrillResult::Warning => "[WARN]",
                    DrillResult::Failed => "[FAIL]",
                };
                println!(
                    "  {} {} - {} ({}ms)",
                    icon, report.drill_id, report.started_at, report.duration_ms
                );
            }
        }

        println!("{:-<50}", "");
    }

    fn display_backup_list(&self, backups: &[crate::sync::BackupMetadata]) {
        println!("\nAvailable Backups");
        println!("{:-<50}", "");

        if backups.is_empty() {
            println!("  No backups found.");
        } else {
            for backup in backups {
                let type_str = match backup.backup_type {
                    BackupType::Full => "FULL",
                    BackupType::Incremental => "INCR",
                };
                println!(
                    "  [{}] {} - {} ({} bytes)",
                    type_str, backup.id, backup.created_at, backup.compressed_size
                );
            }
        }

        println!("{:-<50}", "");
        println!("Total: {} backups", backups.len());
    }

    fn display_recovery_plan(&self, plan: &RecoveryPlan) {
        println!("\nRecovery Plan (Dry Run)");
        println!("{:-<50}", "");
        println!("  Target timestamp: {}", plan.target_timestamp);
        println!("  Base backup: {} ({})", plan.base_backup.id, plan.base_backup.backup_type);
        println!("  Incremental backups: {}", plan.incrementals.len());
        println!("  Total data size: {} bytes", plan.total_size);
        println!("  Estimated duration: {}ms", plan.estimated_duration_ms);

        if !plan.incrementals.is_empty() {
            println!("\n  Incremental steps:");
            for (i, incr) in plan.incrementals.iter().enumerate() {
                println!("    {}. {} ({})", i + 1, incr.id, incr.created_at);
            }
        }

        println!("{:-<50}", "");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_status_args() {
        let args = StatusArgs {
            history: true,
            limit: 5,
            json: false,
        };
        assert!(args.history);
        assert_eq!(args.limit, 5);
    }

    #[test]
    fn test_backup_args() {
        let args = BackupArgs {
            full: true,
            incremental: false,
            source: PathBuf::from("./test.db"),
            backup_dir: PathBuf::from("./backups"),
            compression: 6,
            json: false,
        };
        assert!(args.full);
        assert!(!args.incremental);
    }

    #[test]
    fn test_restore_args() {
        let args = RestoreArgs {
            backup_path: "backup.gz".to_string(),
            target: PathBuf::from("./restored.db"),
            no_backup: false,
            no_verify: false,
            force: false,
            json: false,
        };
        assert_eq!(args.backup_path, "backup.gz");
        assert!(!args.force);
    }

    #[test]
    fn test_drill_args() {
        let args = DrillArgs {
            run: true,
            next: false,
            list: false,
            limit: 10,
            db_path: PathBuf::from("./nagual.db"),
            backup_dir: PathBuf::from("./backups"),
            json: false,
        };
        assert!(args.run);
        assert!(!args.next);
    }

    #[test]
    fn test_pitr_args() {
        let args = PitrArgs {
            timestamp: "2024-01-15T12:00:00Z".to_string(),
            target: PathBuf::from("./restored.db"),
            backup_dir: PathBuf::from("./backups"),
            dry_run: true,
            json: false,
        };
        assert!(args.dry_run);
    }

    #[cfg(feature = "brain-sync")]
    #[test]
    fn test_brain_share_args() {
        let args = BrainShareArgs {
            pattern_id: "test-pattern-123".to_string(),
            db_path: PathBuf::from("./nagual.db"),
        };
        assert_eq!(args.pattern_id, "test-pattern-123");
    }

    #[cfg(feature = "brain-sync")]
    #[test]
    fn test_brain_search_args() {
        let args = BrainSearchArgs {
            query: "error handling".to_string(),
            category: Some("rust".to_string()),
            limit: 5,
        };
        assert_eq!(args.query, "error handling");
        assert_eq!(args.category, Some("rust".to_string()));
        assert_eq!(args.limit, 5);
    }

    #[cfg(feature = "brain-sync")]
    #[test]
    fn test_brain_search_args_defaults() {
        let args = BrainSearchArgs {
            query: "test".to_string(),
            category: None,
            limit: 10,
        };
        assert!(args.category.is_none());
        assert_eq!(args.limit, 10);
    }
}
