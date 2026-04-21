//! Conflict resolution command implementation
//!
//! Lists, inspects, and resolves conflicts between local SQLite
//! and cloud PostgreSQL databases.

use clap::{Args, Subcommand};
use std::path::PathBuf;

use crate::db::{ConflictLog, ConflictLogEntry, ConflictResolution};
use crate::error::Result;

/// Default path for conflict log database.
const DEFAULT_CONFLICT_LOG_PATH: &str = "nagual_conflicts.db";

/// Conflict management command
///
/// Manages sync conflicts between local SQLite and cloud PostgreSQL databases.
/// Supports listing, inspecting, and resolving conflicts with various strategies.
#[derive(Args, Debug)]
pub struct ConflictsCommand {
    #[command(subcommand)]
    pub action: ConflictAction,

    /// Path to conflict log database
    #[arg(long, global = true, default_value = DEFAULT_CONFLICT_LOG_PATH)]
    pub db: PathBuf,
}

/// Available conflict management actions
#[derive(Subcommand, Debug)]
pub enum ConflictAction {
    /// List all pending conflicts
    List {
        /// Filter by table name
        #[arg(long)]
        table: Option<String>,

        /// Show only conflicts newer than this timestamp (RFC3339)
        #[arg(long)]
        since: Option<String>,

        /// Maximum number of conflicts to show
        #[arg(short, long, default_value = "50")]
        limit: usize,

        /// Show all conflicts (not just pending)
        #[arg(long)]
        all: bool,
    },

    /// Show detailed information about a specific conflict
    Show {
        /// Conflict ID to inspect
        conflict_id: String,

        /// Show full diff between versions
        #[arg(long)]
        diff: bool,
    },

    /// Resolve a conflict
    Resolve {
        /// Conflict ID to resolve
        conflict_id: String,

        /// Resolution strategy
        #[arg(long, value_parser = ["local", "remote", "merge", "manual", "skip"])]
        strategy: String,

        /// For manual resolution, path to JSON file with resolved data
        #[arg(long)]
        data_file: Option<String>,
    },

    /// Automatically resolve all conflicts using a strategy
    AutoResolve {
        /// Resolution strategy to apply
        #[arg(long, value_parser = ["local-wins", "remote-wins", "newest-wins"])]
        strategy: String,

        /// Dry run - show what would be resolved without applying
        #[arg(long)]
        dry_run: bool,

        /// Filter to specific table
        #[arg(long)]
        table: Option<String>,

        /// Maximum conflicts to process
        #[arg(short, long, default_value = "100")]
        limit: usize,
    },

    /// Export conflicts to a file for external review
    Export {
        /// Output file path
        #[arg(short, long)]
        output: String,

        /// Export format
        #[arg(long, value_parser = ["json", "csv"], default_value = "json")]
        format: String,

        /// Maximum conflicts to export
        #[arg(short, long, default_value = "1000")]
        limit: usize,
    },

    /// Show conflict statistics
    Stats,

    /// Clean up old resolved conflicts
    Cleanup {
        /// Remove conflicts resolved more than N days ago
        #[arg(long, default_value = "30")]
        older_than_days: u32,

        /// Dry run - show what would be deleted
        #[arg(long)]
        dry_run: bool,
    },
}

/// Represents a sync conflict between databases (legacy - use ConflictLogEntry instead)
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct Conflict {
    pub id: String,
    pub table_name: String,
    pub record_id: String,
    pub local_version: Option<serde_json::Value>,
    pub remote_version: Option<serde_json::Value>,
    pub local_timestamp: chrono::DateTime<chrono::Utc>,
    pub remote_timestamp: chrono::DateTime<chrono::Utc>,
    pub conflict_type: ConflictType,
}

/// Type of conflict detected
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConflictType {
    /// Both sides modified the same record
    UpdateUpdate,
    /// Local update, remote delete
    UpdateDelete,
    /// Local delete, remote update
    DeleteUpdate,
    /// Schema mismatch between versions
    SchemaMismatch,
}

impl std::fmt::Display for ConflictType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConflictType::UpdateUpdate => write!(f, "update-update"),
            ConflictType::UpdateDelete => write!(f, "update-delete"),
            ConflictType::DeleteUpdate => write!(f, "delete-update"),
            ConflictType::SchemaMismatch => write!(f, "schema-mismatch"),
        }
    }
}

impl ConflictsCommand {
    /// Execute the conflicts command
    pub async fn run(&self) -> Result<()> {
        tracing::info!(db_path = ?self.db, "Running conflicts command");

        // Open or create conflict log
        let conflict_log = ConflictLog::new(&self.db)?;

        match &self.action {
            ConflictAction::List { table, since, limit, all } => {
                self.list_conflicts(&conflict_log, table.as_deref(), since.as_deref(), *limit, *all).await
            }
            ConflictAction::Show { conflict_id, diff } => {
                self.show_conflict(&conflict_log, conflict_id, *diff).await
            }
            ConflictAction::Resolve { conflict_id, strategy, data_file } => {
                self.resolve_conflict(&conflict_log, conflict_id, strategy, data_file.as_deref()).await
            }
            ConflictAction::AutoResolve { strategy, dry_run, table, limit } => {
                self.auto_resolve(&conflict_log, strategy, *dry_run, table.as_deref(), *limit).await
            }
            ConflictAction::Export { output, format, limit } => {
                self.export_conflicts(&conflict_log, output, format, *limit).await
            }
            ConflictAction::Stats => {
                self.show_stats(&conflict_log).await
            }
            ConflictAction::Cleanup { older_than_days, dry_run } => {
                self.cleanup(&conflict_log, *older_than_days, *dry_run).await
            }
        }
    }

    /// List conflicts
    async fn list_conflicts(
        &self,
        conflict_log: &ConflictLog,
        table: Option<&str>,
        _since: Option<&str>,
        limit: usize,
        all: bool,
    ) -> Result<()> {
        tracing::debug!(
            table = ?table,
            limit = limit,
            all = all,
            "Listing conflicts"
        );

        let conflicts = if let Some(table_name) = table {
            conflict_log.get_by_table(table_name, limit)?
        } else if all {
            // Get all conflicts (pending + resolved)
            let pending = conflict_log.get_pending(limit)?;
            let resolved = conflict_log.get_by_resolution(ConflictResolution::LocalWins, limit)?;
            let remote = conflict_log.get_by_resolution(ConflictResolution::RemoteWins, limit)?;
            let mut all_conflicts = pending;
            all_conflicts.extend(resolved);
            all_conflicts.extend(remote);
            all_conflicts.truncate(limit);
            all_conflicts
        } else {
            conflict_log.get_pending(limit)?
        };

        if conflicts.is_empty() {
            println!("No conflicts found.");
            return Ok(());
        }

        println!("Conflicts (showing {} of up to {}):", conflicts.len(), limit);
        println!("{:-<80}", "");
        println!(
            "{:<36} {:<15} {:<15} {:<12}",
            "ID", "Table", "Record ID", "Resolution"
        );
        println!("{:-<80}", "");

        for conflict in &conflicts {
            println!(
                "{:<36} {:<15} {:<15} {:<12}",
                &conflict.id[..36.min(conflict.id.len())],
                &conflict.table_name[..15.min(conflict.table_name.len())],
                &conflict.record_id[..15.min(conflict.record_id.len())],
                conflict.resolution
            );
        }

        println!("{:-<80}", "");
        println!("Total: {} conflict(s)", conflicts.len());

        Ok(())
    }

    /// Show detailed conflict information
    async fn show_conflict(&self, conflict_log: &ConflictLog, conflict_id: &str, show_diff: bool) -> Result<()> {
        tracing::debug!(conflict_id = conflict_id, "Showing conflict details");

        match conflict_log.get(conflict_id)? {
            Some(conflict) => {
                println!("Conflict Details");
                println!("{:=<60}", "");
                println!("ID:          {}", conflict.id);
                println!("Table:       {}", conflict.table_name);
                println!("Record ID:   {}", conflict.record_id);
                println!("Resolution:  {}", conflict.resolution);
                println!("Created:     {}", conflict.created_at);
                if let Some(resolved_at) = conflict.resolved_at {
                    println!("Resolved:    {}", resolved_at);
                }
                println!();

                println!("Local Data:");
                println!("{:-<60}", "");
                println!("{}", serde_json::to_string_pretty(&conflict.local_data)?);
                println!();

                println!("Remote Data:");
                println!("{:-<60}", "");
                println!("{}", serde_json::to_string_pretty(&conflict.remote_data)?);

                if show_diff {
                    println!();
                    println!("Diff Analysis:");
                    println!("{:-<60}", "");
                    self.show_diff(&conflict);
                }

                // Show LWW recommendation
                if conflict.is_pending() {
                    println!();
                    println!("LWW Recommendation:");
                    println!("{:-<60}", "");
                    match conflict.lww_winner() {
                        Some(winner) => println!("Based on timestamps: {}", winner),
                        None => println!("Cannot determine winner (missing timestamps)"),
                    }
                }

                Ok(())
            }
            None => {
                println!("Conflict not found: {}", conflict_id);
                Ok(())
            }
        }
    }

    /// Show diff between local and remote data
    fn show_diff(&self, conflict: &ConflictLogEntry) {
        if let (Some(local_obj), Some(remote_obj)) = (
            conflict.local_data.as_object(),
            conflict.remote_data.as_object(),
        ) {
            // Collect all keys
            let mut all_keys: std::collections::HashSet<&String> =
                local_obj.keys().collect();
            all_keys.extend(remote_obj.keys());

            let mut keys: Vec<_> = all_keys.into_iter().collect();
            keys.sort();

            for key in keys {
                let local_val = local_obj.get(key);
                let remote_val = remote_obj.get(key);

                match (local_val, remote_val) {
                    (Some(l), Some(r)) if l != r => {
                        println!("  {}: {} -> {} (CHANGED)", key, l, r);
                    }
                    (Some(l), None) => {
                        println!("  {}: {} (LOCAL ONLY)", key, l);
                    }
                    (None, Some(r)) => {
                        println!("  {}: {} (REMOTE ONLY)", key, r);
                    }
                    _ => {
                        // Same value or both None
                    }
                }
            }
        } else {
            println!("  (Cannot diff non-object values)");
        }
    }

    /// Resolve a specific conflict
    async fn resolve_conflict(
        &self,
        conflict_log: &ConflictLog,
        conflict_id: &str,
        strategy: &str,
        _data_file: Option<&str>,
    ) -> Result<()> {
        tracing::info!(
            conflict_id = conflict_id,
            strategy = strategy,
            "Resolving conflict"
        );

        let resolution = match strategy {
            "local" => ConflictResolution::LocalWins,
            "remote" => ConflictResolution::RemoteWins,
            "merge" => ConflictResolution::Merged,
            "manual" => ConflictResolution::Manual,
            "skip" => ConflictResolution::Skipped,
            _ => {
                println!("Unknown strategy: {}", strategy);
                return Ok(());
            }
        };

        conflict_log.resolve(conflict_id, resolution)?;
        println!("Conflict {} resolved as: {}", conflict_id, resolution);

        Ok(())
    }

    /// Auto-resolve all conflicts with a strategy
    async fn auto_resolve(
        &self,
        conflict_log: &ConflictLog,
        strategy: &str,
        dry_run: bool,
        _table: Option<&str>,
        limit: usize,
    ) -> Result<()> {
        tracing::info!(
            strategy = strategy,
            dry_run = dry_run,
            limit = limit,
            "Auto-resolving conflicts"
        );

        if strategy != "newest-wins" {
            println!("Only 'newest-wins' (LWW) strategy is currently supported for auto-resolve");
            return Ok(());
        }

        if dry_run {
            // Preview what would be resolved
            let pending = conflict_log.get_pending(limit)?;
            println!("Dry run - would resolve {} conflicts:", pending.len());
            println!("{:-<80}", "");

            for conflict in &pending {
                let recommendation = match conflict.lww_winner() {
                    Some(winner) => format!("{}", winner),
                    None => "SKIP (no timestamps)".to_string(),
                };
                println!(
                    "  {} ({}.{}) -> {}",
                    &conflict.id[..8],
                    conflict.table_name,
                    conflict.record_id,
                    recommendation
                );
            }
        } else {
            let result = conflict_log.auto_resolve_lww(limit)?;
            println!("Auto-resolve complete:");
            println!("  Resolved: {}", result.resolved);
            println!("  Skipped:  {}", result.skipped);
            println!("  Failed:   {}", result.failed);
        }

        Ok(())
    }

    /// Export conflicts to file
    async fn export_conflicts(&self, conflict_log: &ConflictLog, output: &str, format: &str, limit: usize) -> Result<()> {
        tracing::info!(output = output, format = format, limit = limit, "Exporting conflicts");

        match format {
            "json" => {
                let json = conflict_log.export_to_json(limit)?;
                std::fs::write(output, json)?;
                println!("Exported conflicts to {} (JSON format)", output);
            }
            "csv" => {
                // Simple CSV export
                let conflicts = conflict_log.get_pending(limit)?;
                let mut csv = String::from("id,table_name,record_id,resolution,created_at\n");
                for c in conflicts {
                    csv.push_str(&format!(
                        "{},{},{},{},{}\n",
                        c.id, c.table_name, c.record_id, c.resolution, c.created_at
                    ));
                }
                std::fs::write(output, csv)?;
                println!("Exported conflicts to {} (CSV format)", output);
            }
            _ => {
                println!("Unknown format: {}", format);
            }
        }

        Ok(())
    }

    /// Show conflict statistics
    async fn show_stats(&self, conflict_log: &ConflictLog) -> Result<()> {
        let stats = conflict_log.stats()?;

        println!("Conflict Statistics");
        println!("{:=<40}", "");
        println!("Total conflicts:     {}", stats.total);
        println!("Pending:             {}", stats.pending);
        println!("Resolved:            {}", stats.resolved);
        println!("  - Local wins:      {}", stats.local_wins);
        println!("  - Remote wins:     {}", stats.remote_wins);

        if let Some(oldest) = stats.oldest_pending {
            println!("Oldest pending:      {}", oldest);
        }

        Ok(())
    }

    /// Clean up old resolved conflicts
    async fn cleanup(&self, conflict_log: &ConflictLog, older_than_days: u32, dry_run: bool) -> Result<()> {
        tracing::info!(older_than_days = older_than_days, dry_run = dry_run, "Cleaning up conflicts");

        if dry_run {
            // For dry run, we'd need to query how many would be deleted
            // This is a simplified version
            println!(
                "Dry run: would delete resolved conflicts older than {} days",
                older_than_days
            );
        } else {
            let deleted = conflict_log.cleanup_resolved(older_than_days)?;
            println!("Cleaned up {} old resolved conflict(s)", deleted);
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conflict_type_display() {
        assert_eq!(format!("{}", ConflictType::UpdateUpdate), "update-update");
        assert_eq!(format!("{}", ConflictType::SchemaMismatch), "schema-mismatch");
    }
}
