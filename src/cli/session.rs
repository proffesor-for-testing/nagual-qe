//! Session management CLI commands.
//!
//! Provides commands for managing development sessions and viewing
//! session analytics including token efficiency metrics.
//!
//! # Usage Examples
//!
//! ```bash
//! # Start a new session
//! nagual session start --domain rust
//!
//! # End the current session
//! nagual session end
//!
//! # Show session statistics
//! nagual session stats
//!
//! # List recent sessions
//! nagual session list --limit 10
//!
//! # Record token usage for current session
//! nagual session tokens 1500
//!
//! # Show current/active session
//! nagual session current
//! ```

use std::path::PathBuf;
use std::sync::Arc;

use clap::{Args, Subcommand};
use serde::Serialize;

use crate::db::{Session, SessionManager, SqliteDb};
use crate::error::Result;

/// Session management commands.
///
/// Provides tools for managing development sessions and viewing
/// session analytics for token efficiency tracking.
#[derive(Args, Debug)]
pub struct SessionCommand {
    #[command(subcommand)]
    pub subcommand: SessionSubcommand,
}

/// Session subcommands.
#[derive(Subcommand, Debug)]
pub enum SessionSubcommand {
    /// Start a new session.
    ///
    /// Creates a new session and optionally associates it with a domain.
    Start(StartArgs),

    /// End the current session.
    ///
    /// Closes the active session by setting its end timestamp.
    End(EndArgs),

    /// Show session statistics.
    ///
    /// Displays aggregated metrics across all sessions including
    /// token efficiency (patterns learned per 1K tokens).
    Stats(StatsArgs),

    /// List recent sessions.
    ///
    /// Shows recent sessions with their metrics.
    List(ListArgs),

    /// Record token usage for a session.
    ///
    /// Adds tokens to the current or specified session.
    Tokens(TokensArgs),

    /// Show the current/active session.
    ///
    /// Displays details of the currently active session, if any.
    Current(CurrentArgs),

    /// Delete a session.
    ///
    /// Removes a session from the database.
    Delete(DeleteArgs),

    /// Cleanup old sessions.
    ///
    /// Removes sessions older than the specified number of days.
    Cleanup(CleanupArgs),
}

/// Arguments for the start subcommand.
#[derive(Args, Debug)]
pub struct StartArgs {
    /// Domain focus for this session (e.g., "rust", "database").
    #[arg(long)]
    pub domain: Option<String>,

    /// Path to SQLite database.
    #[arg(long, default_value = "./nagual.db")]
    pub db_path: PathBuf,

    /// Output results as JSON.
    #[arg(long)]
    pub json: bool,
}

/// Arguments for the end subcommand.
#[derive(Args, Debug)]
pub struct EndArgs {
    /// Session ID to end (defaults to current active session).
    #[arg(long)]
    pub session_id: Option<String>,

    /// Path to SQLite database.
    #[arg(long, default_value = "./nagual.db")]
    pub db_path: PathBuf,

    /// Output results as JSON.
    #[arg(long)]
    pub json: bool,

    /// Warn if session produced zero learning artifacts.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    pub check_learning: bool,
}

/// Arguments for the stats subcommand.
#[derive(Args, Debug)]
pub struct StatsArgs {
    /// Time window in days (e.g., 7, 30, 90). Omit for all-time stats.
    #[arg(long)]
    pub window: Option<u32>,

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

/// Arguments for the list subcommand.
#[derive(Args, Debug)]
pub struct ListArgs {
    /// Maximum number of sessions to show.
    #[arg(long, default_value = "10")]
    pub limit: usize,

    /// Filter by domain.
    #[arg(long)]
    pub domain: Option<String>,

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

/// Arguments for the tokens subcommand.
#[derive(Args, Debug)]
pub struct TokensArgs {
    /// Number of tokens to record.
    pub count: u64,

    /// Session ID (defaults to current active session).
    #[arg(long)]
    pub session_id: Option<String>,

    /// Path to SQLite database.
    #[arg(long, default_value = "./nagual.db")]
    pub db_path: PathBuf,

    /// Output results as JSON.
    #[arg(long)]
    pub json: bool,
}

/// Arguments for the current subcommand.
#[derive(Args, Debug)]
pub struct CurrentArgs {
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
    /// Session ID to delete.
    pub session_id: String,

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

/// Arguments for the cleanup subcommand.
#[derive(Args, Debug)]
pub struct CleanupArgs {
    /// Delete sessions older than this many days.
    #[arg(long, default_value = "90")]
    pub older_than: u32,

    /// Dry run - show what would be deleted without deleting.
    #[arg(long)]
    pub dry_run: bool,

    /// Path to SQLite database.
    #[arg(long, default_value = "./nagual.db")]
    pub db_path: PathBuf,

    /// Output results as JSON.
    #[arg(long)]
    pub json: bool,
}

impl SessionCommand {
    /// Execute the session command.
    pub async fn run(&self) -> Result<()> {
        match &self.subcommand {
            SessionSubcommand::Start(args) => run_start(args).await,
            SessionSubcommand::End(args) => run_end(args).await,
            SessionSubcommand::Stats(args) => run_stats(args).await,
            SessionSubcommand::List(args) => run_list(args).await,
            SessionSubcommand::Tokens(args) => run_tokens(args).await,
            SessionSubcommand::Current(args) => run_current(args).await,
            SessionSubcommand::Delete(args) => run_delete(args).await,
            SessionSubcommand::Cleanup(args) => run_cleanup(args).await,
        }
    }
}

/// Initialize the session manager from a database path.
async fn init_manager(db_path: &PathBuf) -> Result<SessionManager> {
    // Ensure parent directory exists
    if let Some(parent) = db_path.parent() {
        if !parent.exists() {
            std::fs::create_dir_all(parent)?;
        }
    }

    let db = Arc::new(SqliteDb::open(db_path)?);

    // Ensure sessions table exists
    db.execute_batch(
        r#"CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            started_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            ended_at TEXT,
            tokens_used INTEGER DEFAULT 0,
            patterns_learned INTEGER DEFAULT 0,
            patterns_retrieved INTEGER DEFAULT 0,
            domain TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_sessions_started_at ON sessions(started_at);
        CREATE INDEX IF NOT EXISTS idx_sessions_domain ON sessions(domain);"#,
    ).await?;

    Ok(SessionManager::new(db))
}

/// Run the start command.
async fn run_start(args: &StartArgs) -> Result<()> {
    let manager = init_manager(&args.db_path).await?;

    // Check for existing active session
    if let Some(active) = manager.get_active_session().await? {
        let output = StartOutput {
            success: false,
            session: None,
            message: format!("Active session already exists: {}", active.id),
        };

        if args.json {
            println!("{}", serde_json::to_string_pretty(&output)?);
        } else {
            eprintln!("Error: {}", output.message);
            eprintln!("Use 'nagual session end' to close it first.");
        }
        return Ok(());
    }

    let session = manager.start_session(args.domain.as_deref()).await?;

    let output = StartOutput {
        success: true,
        session: Some(session.clone()),
        message: format!("Session started: {}", session.id),
    };

    if args.json {
        println!("{}", serde_json::to_string_pretty(&output)?);
    } else {
        println!("\nSession Started");
        println!("{:-<50}", "");
        println!("  ID: {}", session.id);
        println!("  Started: {}", session.started_at.format("%Y-%m-%d %H:%M:%S UTC"));
        if let Some(ref domain) = session.domain {
            println!("  Domain: {}", domain);
        }
        println!("{:-<50}\n", "");
    }

    Ok(())
}

#[derive(Serialize)]
struct StartOutput {
    success: bool,
    session: Option<Session>,
    message: String,
}

/// Run the end command.
async fn run_end(args: &EndArgs) -> Result<()> {
    let manager = init_manager(&args.db_path).await?;

    let session_id = if let Some(ref id) = args.session_id {
        id.clone()
    } else {
        // Find active session
        match manager.get_active_session().await? {
            Some(s) => s.id,
            None => {
                let output = EndOutput {
                    success: false,
                    session_id: None,
                    message: "No active session found".to_string(),
                };

                if args.json {
                    println!("{}", serde_json::to_string_pretty(&output)?);
                } else {
                    eprintln!("Error: No active session found.");
                }
                return Ok(());
            }
        }
    };

    // Get session details before ending (for validation)
    let _session = manager.get_session(&session_id).await?;

    manager.end_session(&session_id).await?;

    // Get updated session
    let ended_session = manager.get_session(&session_id).await?;

    let output = EndOutput {
        success: true,
        session_id: Some(session_id.clone()),
        message: format!("Session ended: {}", session_id),
    };

    if args.json {
        println!("{}", serde_json::to_string_pretty(&output)?);
    } else {
        println!("\nSession Ended");
        println!("{:-<50}", "");
        println!("  ID: {}", session_id);
        if let Some(ref s) = ended_session {
            if let Some(duration) = s.duration_secs() {
                println!("  Duration: {} seconds", duration);
            }
            println!("  Tokens used: {}", s.tokens_used);
            println!("  Patterns learned: {}", s.patterns_learned);
            println!("  Patterns retrieved: {}", s.patterns_retrieved);
            println!("  Efficiency: {:.2} patterns/1K tokens", s.efficiency());
        }
        println!("{:-<50}\n", "");
    }

    if args.check_learning {
        if let Some(ref s) = ended_session {
            if s.patterns_learned == 0 {
                eprintln!(
                    "WARNING: Session {} produced 0 learning artifacts.",
                    &session_id[..8.min(session_id.len())]
                );
                eprintln!(
                    "Consider recording insights with: nagual knowledge store <problem> --solution <solution>"
                );
            }
        }
    }

    Ok(())
}

#[derive(Serialize)]
struct EndOutput {
    success: bool,
    session_id: Option<String>,
    message: String,
}

/// Run the stats command.
async fn run_stats(args: &StatsArgs) -> Result<()> {
    let manager = init_manager(&args.db_path).await?;

    let stats = if let Some(days) = args.window {
        manager.get_stats_for_window(days).await?
    } else {
        manager.get_stats().await?
    };

    if args.json {
        println!("{}", serde_json::to_string_pretty(&stats)?);
    } else {
        println!("\nSession Statistics");
        if let Some(days) = args.window {
            println!("(Last {} days)", days);
        }
        println!("{:-<50}", "");
        println!("  Total sessions: {}", stats.total_sessions);
        println!("  Active sessions: {}", stats.active_sessions);
        println!();
        println!("  Token Usage:");
        println!("    Total tokens: {}", stats.total_tokens);
        println!("    Avg per session: {:.0}", stats.avg_tokens_per_session);
        println!();
        println!("  Pattern Activity:");
        println!("    Patterns learned: {}", stats.total_patterns_learned);
        println!("    Patterns retrieved: {}", stats.total_patterns_retrieved);
        println!("    Avg learned/session: {:.1}", stats.avg_patterns_per_session);
        println!();
        println!("  Efficiency:");
        println!("    Patterns per 1K tokens: {:.2}", stats.efficiency);
        if stats.avg_duration_secs > 0.0 {
            println!("    Avg session duration: {}", format_duration(stats.avg_duration_secs as u64));
        }
        println!("{:-<50}\n", "");
    }

    Ok(())
}

/// Run the list command.
async fn run_list(args: &ListArgs) -> Result<()> {
    let manager = init_manager(&args.db_path).await?;

    let sessions = if let Some(ref domain) = args.domain {
        manager.list_sessions_by_domain(domain, args.limit).await?
    } else {
        manager.list_sessions(args.limit).await?
    };

    if args.json {
        println!("{}", serde_json::to_string_pretty(&sessions)?);
    } else {
        println!("\nRecent Sessions ({} total)", sessions.len());
        if let Some(ref domain) = args.domain {
            println!("Domain: {}", domain);
        }
        println!("{:-<70}", "");

        if sessions.is_empty() {
            println!("  No sessions found.");
        } else {
            for (i, session) in sessions.iter().enumerate() {
                let status = if session.is_active() { "ACTIVE" } else { "ended" };
                let domain = session.domain.as_deref().unwrap_or("-");
                let short_id = &session.id[..8];

                println!(
                    "  {}. [{}] {} | {} | {} tokens | {} learned | {:.2} eff",
                    i + 1,
                    status,
                    short_id,
                    domain,
                    session.tokens_used,
                    session.patterns_learned,
                    session.efficiency()
                );

                if args.verbose {
                    println!(
                        "     Started: {} | Retrieved: {}",
                        session.started_at.format("%Y-%m-%d %H:%M"),
                        session.patterns_retrieved
                    );
                    if let Some(ref ended) = session.ended_at {
                        println!("     Ended: {}", ended.format("%Y-%m-%d %H:%M"));
                    }
                }
            }
        }

        println!("{:-<70}\n", "");
    }

    Ok(())
}

/// Run the tokens command.
async fn run_tokens(args: &TokensArgs) -> Result<()> {
    let manager = init_manager(&args.db_path).await?;

    let session_id = if let Some(ref id) = args.session_id {
        id.clone()
    } else {
        match manager.get_active_session().await? {
            Some(s) => s.id,
            None => {
                let output = TokensOutput {
                    success: false,
                    session_id: None,
                    tokens_added: 0,
                    total_tokens: 0,
                    message: "No active session found".to_string(),
                };

                if args.json {
                    println!("{}", serde_json::to_string_pretty(&output)?);
                } else {
                    eprintln!("Error: No active session found.");
                    eprintln!("Use 'nagual session start' to create one.");
                }
                return Ok(());
            }
        }
    };

    manager.record_tokens(&session_id, args.count).await?;

    let session = manager.get_session(&session_id).await?;
    let total = session.as_ref().map(|s| s.tokens_used).unwrap_or(0);

    let output = TokensOutput {
        success: true,
        session_id: Some(session_id.clone()),
        tokens_added: args.count,
        total_tokens: total,
        message: format!("Recorded {} tokens", args.count),
    };

    if args.json {
        println!("{}", serde_json::to_string_pretty(&output)?);
    } else {
        println!("\nTokens Recorded");
        println!("{:-<50}", "");
        println!("  Session: {}", &session_id[..8]);
        println!("  Tokens added: {}", args.count);
        println!("  Total tokens: {}", total);
        println!("{:-<50}\n", "");
    }

    Ok(())
}

#[derive(Serialize)]
struct TokensOutput {
    success: bool,
    session_id: Option<String>,
    tokens_added: u64,
    total_tokens: u64,
    message: String,
}

/// Run the current command.
async fn run_current(args: &CurrentArgs) -> Result<()> {
    let manager = init_manager(&args.db_path).await?;

    let session = manager.get_active_session().await?;

    if args.json {
        println!("{}", serde_json::to_string_pretty(&session)?);
    } else {
        match session {
            Some(s) => {
                let duration = chrono::Utc::now()
                    .signed_duration_since(s.started_at)
                    .num_seconds();

                println!("\nCurrent Session");
                println!("{:-<50}", "");
                println!("  ID: {}", s.id);
                println!("  Started: {}", s.started_at.format("%Y-%m-%d %H:%M:%S UTC"));
                println!("  Duration: {}", format_duration(duration as u64));
                if let Some(ref domain) = s.domain {
                    println!("  Domain: {}", domain);
                }
                println!();
                println!("  Metrics:");
                println!("    Tokens used: {}", s.tokens_used);
                println!("    Patterns learned: {}", s.patterns_learned);
                println!("    Patterns retrieved: {}", s.patterns_retrieved);
                println!("    Efficiency: {:.2} patterns/1K tokens", s.efficiency());
                println!("{:-<50}\n", "");
            }
            None => {
                println!("\nNo active session.");
                println!("Use 'nagual session start' to begin a new session.\n");
            }
        }
    }

    Ok(())
}

/// Run the delete command.
async fn run_delete(args: &DeleteArgs) -> Result<()> {
    let manager = init_manager(&args.db_path).await?;

    // Check if session exists
    let session = manager.get_session(&args.session_id).await?;
    if session.is_none() {
        let output = DeleteOutput {
            success: false,
            session_id: args.session_id.clone(),
            message: "Session not found".to_string(),
        };

        if args.json {
            println!("{}", serde_json::to_string_pretty(&output)?);
        } else {
            eprintln!("Error: Session not found: {}", args.session_id);
        }
        return Ok(());
    }

    // Confirm if not forced
    if !args.force && !args.json {
        println!("Delete session {}? [y/N] ", &args.session_id[..8.min(args.session_id.len())]);
        let mut input = String::new();
        std::io::stdin().read_line(&mut input)?;
        if !input.trim().eq_ignore_ascii_case("y") {
            println!("Cancelled.");
            return Ok(());
        }
    }

    let deleted = manager.delete_session(&args.session_id).await?;

    let output = DeleteOutput {
        success: deleted,
        session_id: args.session_id.clone(),
        message: if deleted {
            "Session deleted".to_string()
        } else {
            "Failed to delete session".to_string()
        },
    };

    if args.json {
        println!("{}", serde_json::to_string_pretty(&output)?);
    } else if deleted {
        println!("Session deleted: {}", args.session_id);
    }

    Ok(())
}

#[derive(Serialize)]
struct DeleteOutput {
    success: bool,
    session_id: String,
    message: String,
}

/// Run the cleanup command.
async fn run_cleanup(args: &CleanupArgs) -> Result<()> {
    let manager = init_manager(&args.db_path).await?;

    if args.dry_run {
        // Count how many would be deleted
        let stats_before = manager.get_stats().await?;
        let stats_window = manager.get_stats_for_window(args.older_than).await?;
        let would_delete = stats_before.total_sessions.saturating_sub(stats_window.total_sessions);

        let output = CleanupOutput {
            success: true,
            sessions_deleted: 0,
            dry_run: true,
            message: format!(
                "Would delete {} sessions older than {} days",
                would_delete, args.older_than
            ),
        };

        if args.json {
            println!("{}", serde_json::to_string_pretty(&output)?);
        } else {
            println!("\nDry Run: Would delete {} sessions older than {} days.\n",
                would_delete, args.older_than);
        }
    } else {
        let deleted = manager.cleanup_old_sessions(args.older_than).await?;

        let output = CleanupOutput {
            success: true,
            sessions_deleted: deleted,
            dry_run: false,
            message: format!("Deleted {} sessions older than {} days", deleted, args.older_than),
        };

        if args.json {
            println!("{}", serde_json::to_string_pretty(&output)?);
        } else {
            println!("\nCleanup Complete");
            println!("{:-<50}", "");
            println!("  Sessions deleted: {}", deleted);
            println!("  Older than: {} days", args.older_than);
            println!("{:-<50}\n", "");
        }
    }

    Ok(())
}

#[derive(Serialize)]
struct CleanupOutput {
    success: bool,
    sessions_deleted: usize,
    dry_run: bool,
    message: String,
}

/// Format duration in human-readable form.
fn format_duration(secs: u64) -> String {
    if secs < 60 {
        format!("{}s", secs)
    } else if secs < 3600 {
        format!("{}m {}s", secs / 60, secs % 60)
    } else if secs < 86400 {
        format!("{}h {}m", secs / 3600, (secs % 3600) / 60)
    } else {
        format!("{}d {}h", secs / 86400, (secs % 86400) / 3600)
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
        Session(SessionCommand),
    }

    #[test]
    fn test_cli_parse_session_start() {
        let args = vec!["test", "session", "start", "--domain", "rust"];
        let cli = TestCli::try_parse_from(args);
        assert!(cli.is_ok());
    }

    #[test]
    fn test_cli_parse_session_end() {
        let args = vec!["test", "session", "end"];
        let cli = TestCli::try_parse_from(args);
        assert!(cli.is_ok());
    }

    #[test]
    fn test_cli_parse_session_stats() {
        let args = vec!["test", "session", "stats", "--window", "7"];
        let cli = TestCli::try_parse_from(args);
        assert!(cli.is_ok());
    }

    #[test]
    fn test_cli_parse_session_list() {
        let args = vec!["test", "session", "list", "--limit", "20", "--domain", "rust"];
        let cli = TestCli::try_parse_from(args);
        assert!(cli.is_ok());
    }

    #[test]
    fn test_cli_parse_session_tokens() {
        let args = vec!["test", "session", "tokens", "1500"];
        let cli = TestCli::try_parse_from(args);
        assert!(cli.is_ok());
    }

    #[test]
    fn test_cli_parse_session_current() {
        let args = vec!["test", "session", "current"];
        let cli = TestCli::try_parse_from(args);
        assert!(cli.is_ok());
    }

    #[test]
    fn test_cli_parse_session_delete() {
        let args = vec!["test", "session", "delete", "abc-123", "--force"];
        let cli = TestCli::try_parse_from(args);
        assert!(cli.is_ok());
    }

    #[test]
    fn test_cli_parse_session_cleanup() {
        let args = vec!["test", "session", "cleanup", "--older-than", "30", "--dry-run"];
        let cli = TestCli::try_parse_from(args);
        assert!(cli.is_ok());
    }

    #[test]
    fn test_cli_parse_session_end_with_check_learning() {
        // Default: check_learning is true
        let args = vec!["test", "session", "end"];
        let cli = TestCli::try_parse_from(args).unwrap();
        match cli.command {
            TestCommand::Session(cmd) => match cmd.subcommand {
                SessionSubcommand::End(end_args) => {
                    assert!(end_args.check_learning);
                }
                _ => panic!("Expected End subcommand"),
            },
        }

        // Explicit false
        let args = vec!["test", "session", "end", "--check-learning", "false"];
        let cli = TestCli::try_parse_from(args).unwrap();
        match cli.command {
            TestCommand::Session(cmd) => match cmd.subcommand {
                SessionSubcommand::End(end_args) => {
                    assert!(!end_args.check_learning);
                }
                _ => panic!("Expected End subcommand"),
            },
        }

        // Explicit true
        let args = vec!["test", "session", "end", "--check-learning", "true"];
        let cli = TestCli::try_parse_from(args).unwrap();
        match cli.command {
            TestCommand::Session(cmd) => match cmd.subcommand {
                SessionSubcommand::End(end_args) => {
                    assert!(end_args.check_learning);
                }
                _ => panic!("Expected End subcommand"),
            },
        }
    }

    #[test]
    fn test_format_duration() {
        assert_eq!(format_duration(30), "30s");
        assert_eq!(format_duration(90), "1m 30s");
        assert_eq!(format_duration(3661), "1h 1m");
        assert_eq!(format_duration(90061), "1d 1h");
    }
}
