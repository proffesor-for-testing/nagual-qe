//! CLI commands for cloud sync (push/pull/status).

use std::path::PathBuf;

use clap::{Args, Subcommand};

use crate::cli::common::init_storage_sqlite_only;
use crate::cloud::client::CloudClient;
use crate::cloud::sync_state;
use crate::db::SqliteDb;
use crate::error::{NagualError, Result};

/// Cloud sync commands for pushing/pulling patterns to/from a remote nagual server.
#[derive(Args, Debug)]
pub struct CloudCommand {
    #[command(subcommand)]
    pub command: CloudSubcommand,
}

#[derive(Subcommand, Debug)]
pub enum CloudSubcommand {
    /// Push local pattern changes to the cloud
    Push {
        /// Remote nagual server URL
        #[arg(long, env = "NAGUAL_CLOUD_URL")]
        remote: Option<String>,

        /// API token for authentication
        #[arg(long, env = "NAGUAL_CLOUD_TOKEN")]
        token: Option<String>,

        /// Path to local SQLite database
        #[arg(long, default_value = "./nagual.db")]
        db_path: PathBuf,

        /// Force full push (ignore last sync timestamp)
        #[arg(long)]
        full: bool,
    },

    /// Pull pattern changes from the cloud
    Pull {
        /// Remote nagual server URL
        #[arg(long, env = "NAGUAL_CLOUD_URL")]
        remote: Option<String>,

        /// API token for authentication
        #[arg(long, env = "NAGUAL_CLOUD_TOKEN")]
        token: Option<String>,

        /// Path to local SQLite database
        #[arg(long, default_value = "./nagual.db")]
        db_path: PathBuf,

        /// Force full pull (ignore last sync timestamp)
        #[arg(long)]
        full: bool,
    },

    /// Show sync status with a remote server
    Status {
        /// Remote nagual server URL
        #[arg(long, env = "NAGUAL_CLOUD_URL")]
        remote: Option<String>,

        /// API token for authentication
        #[arg(long, env = "NAGUAL_CLOUD_TOKEN")]
        token: Option<String>,

        /// Path to local SQLite database
        #[arg(long, default_value = "./nagual.db")]
        db_path: PathBuf,
    },
}

impl CloudCommand {
    pub async fn run(&self) -> Result<()> {
        match &self.command {
            CloudSubcommand::Push {
                remote,
                token,
                db_path,
                full,
            } => {
                let remote = resolve_cloud_url(remote.as_deref());
                let token = resolve_cloud_token(token.as_deref())?;

                let db = SqliteDb::open(db_path)?;

                println!("Pushing to {}...", remote);
                let summary =
                    crate::cloud::push::cloud_push(&db, &remote, &token, *full).await?;

                if summary.total == 0 {
                    println!("Already up to date. No patterns to push.");
                } else {
                    println!(
                        "Pushed {} patterns ({} new, {} updated, {} skipped)",
                        summary.total, summary.created, summary.updated, summary.skipped
                    );
                }

                Ok(())
            }

            CloudSubcommand::Pull {
                remote,
                token,
                db_path,
                full,
            } => {
                let remote = resolve_cloud_url(remote.as_deref());
                let token = resolve_cloud_token(token.as_deref())?;

                let storage = init_storage_sqlite_only(db_path).await?;
                let db = SqliteDb::open(db_path)?;

                println!("Pulling from {}...", remote);
                let summary =
                    crate::cloud::pull::cloud_pull(&storage, &db, &remote, &token, *full)
                        .await?;

                if summary.total == 0 {
                    println!("Already up to date. No new patterns from cloud.");
                } else {
                    println!("Pulled {} patterns from cloud", summary.total);
                }

                Ok(())
            }

            CloudSubcommand::Status {
                remote,
                token,
                db_path,
            } => {
                let remote = resolve_cloud_url(remote.as_deref());
                let db = SqliteDb::open(db_path)?;

                // Show local sync state
                sync_state::init_sync_state_table(&db).await?;
                let state = sync_state::get_sync_state(&db, &remote).await?;

                println!("Cloud Sync Status");
                println!("=================");
                println!("  Remote:      {}", remote);

                if let Some(ref state) = state {
                    if let Some(ref push_at) = state.last_push_at {
                        println!("  Last push:   {} ({} patterns)", push_at.format("%Y-%m-%d %H:%M:%S UTC"), state.last_push_count);
                    } else {
                        println!("  Last push:   never");
                    }
                    if let Some(ref pull_at) = state.last_pull_at {
                        println!("  Last pull:   {} ({} patterns)", pull_at.format("%Y-%m-%d %H:%M:%S UTC"), state.last_pull_count);
                    } else {
                        println!("  Last pull:   never");
                    }
                } else {
                    println!("  Last push:   never");
                    println!("  Last pull:   never");
                }

                // Try to reach the remote server
                if let Some(ref token) = token {
                    let client = CloudClient::new(&remote, token);
                    match client.status().await {
                        Ok(status) => {
                            println!("  Server:      {} ({})", status.status,
                                status.pattern_count.map_or("? patterns".to_string(), |c| format!("{} patterns", c)));
                        }
                        Err(e) => {
                            println!("  Server:      unreachable ({})", e);
                        }
                    }
                } else {
                    // Try resolving token for status check
                    match resolve_cloud_token(None) {
                        Ok(token) => {
                            let client = CloudClient::new(&remote, &token);
                            match client.status().await {
                                Ok(status) => {
                                    println!("  Server:      {} ({})", status.status,
                                        status.pattern_count.map_or("? patterns".to_string(), |c| format!("{} patterns", c)));
                                }
                                Err(e) => {
                                    println!("  Server:      unreachable ({})", e);
                                }
                            }
                        }
                        Err(_) => {
                            println!("  Server:      no token configured (use --token or NAGUAL_CLOUD_TOKEN)");
                        }
                    }
                }

                Ok(())
            }
        }
    }
}

/// Resolve the cloud URL from multiple sources.
///
/// Priority:
/// 1. Explicit `--remote` flag
/// 2. `NAGUAL_CLOUD_URL` env var (handled by clap `env`)
/// 3. `~/.nagual/config.toml` → `cloud_url`
/// 4. Default: `http://localhost:3333` (local dev)
///
/// For production deployments, always set `NAGUAL_CLOUD_URL` or `cloud_url`
/// in `~/.nagual/config.toml` to point at your own `nagual serve` instance.
fn resolve_cloud_url(explicit: Option<&str>) -> String {
    if let Some(url) = explicit {
        if !url.is_empty() {
            return url.to_string();
        }
    }

    // Config file fallback
    if let Some(home) = std::env::var("HOME").ok().map(PathBuf::from) {
        let config_path = home.join(".nagual").join("config.toml");
        if let Ok(content) = std::fs::read_to_string(&config_path) {
            for line in content.lines() {
                let trimmed = line.trim();
                if trimmed.starts_with("cloud_url") {
                    if let Some(value) = trimmed.split('=').nth(1) {
                        let url = value.trim().trim_matches('"').trim_matches('\'');
                        if !url.is_empty() {
                            return url.to_string();
                        }
                    }
                }
            }
        }
    }

    "http://localhost:3333".to_string()
}

/// Resolve the cloud API token from multiple sources.
///
/// Priority:
/// 1. Explicit `--token` flag
/// 2. `NAGUAL_CLOUD_TOKEN` env var (handled by clap `env`)
/// 3. `~/.nagual/config.toml` → `cloud_token`
/// 4. `~/.nagual/config.toml` → `api_token` (fallback)
fn resolve_cloud_token(explicit: Option<&str>) -> Result<String> {
    if let Some(token) = explicit {
        if !token.is_empty() {
            return Ok(token.to_string());
        }
    }

    // Config file fallback
    if let Some(home) = std::env::var("HOME").ok().map(PathBuf::from) {
        let config_path = home.join(".nagual").join("config.toml");
        if let Ok(content) = std::fs::read_to_string(&config_path) {
            // First try cloud_token
            for line in content.lines() {
                let trimmed = line.trim();
                if trimmed.starts_with("cloud_token") {
                    if let Some(value) = trimmed.split('=').nth(1) {
                        let token = value.trim().trim_matches('"').trim_matches('\'');
                        if !token.is_empty() {
                            return Ok(token.to_string());
                        }
                    }
                }
            }
            // Fallback to api_token
            for line in content.lines() {
                let trimmed = line.trim();
                if trimmed.starts_with("api_token") && !trimmed.starts_with("api_token_") {
                    if let Some(value) = trimmed.split('=').nth(1) {
                        let token = value.trim().trim_matches('"').trim_matches('\'');
                        if !token.is_empty() {
                            return Ok(token.to_string());
                        }
                    }
                }
            }
        }
    }

    Err(NagualError::Config {
        message: "No cloud token found. Use --token, NAGUAL_CLOUD_TOKEN env var, or set cloud_token in ~/.nagual/config.toml".to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[derive(Parser, Debug)]
    struct TestCli {
        #[command(subcommand)]
        cmd: TestCmd,
    }

    #[derive(clap::Subcommand, Debug)]
    enum TestCmd {
        Cloud(CloudCommand),
    }

    #[test]
    fn test_cloud_push_parse() {
        let args = vec!["test", "cloud", "push", "--token", "abc123"];
        let cli = TestCli::try_parse_from(args);
        assert!(cli.is_ok());
    }

    #[test]
    fn test_cloud_pull_parse() {
        let args = vec!["test", "cloud", "pull", "--full", "--token", "abc"];
        let cli = TestCli::try_parse_from(args);
        assert!(cli.is_ok());
    }

    #[test]
    fn test_cloud_status_parse() {
        let args = vec!["test", "cloud", "status"];
        let cli = TestCli::try_parse_from(args);
        assert!(cli.is_ok());
    }

    #[test]
    fn test_cloud_push_with_remote() {
        let args = vec![
            "test", "cloud", "push",
            "--remote", "http://localhost:3334",
            "--token", "test-token",
            "--db-path", "/tmp/test.db",
        ];
        let cli = TestCli::try_parse_from(args);
        assert!(cli.is_ok());
    }

    #[test]
    fn test_resolve_cloud_url_default() {
        let url = resolve_cloud_url(None);
        // Either from config or default
        assert!(!url.is_empty());
    }

    #[test]
    fn test_resolve_cloud_url_explicit() {
        let url = resolve_cloud_url(Some("http://localhost:3334"));
        assert_eq!(url, "http://localhost:3334");
    }
}
