//! API key management CLI commands.
//!
//! Create, list, and revoke per-agent API keys for `nagual serve` authentication.
//!
//! # Usage
//!
//! ```bash
//! nagual apikey create claude-cowork --scopes read,write
//! nagual apikey list
//! nagual apikey revoke claude-cowork
//! ```

use std::path::PathBuf;
use std::sync::Arc;

use clap::{Args, Subcommand};

use crate::db::SqliteDb;
use crate::error::Result;
use crate::security::ApiKeyStore;

/// API key management commands.
#[derive(Args, Debug)]
pub struct ApiKeyCommand {
    #[command(subcommand)]
    pub subcommand: ApiKeySubcommand,
}

/// API key subcommands.
#[derive(Subcommand, Debug)]
pub enum ApiKeySubcommand {
    /// Create a new API key for an agent.
    ///
    /// Prints the plaintext key exactly once — it cannot be recovered.
    Create(CreateArgs),

    /// List all API keys.
    List(ListArgs),

    /// Revoke an API key by name or ID.
    Revoke(RevokeArgs),
}

/// Arguments for `apikey create`.
#[derive(Args, Debug)]
pub struct CreateArgs {
    /// Human-readable name for the key (e.g., "claude-cowork").
    #[arg(value_name = "NAME")]
    pub name: String,

    /// Comma-separated scopes (read, write, admin).
    #[arg(long, default_value = "read,write", value_delimiter = ',')]
    pub scopes: Vec<String>,

    /// Path to SQLite database.
    #[arg(long, default_value = "./nagual.db")]
    pub db_path: PathBuf,

    /// Output as JSON.
    #[arg(long)]
    pub json: bool,
}

/// Arguments for `apikey list`.
#[derive(Args, Debug)]
pub struct ListArgs {
    /// Include revoked keys.
    #[arg(long)]
    pub include_revoked: bool,

    /// Path to SQLite database.
    #[arg(long, default_value = "./nagual.db")]
    pub db_path: PathBuf,

    /// Output as JSON.
    #[arg(long)]
    pub json: bool,
}

/// Arguments for `apikey revoke`.
#[derive(Args, Debug)]
pub struct RevokeArgs {
    /// Name or ID of the key to revoke.
    #[arg(value_name = "NAME_OR_ID")]
    pub name_or_id: String,

    /// Path to SQLite database.
    #[arg(long, default_value = "./nagual.db")]
    pub db_path: PathBuf,

    /// Output as JSON.
    #[arg(long)]
    pub json: bool,
}

impl ApiKeyCommand {
    pub async fn run(&self) -> Result<()> {
        match &self.subcommand {
            ApiKeySubcommand::Create(args) => run_create(args).await,
            ApiKeySubcommand::List(args) => run_list(args).await,
            ApiKeySubcommand::Revoke(args) => run_revoke(args).await,
        }
    }
}

async fn run_create(args: &CreateArgs) -> Result<()> {
    let db = Arc::new(SqliteDb::open(&args.db_path)?);
    let store = ApiKeyStore::new(db).await?;

    let (plaintext, record) = store
        .create_key(&args.name, &args.scopes, None)
        .await?;

    if args.json {
        let out = serde_json::json!({
            "key": plaintext,
            "id": record.id,
            "name": record.name,
            "prefix": record.key_prefix,
            "scopes": record.scopes,
            "created_at": record.created_at.to_rfc3339(),
        });
        println!("{}", serde_json::to_string_pretty(&out).unwrap());
    } else {
        println!();
        println!("  API Key Created");
        println!("  ===============");
        println!("  Name:    {}", record.name);
        println!("  Scopes:  {}", record.scopes.join(", "));
        println!("  Prefix:  {}", record.key_prefix);
        println!();
        println!("  Key:     {}", plaintext);
        println!();
        println!("  Save this key now — it will not be shown again.");
        println!();
    }

    Ok(())
}

async fn run_list(args: &ListArgs) -> Result<()> {
    let db = Arc::new(SqliteDb::open(&args.db_path)?);
    let store = ApiKeyStore::new(db).await?;

    let keys = store.list_keys(args.include_revoked).await?;

    if args.json {
        println!("{}", serde_json::to_string_pretty(&keys).unwrap());
        return Ok(());
    }

    if keys.is_empty() {
        println!("No API keys found.");
        return Ok(());
    }

    println!();
    println!("  {:<20} {:<12} {:<20} {:<20} {}",
        "NAME", "PREFIX", "SCOPES", "LAST USED", "STATUS");
    println!("  {}", "-".repeat(84));

    for key in &keys {
        let last_used = key
            .last_used_at
            .map(|dt| dt.format("%Y-%m-%d %H:%M").to_string())
            .unwrap_or_else(|| "never".to_string());
        let status = if key.revoked_at.is_some() { "revoked" } else { "active" };
        println!("  {:<20} {:<12} {:<20} {:<20} {}",
            key.name, key.key_prefix, key.scopes.join(","), last_used, status);
    }
    println!();

    Ok(())
}

async fn run_revoke(args: &RevokeArgs) -> Result<()> {
    let db = Arc::new(SqliteDb::open(&args.db_path)?);
    let store = ApiKeyStore::new(db).await?;

    let revoked = store.revoke_key(&args.name_or_id).await?;

    if args.json {
        let out = serde_json::json!({
            "name_or_id": args.name_or_id,
            "revoked": revoked,
        });
        println!("{}", serde_json::to_string_pretty(&out).unwrap());
    } else if revoked {
        println!("Key '{}' revoked.", args.name_or_id);
    } else {
        println!("No active key found for '{}'.", args.name_or_id);
    }

    Ok(())
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
        Apikey(ApiKeyCommand),
    }

    #[test]
    fn test_parse_create() {
        let args = vec!["test", "apikey", "create", "my-agent"];
        let cli = TestCli::try_parse_from(args).unwrap();
        match cli.cmd {
            TestCmd::Apikey(cmd) => match cmd.subcommand {
                ApiKeySubcommand::Create(a) => {
                    assert_eq!(a.name, "my-agent");
                    assert_eq!(a.scopes, vec!["read", "write"]);
                }
                _ => panic!("Expected Create"),
            },
        }
    }

    #[test]
    fn test_parse_create_with_scopes() {
        let args = vec!["test", "apikey", "create", "admin-bot", "--scopes", "read,write,admin"];
        let cli = TestCli::try_parse_from(args).unwrap();
        match cli.cmd {
            TestCmd::Apikey(cmd) => match cmd.subcommand {
                ApiKeySubcommand::Create(a) => {
                    assert_eq!(a.name, "admin-bot");
                    assert_eq!(a.scopes, vec!["read", "write", "admin"]);
                }
                _ => panic!("Expected Create"),
            },
        }
    }

    #[test]
    fn test_parse_list() {
        let args = vec!["test", "apikey", "list", "--include-revoked"];
        let cli = TestCli::try_parse_from(args).unwrap();
        match cli.cmd {
            TestCmd::Apikey(cmd) => match cmd.subcommand {
                ApiKeySubcommand::List(a) => {
                    assert!(a.include_revoked);
                }
                _ => panic!("Expected List"),
            },
        }
    }

    #[test]
    fn test_parse_revoke() {
        let args = vec!["test", "apikey", "revoke", "old-agent"];
        let cli = TestCli::try_parse_from(args).unwrap();
        match cli.cmd {
            TestCmd::Apikey(cmd) => match cmd.subcommand {
                ApiKeySubcommand::Revoke(a) => {
                    assert_eq!(a.name_or_id, "old-agent");
                }
                _ => panic!("Expected Revoke"),
            },
        }
    }
}
