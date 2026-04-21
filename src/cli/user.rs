//! User management CLI commands for dashboard login.
//!
//! # Usage
//!
//! ```bash
//! nagual user create admin --role admin
//! nagual user list
//! nagual user delete alice
//! ```

use std::path::PathBuf;
use std::sync::Arc;

use clap::{Args, Subcommand};

use crate::db::users::UserStore;
use crate::db::SqliteDb;
use crate::error::Result;

/// User management commands for dashboard login.
#[derive(Args, Debug)]
pub struct UserCommand {
    #[command(subcommand)]
    pub subcommand: UserSubcommand,
}

/// User subcommands.
#[derive(Subcommand, Debug)]
pub enum UserSubcommand {
    /// Create a new dashboard user.
    ///
    /// Generates a random temporary password and prints it once.
    Create(UserCreateArgs),

    /// List all dashboard users.
    List(UserListArgs),

    /// Delete a dashboard user.
    Delete(UserDeleteArgs),
}

/// Arguments for `user create`.
#[derive(Args, Debug)]
pub struct UserCreateArgs {
    /// Username for the new user.
    #[arg(value_name = "USERNAME")]
    pub username: String,

    /// Role: admin (full access) or viewer (read-only dashboard).
    #[arg(long, default_value = "viewer")]
    pub role: String,

    /// Set a specific password (if omitted, a random one is generated).
    #[arg(long)]
    pub password: Option<String>,

    /// Path to SQLite database.
    #[arg(long, default_value = "./nagual.db")]
    pub db_path: PathBuf,

    /// Output as JSON.
    #[arg(long)]
    pub json: bool,
}

/// Arguments for `user list`.
#[derive(Args, Debug)]
pub struct UserListArgs {
    /// Path to SQLite database.
    #[arg(long, default_value = "./nagual.db")]
    pub db_path: PathBuf,

    /// Output as JSON.
    #[arg(long)]
    pub json: bool,
}

/// Arguments for `user delete`.
#[derive(Args, Debug)]
pub struct UserDeleteArgs {
    /// Username to delete.
    #[arg(value_name = "USERNAME")]
    pub username: String,

    /// Path to SQLite database.
    #[arg(long, default_value = "./nagual.db")]
    pub db_path: PathBuf,

    /// Output as JSON.
    #[arg(long)]
    pub json: bool,
}

impl UserCommand {
    pub async fn run(&self) -> Result<()> {
        match &self.subcommand {
            UserSubcommand::Create(args) => run_create(args).await,
            UserSubcommand::List(args) => run_list(args).await,
            UserSubcommand::Delete(args) => run_delete(args).await,
        }
    }
}

/// Generate a random 16-character alphanumeric password.
fn generate_password() -> String {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    (0..16)
        .map(|_| {
            let idx = rng.gen_range(0..62);
            match idx {
                0..=9 => (b'0' + idx) as char,
                10..=35 => (b'a' + idx - 10) as char,
                _ => (b'A' + idx - 36) as char,
            }
        })
        .collect()
}

async fn run_create(args: &UserCreateArgs) -> Result<()> {
    // Validate role
    if args.role != "admin" && args.role != "viewer" {
        eprintln!("Error: role must be 'admin' or 'viewer'");
        std::process::exit(1);
    }

    let db = Arc::new(SqliteDb::open(&args.db_path)?);
    let store = UserStore::new(db).await?;

    let password = args.password.clone().unwrap_or_else(generate_password);
    let user = store.create_user(&args.username, &password, &args.role).await?;

    if args.json {
        let out = serde_json::json!({
            "id": user.id,
            "username": user.username,
            "role": user.role,
            "password": password,
            "created_at": user.created_at.to_rfc3339(),
        });
        println!("{}", serde_json::to_string_pretty(&out).unwrap());
    } else {
        println!();
        println!("  User Created");
        println!("  ============");
        println!("  Username: {}", user.username);
        println!("  Role:     {}", user.role);
        println!("  Password: {}", password);
        println!();
        println!("  Save this password now — it will not be shown again.");
        println!("  Change it via: nagual user create {} --password <new>", user.username);
        println!();
    }

    Ok(())
}

async fn run_list(args: &UserListArgs) -> Result<()> {
    let db = Arc::new(SqliteDb::open(&args.db_path)?);
    let store = UserStore::new(db).await?;

    let users = store.list_users().await?;

    if args.json {
        println!("{}", serde_json::to_string_pretty(&users).unwrap());
        return Ok(());
    }

    if users.is_empty() {
        println!("No users found. Create one with: nagual user create <username> --role admin");
        return Ok(());
    }

    println!();
    println!(
        "  {:<20} {:<10} {:<24} {}",
        "USERNAME", "ROLE", "CREATED", "LAST LOGIN"
    );
    println!("  {}", "-".repeat(74));

    for user in &users {
        let last_login = user
            .last_login
            .map(|dt| dt.format("%Y-%m-%d %H:%M").to_string())
            .unwrap_or_else(|| "never".to_string());
        println!(
            "  {:<20} {:<10} {:<24} {}",
            user.username,
            user.role,
            user.created_at.format("%Y-%m-%d %H:%M"),
            last_login
        );
    }
    println!();

    Ok(())
}

async fn run_delete(args: &UserDeleteArgs) -> Result<()> {
    let db = Arc::new(SqliteDb::open(&args.db_path)?);
    let store = UserStore::new(db).await?;

    let deleted = store.delete_user(&args.username).await?;

    if args.json {
        let out = serde_json::json!({
            "username": args.username,
            "deleted": deleted,
        });
        println!("{}", serde_json::to_string_pretty(&out).unwrap());
    } else if deleted {
        println!("User '{}' deleted.", args.username);
    } else {
        println!("No user found with username '{}'.", args.username);
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
        User(UserCommand),
    }

    #[test]
    fn test_parse_create() {
        let args = vec!["test", "user", "create", "admin", "--role", "admin"];
        let cli = TestCli::try_parse_from(args).unwrap();
        match cli.cmd {
            TestCmd::User(cmd) => match cmd.subcommand {
                UserSubcommand::Create(a) => {
                    assert_eq!(a.username, "admin");
                    assert_eq!(a.role, "admin");
                    assert!(a.password.is_none());
                }
                _ => panic!("Expected Create"),
            },
        }
    }

    #[test]
    fn test_parse_create_with_password() {
        let args = vec![
            "test", "user", "create", "bob", "--role", "viewer", "--password", "secret123",
        ];
        let cli = TestCli::try_parse_from(args).unwrap();
        match cli.cmd {
            TestCmd::User(cmd) => match cmd.subcommand {
                UserSubcommand::Create(a) => {
                    assert_eq!(a.username, "bob");
                    assert_eq!(a.role, "viewer");
                    assert_eq!(a.password, Some("secret123".to_string()));
                }
                _ => panic!("Expected Create"),
            },
        }
    }

    #[test]
    fn test_parse_list() {
        let args = vec!["test", "user", "list"];
        let cli = TestCli::try_parse_from(args).unwrap();
        match cli.cmd {
            TestCmd::User(cmd) => {
                assert!(matches!(cmd.subcommand, UserSubcommand::List(_)));
            }
        }
    }

    #[test]
    fn test_parse_delete() {
        let args = vec!["test", "user", "delete", "old-user"];
        let cli = TestCli::try_parse_from(args).unwrap();
        match cli.cmd {
            TestCmd::User(cmd) => match cmd.subcommand {
                UserSubcommand::Delete(a) => {
                    assert_eq!(a.username, "old-user");
                }
                _ => panic!("Expected Delete"),
            },
        }
    }

    #[test]
    fn test_generate_password() {
        let pw = generate_password();
        assert_eq!(pw.len(), 16);
        assert!(pw.chars().all(|c| c.is_ascii_alphanumeric()));
    }
}
