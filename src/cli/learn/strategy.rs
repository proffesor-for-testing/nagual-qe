//! Strategy cache CLI commands.
//!
//! Provides commands for managing the EGUR-inspired strategy cache:
//! - Store successful approaches per problem category
//! - Search strategies by category
//! - List all cached strategies with reward tracking
//!
//! # Usage Examples
//!
//! ```bash
//! # Store a strategy
//! nagual learn strategy store "debugging" "Binary search for root cause" \
//!   --steps "reproduce,bisect,isolate,verify,fix"
//!
//! # Search strategies
//! nagual learn strategy search "debug"
//!
//! # List all strategies
//! nagual learn strategy list
//! ```

use std::path::PathBuf;

use clap::{Args, Subcommand};

use crate::cli::common::init_storage;
use crate::error::Result;
use crate::reasoning_bank::pattern::PatternId;

/// Arguments for the strategy subcommand.
#[derive(Args, Debug)]
pub struct StrategyArgs {
    #[command(subcommand)]
    pub action: StrategyAction,
}

/// Strategy cache actions.
#[derive(Subcommand, Debug)]
pub enum StrategyAction {
    /// Store a strategy in the cache.
    Store {
        /// Category for the strategy (e.g., "performance", "debugging").
        #[arg(value_name = "CATEGORY")]
        category: String,

        /// Description of the strategy.
        #[arg(value_name = "DESCRIPTION")]
        description: String,

        /// Comma-separated steps.
        #[arg(long)]
        steps: Option<String>,

        /// Path to SQLite database.
        #[arg(long, default_value = "./nagual.db")]
        db_path: PathBuf,
    },

    /// Search strategies by category.
    Search {
        /// Category to search for.
        #[arg(value_name = "CATEGORY")]
        category: String,

        /// Maximum results.
        #[arg(long, default_value = "10")]
        max: usize,

        /// Path to SQLite database.
        #[arg(long, default_value = "./nagual.db")]
        db_path: PathBuf,
    },

    /// List all strategies.
    List {
        /// Maximum results.
        #[arg(long, default_value = "20")]
        max: usize,

        /// Path to SQLite database.
        #[arg(long, default_value = "./nagual.db")]
        db_path: PathBuf,
    },
}

/// Run the strategy subcommand.
pub async fn run(args: &StrategyArgs) -> Result<()> {
    match &args.action {
        StrategyAction::Store { category, description, steps, db_path } => {
            let storage = init_storage(db_path, None).await?;
            let id = PatternId::new().to_string();
            let now = chrono::Utc::now().to_rfc3339();
            let steps_json = steps
                .as_ref()
                .map(|s| {
                    let step_list: Vec<&str> = s.split(',').map(|s| s.trim()).collect();
                    serde_json::to_string(&step_list).unwrap_or_else(|_| "[]".to_string())
                })
                .unwrap_or_else(|| "[]".to_string());

            let sql = r#"
                INSERT INTO strategy_cache (id, category, description, steps, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
            "#;
            storage.adapter().sqlite().execute(
                sql,
                &[&id, category, description, &steps_json, &now, &now],
            ).await?;

            println!("\nStrategy Stored");
            println!("{:-<50}", "");
            println!("  ID: {}", id);
            println!("  Category: {}", category);
            println!("  Description: {}", description);
            if let Some(s) = steps {
                println!("  Steps: {}", s);
            }
        }
        StrategyAction::Search { category, max, db_path } => {
            let storage = init_storage(db_path, None).await?;
            let sql = r#"
                SELECT id, category, description, steps, success_count, failure_count, avg_reward
                FROM strategy_cache
                WHERE category LIKE ?
                ORDER BY avg_reward DESC
                LIMIT ?
            "#;
            let search = format!("%{}%", category);
            let limit_i64 = *max as i64;

            let results: Vec<(String, String, String, String, i32, i32, f64)> = storage
                .adapter()
                .sqlite()
                .query(sql, &[&search, &limit_i64], |row| {
                    Ok((
                        row.get(0)?,
                        row.get(1)?,
                        row.get(2)?,
                        row.get::<_, Option<String>>(3)?.unwrap_or_else(|| "[]".to_string()),
                        row.get(4)?,
                        row.get(5)?,
                        row.get(6)?,
                    ))
                })
                .await?;

            println!("\nStrategy Search: '{}'", category);
            println!("{:-<60}", "");
            if results.is_empty() {
                println!("  No strategies found.");
            } else {
                for (i, (id, cat, desc, steps, succ, fail, reward)) in results.iter().enumerate() {
                    println!(
                        "\n{}. [{}] {} (reward: {:.2}, {}/{} success)",
                        i + 1, cat, desc, reward, succ, succ + fail
                    );
                    if steps != "[]" {
                        println!("   Steps: {}", steps);
                    }
                    println!("   ID: {}", id);
                }
            }
        }
        StrategyAction::List { max, db_path } => {
            let storage = init_storage(db_path, None).await?;
            let sql = r#"
                SELECT id, category, description, success_count, failure_count, avg_reward
                FROM strategy_cache
                ORDER BY avg_reward DESC
                LIMIT ?
            "#;
            let limit_i64 = *max as i64;

            let results: Vec<(String, String, String, i32, i32, f64)> = storage
                .adapter()
                .sqlite()
                .query(sql, &[&limit_i64], |row| {
                    Ok((
                        row.get(0)?,
                        row.get(1)?,
                        row.get(2)?,
                        row.get(3)?,
                        row.get(4)?,
                        row.get(5)?,
                    ))
                })
                .await?;

            println!("\nStrategy Cache ({} entries)", results.len());
            println!("{:-<60}", "");
            if results.is_empty() {
                println!("  No strategies cached yet.");
            } else {
                for (i, (id, cat, desc, succ, fail, reward)) in results.iter().enumerate() {
                    println!(
                        "{}. [{}] {} (reward: {:.2}, {}/{} success) [{}]",
                        i + 1,
                        cat,
                        desc,
                        reward,
                        succ,
                        succ + fail,
                        &id[..8]
                    );
                }
            }
        }
    }

    Ok(())
}
