//! Gene Transfusion CLI command.
//!
//! Extracts reusable patterns from existing codebases and stores them
//! in the ReasoningBank for self-learning.
//!
//! # Usage Examples
//!
//! ```bash
//! # Scan current directory (dry run)
//! nagual transfuse . --dry-run
//!
//! # Scan a specific path with verbose output
//! nagual transfuse /path/to/project --verbose
//!
//! # Scan with lower confidence threshold
//! nagual transfuse ./src --min-confidence 0.6
//!
//! # Output as JSON
//! nagual transfuse ./src --json
//!
//! # Limit files processed
//! nagual transfuse ./src --max-files 100
//! ```

use std::path::PathBuf;
use std::sync::Arc;

use clap::Args;
use serde::Serialize;

use crate::db::{DualWriteAdapter, DualWriteConfig, PostgresDb, SqliteDb};
use crate::error::Result;
use crate::reasoning_bank::storage::{PatternStorage, StorageConfig};
use crate::reasoning_bank::transfusion::{ExtractedPattern, Transfuser, TransfusionConfig, TransfusionResult};

/// Extract patterns from existing codebases (Gene Transfusion).
///
/// Scans source code files to detect common patterns (error handling,
/// async patterns, API design, testing, database patterns) and stores
/// them in the ReasoningBank for future reference and self-learning.
#[derive(Args, Debug)]
pub struct TransfuseCommand {
    /// Path to scan for patterns.
    #[arg(value_name = "PATH", default_value = ".")]
    pub path: PathBuf,

    /// Minimum confidence threshold (0.0-1.0).
    #[arg(long, default_value = "0.7")]
    pub min_confidence: f32,

    /// Dry run - show what would be extracted without storing.
    #[arg(long)]
    pub dry_run: bool,

    /// File extensions to include (comma-separated).
    /// Defaults: rs,py,ts,tsx,js,jsx,go,java,kt,swift,rb,ex,exs
    #[arg(long, value_delimiter = ',')]
    pub extensions: Vec<String>,

    /// Directories to exclude (comma-separated).
    /// Defaults: target,node_modules,.git,vendor,dist,build
    #[arg(long, value_delimiter = ',')]
    pub exclude_dirs: Vec<String>,

    /// Maximum number of files to process (0 = unlimited).
    #[arg(long, default_value = "0")]
    pub max_files: usize,

    /// Maximum file size in KB (default: 1024 = 1MB).
    #[arg(long, default_value = "1024")]
    pub max_file_size: usize,

    /// Output results as JSON.
    #[arg(long)]
    pub json: bool,

    /// Show verbose output including pattern details.
    #[arg(short, long)]
    pub verbose: bool,

    /// Path to SQLite database for storing patterns.
    #[arg(long, default_value = "./nagual.db")]
    pub db_path: PathBuf,

    /// PostgreSQL connection URL for dual-write.
    #[arg(long, env = "DATABASE_URL")]
    pub postgres_url: Option<String>,

    /// Show extracted patterns grouped by detector.
    #[arg(long)]
    pub show_by_detector: bool,
}

impl TransfuseCommand {
    /// Execute the transfuse command.
    pub async fn run(&self) -> Result<()> {
        tracing::info!(
            path = %self.path.display(),
            dry_run = self.dry_run,
            min_confidence = self.min_confidence,
            "Starting Gene Transfusion"
        );

        // Build configuration
        let mut config = TransfusionConfig::default();
        config.min_confidence = self.min_confidence;
        config.dry_run = self.dry_run;
        config.max_files = self.max_files;
        config.max_file_size = self.max_file_size * 1024; // Convert KB to bytes

        if !self.extensions.is_empty() {
            config.include_extensions = self.extensions.clone();
        }

        if !self.exclude_dirs.is_empty() {
            config.exclude_dirs = self.exclude_dirs.clone();
        }

        // Run transfusion
        let transfuser = Transfuser::new(config);
        let result = transfuser.transfuse(&self.path)?;

        // Store patterns if not dry run
        let stored_count = if !self.dry_run && !result.patterns.is_empty() {
            self.store_patterns(&result.patterns).await?
        } else {
            0
        };

        // Output results
        if self.json {
            self.output_json(&result, stored_count)?;
        } else {
            self.output_text(&result, stored_count);
        }

        Ok(())
    }

    async fn store_patterns(&self, patterns: &[ExtractedPattern]) -> Result<usize> {
        // Initialize storage
        if let Some(parent) = self.db_path.parent() {
            if !parent.exists() {
                std::fs::create_dir_all(parent)?;
            }
        }

        let sqlite = Arc::new(SqliteDb::open(&self.db_path)?);

        let postgres = if let Some(ref url) = self.postgres_url {
            match PostgresDb::connect(url, 5).await {
                Ok(pg) => Some(Arc::new(pg)),
                Err(e) => {
                    tracing::warn!(error = %e, "PostgreSQL unavailable, using SQLite only");
                    if !self.json {
                        eprintln!("Warning: PostgreSQL unavailable ({}), using SQLite only", e);
                    }
                    None
                }
            }
        } else {
            None
        };

        let config = DualWriteConfig {
            dlq_path: self.db_path
                .with_extension("dlq.db")
                .to_string_lossy()
                .to_string(),
            ..Default::default()
        };
        let adapter = Arc::new(DualWriteAdapter::new(sqlite, postgres, config)?);
        let storage = PatternStorage::new(adapter, StorageConfig::default()).await?;

        let mut stored = 0;
        for extracted in patterns {
            let pattern = extracted.to_pattern();
            match storage.store_pattern(&pattern).await {
                Ok(_) => stored += 1,
                Err(e) => {
                    tracing::warn!(
                        error = %e,
                        file = %extracted.source_file,
                        "Failed to store pattern"
                    );
                }
            }
        }

        // Allow background PostgreSQL writes to complete
        if self.postgres_url.is_some() {
            tokio::time::sleep(std::time::Duration::from_millis(200)).await;
        }

        Ok(stored)
    }

    fn output_json(&self, result: &TransfusionResult, stored_count: usize) -> Result<()> {
        let output = JsonOutput {
            files_scanned: result.files_scanned,
            patterns_extracted: result.patterns_extracted,
            patterns_stored: stored_count,
            patterns_skipped: result.patterns_skipped,
            by_category: result.by_category.clone(),
            by_detector: result.by_detector.clone(),
            dry_run: self.dry_run,
            patterns: if self.verbose {
                Some(result.patterns.clone())
            } else {
                None
            },
            errors: if result.errors.is_empty() {
                None
            } else {
                Some(result.errors.clone())
            },
        };
        println!("{}", serde_json::to_string_pretty(&output)?);
        Ok(())
    }

    fn output_text(&self, result: &TransfusionResult, stored_count: usize) {
        println!("\n{}", "=".repeat(60));
        println!("  Gene Transfusion Complete");
        if self.dry_run {
            println!("  (DRY RUN - no patterns stored)");
        }
        println!("{}", "=".repeat(60));

        println!("\n  Path scanned: {}", self.path.display());
        println!("  Files scanned: {}", result.files_scanned);
        println!("  Patterns extracted: {}", result.patterns_extracted);
        if !self.dry_run {
            println!("  Patterns stored: {}", stored_count);
        }
        println!("  Patterns skipped: {}", result.patterns_skipped);

        if !result.by_category.is_empty() {
            println!("\n  By Category:");
            let mut categories: Vec<_> = result.by_category.iter().collect();
            categories.sort_by(|a, b| b.1.cmp(a.1));
            for (category, count) in categories {
                println!("    {:<25} {}", category, count);
            }
        }

        if self.show_by_detector && !result.by_detector.is_empty() {
            println!("\n  By Detector:");
            let mut detectors: Vec<_> = result.by_detector.iter().collect();
            detectors.sort_by(|a, b| b.1.cmp(a.1));
            for (detector, count) in detectors {
                println!("    {:<25} {}", detector, count);
            }
        }

        if self.verbose && !result.patterns.is_empty() {
            println!("\n  Extracted Patterns:");
            println!("  {}", "-".repeat(56));

            for (i, pattern) in result.patterns.iter().enumerate() {
                if i >= 10 && !self.verbose {
                    println!("\n  ... and {} more patterns", result.patterns.len() - 10);
                    break;
                }

                println!("\n  {}. {}", i + 1, truncate(&pattern.problem, 50));
                println!("     Domain: {}", pattern.domain);
                println!("     Source: {}:{}", pattern.source_file, pattern.line_number);
                println!("     Confidence: {:.0}%", pattern.confidence * 100.0);
                println!("     Tags: {}", pattern.tags.join(", "));

                if self.verbose {
                    println!("     Solution preview:");
                    for line in pattern.solution.lines().take(5) {
                        println!("       {}", truncate(line, 70));
                    }
                    if pattern.solution.lines().count() > 5 {
                        println!("       ... (truncated)");
                    }
                }
            }
        }

        if !result.errors.is_empty() {
            println!("\n  Errors ({}):", result.errors.len());
            for (path, error) in result.errors.iter().take(5) {
                println!("    {}: {}", truncate(path, 40), truncate(error, 30));
            }
            if result.errors.len() > 5 {
                println!("    ... and {} more errors", result.errors.len() - 5);
            }
        }

        println!("\n{}\n", "=".repeat(60));
    }
}

#[derive(Serialize)]
struct JsonOutput {
    files_scanned: usize,
    patterns_extracted: usize,
    patterns_stored: usize,
    patterns_skipped: usize,
    by_category: std::collections::HashMap<String, usize>,
    by_detector: std::collections::HashMap<String, usize>,
    dry_run: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    patterns: Option<Vec<ExtractedPattern>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    errors: Option<std::collections::HashMap<String, String>>,
}

/// Truncate a string to a maximum length.
fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len.saturating_sub(3)])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[derive(Parser, Debug)]
    struct TestCli {
        #[command(flatten)]
        transfuse: TransfuseCommand,
    }

    #[test]
    fn test_cli_parse_transfuse_basic() {
        let args = vec!["test", "./src"];
        let cli = TestCli::try_parse_from(args);
        assert!(cli.is_ok());
        let cli = cli.unwrap();
        assert_eq!(cli.transfuse.path, PathBuf::from("./src"));
        assert!(!cli.transfuse.dry_run);
    }

    #[test]
    fn test_cli_parse_transfuse_dry_run() {
        let args = vec!["test", "./src", "--dry-run"];
        let cli = TestCli::try_parse_from(args);
        assert!(cli.is_ok());
        assert!(cli.unwrap().transfuse.dry_run);
    }

    #[test]
    fn test_cli_parse_transfuse_with_options() {
        let args = vec![
            "test",
            "/path/to/project",
            "--min-confidence",
            "0.8",
            "--extensions",
            "rs,go",
            "--exclude-dirs",
            "vendor,build",
            "--max-files",
            "100",
            "--verbose",
            "--json",
        ];
        let cli = TestCli::try_parse_from(args);
        assert!(cli.is_ok());
        let cmd = cli.unwrap().transfuse;
        assert_eq!(cmd.path, PathBuf::from("/path/to/project"));
        assert!((cmd.min_confidence - 0.8).abs() < 0.001);
        assert_eq!(cmd.extensions, vec!["rs", "go"]);
        assert_eq!(cmd.exclude_dirs, vec!["vendor", "build"]);
        assert_eq!(cmd.max_files, 100);
        assert!(cmd.verbose);
        assert!(cmd.json);
    }

    #[test]
    fn test_truncate() {
        assert_eq!(truncate("Hello", 10), "Hello");
        assert_eq!(truncate("Hello World!", 8), "Hello...");
        assert_eq!(truncate("", 5), "");
    }
}
