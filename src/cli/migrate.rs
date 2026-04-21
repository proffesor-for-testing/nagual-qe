//! Migration command implementation.
//!
//! Provides CLI commands for database migrations:
//! - `migrate status` - Show current migration status
//! - `migrate up` - Apply pending migrations
//! - `migrate rollback` - Revert last migration(s)
//! - `migrate create <name>` - Create a new migration

use std::fs;
use std::path::PathBuf;
use std::sync::Arc;

use chrono::Utc;
use clap::{Args, Subcommand};

use crate::db::{DatabaseConfig, SqliteDb};
use crate::error::Result;
use crate::migration::{
    generate_migration_template, migrations_dir, CoordinationResult, DualMigrationCoordinator,
    MigrationOutcome, MigrationRunner,
};

/// Database migration command.
///
/// Applies pending migrations to SQLite and/or PostgreSQL databases.
/// Supports bidirectional sync and conflict detection.
#[derive(Args, Debug)]
pub struct MigrateCommand {
    #[command(subcommand)]
    pub action: Option<MigrateAction>,

    /// Apply pending migrations (legacy flag, use `migrate up` instead)
    #[arg(long, conflicts_with = "down", hide = true)]
    pub up: bool,

    /// Rollback migrations (legacy flag, use `migrate rollback` instead)
    #[arg(long, conflicts_with = "up", hide = true)]
    pub down: bool,

    /// Number of migrations to apply/rollback
    #[arg(short = 'n', long, default_value = "1")]
    pub steps: u32,

    /// Target specific database backend
    #[arg(long, value_parser = ["sqlite", "postgres", "both"])]
    pub target: Option<String>,

    /// Show migration status without applying (legacy, use `migrate status`)
    #[arg(long, hide = true)]
    pub status: bool,

    /// Force migration even with pending conflicts
    #[arg(long)]
    pub force: bool,

    /// Dry run - show what would be done without executing
    #[arg(long)]
    pub dry_run: bool,

    /// Path to SQLite database
    #[arg(long, default_value = "nagual.db")]
    pub sqlite_path: String,

    /// PostgreSQL connection URL
    #[arg(long, env = "DATABASE_URL")]
    pub postgres_url: Option<String>,

    /// Path to migrations directory
    #[arg(long)]
    pub migrations_path: Option<PathBuf>,
}

/// Migration subcommands.
#[derive(Subcommand, Debug)]
pub enum MigrateAction {
    /// Show current migration status.
    Status,

    /// Apply all pending migrations.
    Up {
        /// Maximum number of migrations to apply.
        #[arg(short = 'n', long)]
        limit: Option<u32>,
    },

    /// Rollback the last migration(s).
    Rollback {
        /// Number of migrations to rollback.
        #[arg(short = 'n', long, default_value = "1")]
        steps: u32,
    },

    /// Create a new migration file.
    Create {
        /// Name of the migration (e.g., "create_users_table").
        name: String,
    },

    /// Sync databases to consistent state.
    Sync,

    /// Force reset migration state (dangerous).
    Reset {
        /// Confirm reset operation.
        #[arg(long)]
        confirm: bool,
    },

    /// Migrate linear rewards to Bayesian quality scores.
    ///
    /// Converts existing pattern reward values (0.0-1.0) to Beta distribution
    /// parameters (alpha, beta). This is a one-time migration that preserves
    /// original rewards. Only patterns still at the default Beta(1,1) prior
    /// with a non-zero reward are updated.
    Rewards(RewardsArgs),

    /// Re-embed patterns using the hash embedder.
    ///
    /// Generates hash-based embeddings for patterns that don't have them yet.
    /// Useful when running without ONNX support. The hash embedder produces
    /// deterministic 128-dimensional vectors using SHAKE-256 structured hashing.
    Embeddings(EmbeddingsArgs),
}

/// Arguments for the rewards migration command.
#[derive(Args, Debug)]
pub struct RewardsArgs {
    /// Database path
    #[arg(long, default_value = "nagual.db")]
    pub db_path: String,

    /// Dry run (show what would be migrated without making changes)
    #[arg(long)]
    pub dry_run: bool,
}

/// Arguments for the embeddings migration command.
#[derive(Args, Debug)]
pub struct EmbeddingsArgs {
    /// Database path
    #[arg(long, default_value = "nagual.db")]
    pub db_path: String,

    /// Only embed patterns missing embeddings
    #[arg(long)]
    pub missing_only: bool,

    /// Dry run (show what would be migrated without making changes)
    #[arg(long)]
    pub dry_run: bool,
}

impl MigrateCommand {
    /// Execute the migration command.
    pub async fn run(&self) -> Result<()> {
        // Handle subcommand if provided
        if let Some(ref action) = self.action {
            return self.run_action(action).await;
        }

        // Handle legacy flags for backward compatibility
        if self.status {
            return self.show_status().await;
        }

        if self.up {
            return self.migrate_up(self.steps).await;
        }

        if self.down {
            return self.migrate_rollback(self.steps).await;
        }

        // Default to showing status
        self.show_status().await
    }

    /// Run a specific migration action.
    async fn run_action(&self, action: &MigrateAction) -> Result<()> {
        match action {
            MigrateAction::Status => self.show_status().await,
            MigrateAction::Up { limit } => self.migrate_up(limit.unwrap_or(u32::MAX)).await,
            MigrateAction::Rollback { steps } => self.migrate_rollback(*steps).await,
            MigrateAction::Create { name } => self.create_migration(name).await,
            MigrateAction::Sync => self.sync_databases().await,
            MigrateAction::Reset { confirm } => self.reset_migrations(*confirm).await,
            MigrateAction::Rewards(args) => self.migrate_rewards(args).await,
            MigrateAction::Embeddings(args) => self.migrate_embeddings(args).await,
        }
    }

    /// Get the migrations directory path.
    fn migrations_path(&self) -> PathBuf {
        self.migrations_path
            .clone()
            .unwrap_or_else(migrations_dir)
    }

    /// Create a database configuration from CLI options.
    fn db_config(&self) -> DatabaseConfig {
        DatabaseConfig {
            sqlite_path: self.sqlite_path.clone(),
            postgres_url: self.postgres_url.clone(),
            ..Default::default()
        }
    }

    /// Show current migration status for all databases.
    async fn show_status(&self) -> Result<()> {
        println!("Migration Status");
        println!("================\n");

        // Show SQLite status
        let sqlite_db = SqliteDb::open(&self.sqlite_path)?;
        let runner = MigrationRunner::with_migrations_path(sqlite_db, self.migrations_path());
        let sqlite_status = runner.status().await?;

        println!("SQLite Database: {}", self.sqlite_path);
        println!("-----------------{}", "-".repeat(self.sqlite_path.len()));

        if let Some(version) = sqlite_status.current_version {
            println!("  Current version: {}", version);
        } else {
            println!("  Current version: (none - no migrations applied)");
        }

        println!("  Applied migrations: {}", sqlite_status.applied.len());
        println!("  Pending migrations: {}", sqlite_status.pending.len());

        if sqlite_status.is_locked {
            println!("  Status: LOCKED (another process is running migrations)");
        }

        if let Some(ref checkpoint) = sqlite_status.checkpoint {
            println!(
                "  Checkpoint: Migration {} at step {}/{}",
                checkpoint.version, checkpoint.step_index, checkpoint.total_steps
            );
        }

        // Show applied migrations
        if !sqlite_status.applied.is_empty() {
            println!("\n  Applied:");
            for m in sqlite_status.applied.iter().rev().take(5) {
                println!(
                    "    {} - {} ({}ms)",
                    m.version,
                    m.description,
                    m.execution_time_ms.unwrap_or(0)
                );
            }
            if sqlite_status.applied.len() > 5 {
                println!("    ... and {} more", sqlite_status.applied.len() - 5);
            }
        }

        // Show pending migrations
        if !sqlite_status.pending.is_empty() {
            println!("\n  Pending:");
            for m in sqlite_status.pending.iter().take(5) {
                println!("    {} - {}", m.version, m.name);
            }
            if sqlite_status.pending.len() > 5 {
                println!("    ... and {} more", sqlite_status.pending.len() - 5);
            }
        }

        // PostgreSQL status if configured
        if let Some(ref pg_url) = self.postgres_url {
            println!("\nPostgreSQL Database:");
            println!("-------------------");

            // Mask password in URL
            let masked_url = mask_password(pg_url);
            println!("  URL: {}", masked_url);

            // Connect and check status
            match sqlx::postgres::PgPoolOptions::new()
                .max_connections(1)
                .connect(pg_url)
                .await
            {
                Ok(pool) => {
                    let pg_runner = crate::migration::PostgresMigrationRunner::with_migrations_path(
                        pool,
                        self.migrations_path(),
                    );

                    match pg_runner.status().await {
                        Ok(pg_status) => {
                            if let Some(version) = pg_status.current_version {
                                println!("  Current version: {}", version);
                            } else {
                                println!("  Current version: (none)");
                            }
                            println!("  Applied migrations: {}", pg_status.applied.len());
                            println!("  Pending migrations: {}", pg_status.pending.len());

                            // Check consistency
                            if sqlite_status.current_version != pg_status.current_version {
                                println!("\n  WARNING: Databases are INCONSISTENT!");
                                println!(
                                    "    SQLite: v{:?}, PostgreSQL: v{:?}",
                                    sqlite_status.current_version, pg_status.current_version
                                );
                                println!("    Run `migrate sync` to synchronize.");
                            }
                        }
                        Err(e) => {
                            println!("  Status: Error - {}", e);
                        }
                    }
                }
                Err(e) => {
                    println!("  Status: Connection failed - {}", e);
                }
            }
        } else {
            println!("\nPostgreSQL: Not configured");
            println!("  Set --postgres-url or DATABASE_URL to enable cloud sync.");
        }

        println!();
        Ok(())
    }

    /// Apply pending migrations.
    async fn migrate_up(&self, limit: u32) -> Result<()> {
        if self.dry_run {
            println!("DRY RUN: Would apply up to {} migration(s)", limit);

            let sqlite_db = SqliteDb::open(&self.sqlite_path)?;
            let runner = MigrationRunner::with_migrations_path(sqlite_db, self.migrations_path());
            let status = runner.status().await?;

            if status.pending.is_empty() {
                println!("No pending migrations.");
            } else {
                println!("Would apply:");
                for m in status.pending.iter().take(limit as usize) {
                    println!("  {} - {}", m.version, m.name);
                }
            }
            return Ok(());
        }

        let target = self.target.as_deref().unwrap_or("both");
        println!("Applying migrations to {}...\n", target);

        match target {
            "sqlite" => {
                let sqlite_db = SqliteDb::open(&self.sqlite_path)?;
                let mut runner =
                    MigrationRunner::with_migrations_path(sqlite_db, self.migrations_path());

                match runner.run_up().await {
                    Ok(applied) => {
                        println!("Applied {} migration(s) to SQLite:", applied.len());
                        for m in &applied {
                            println!(
                                "  {} - {} ({}ms)",
                                m.version,
                                m.description,
                                m.execution_time_ms.unwrap_or(0)
                            );
                        }
                    }
                    Err(e) => {
                        if e.to_string().contains("No pending migrations") {
                            println!("No pending migrations.");
                        } else {
                            return Err(e);
                        }
                    }
                }
            }
            "postgres" => {
                let Some(ref pg_url) = self.postgres_url else {
                    println!("PostgreSQL not configured. Set --postgres-url or DATABASE_URL.");
                    return Ok(());
                };

                let pool = sqlx::postgres::PgPoolOptions::new()
                    .max_connections(2)
                    .connect(pg_url)
                    .await?;

                let runner = crate::migration::PostgresMigrationRunner::with_migrations_path(
                    pool,
                    self.migrations_path(),
                );

                match runner.run_up().await {
                    Ok(applied) => {
                        println!("Applied {} migration(s) to PostgreSQL:", applied.len());
                        for m in &applied {
                            println!(
                                "  {} - {} ({}ms)",
                                m.version,
                                m.description,
                                m.execution_time_ms.unwrap_or(0)
                            );
                        }
                    }
                    Err(e) => {
                        if e.to_string().contains("No pending migrations") {
                            println!("No pending migrations.");
                        } else {
                            return Err(e);
                        }
                    }
                }
            }
            "both" | _ => {
                let result = self.run_coordinated_up().await?;
                self.print_coordination_result(&result);
            }
        }

        Ok(())
    }

    /// Run coordinated migration on both databases.
    async fn run_coordinated_up(&self) -> Result<CoordinationResult> {
        let sqlite = Arc::new(SqliteDb::open(&self.sqlite_path)?);

        let postgres = if let Some(ref pg_url) = self.postgres_url {
            Some(
                sqlx::postgres::PgPoolOptions::new()
                    .max_connections(2)
                    .connect(pg_url)
                    .await?,
            )
        } else {
            None
        };

        let coordinator = if let Some(pool) = postgres {
            DualMigrationCoordinator::new(sqlite, pool)
        } else {
            DualMigrationCoordinator::new_sqlite_only(sqlite)
        };

        coordinator.run_up().await
    }

    /// Rollback migration(s).
    async fn migrate_rollback(&self, steps: u32) -> Result<()> {
        if self.dry_run {
            println!("DRY RUN: Would rollback {} migration(s)", steps);
            return Ok(());
        }

        let target = self.target.as_deref().unwrap_or("both");
        println!("Rolling back {} migration(s) from {}...\n", steps, target);

        for _ in 0..steps {
            match target {
                "sqlite" => {
                    let sqlite_db = SqliteDb::open(&self.sqlite_path)?;
                    let mut runner =
                        MigrationRunner::with_migrations_path(sqlite_db, self.migrations_path());

                    match runner.run_down().await {
                        Ok(rolled_back) => {
                            println!(
                                "Rolled back: {} - {}",
                                rolled_back.version, rolled_back.description
                            );
                        }
                        Err(e) => {
                            println!("Rollback error: {}", e);
                            break;
                        }
                    }
                }
                "postgres" => {
                    let Some(ref pg_url) = self.postgres_url else {
                        println!("PostgreSQL not configured.");
                        return Ok(());
                    };

                    let pool = sqlx::postgres::PgPoolOptions::new()
                        .max_connections(2)
                        .connect(pg_url)
                        .await?;

                    let runner = crate::migration::PostgresMigrationRunner::with_migrations_path(
                        pool,
                        self.migrations_path(),
                    );

                    match runner.run_down().await {
                        Ok(rolled_back) => {
                            println!(
                                "Rolled back: {} - {}",
                                rolled_back.version, rolled_back.description
                            );
                        }
                        Err(e) => {
                            println!("Rollback error: {}", e);
                            break;
                        }
                    }
                }
                "both" | _ => {
                    let result = self.run_coordinated_down().await?;
                    self.print_coordination_result(&result);
                }
            }
        }

        Ok(())
    }

    /// Run coordinated rollback on both databases.
    async fn run_coordinated_down(&self) -> Result<CoordinationResult> {
        let sqlite = Arc::new(SqliteDb::open(&self.sqlite_path)?);

        let postgres = if let Some(ref pg_url) = self.postgres_url {
            Some(
                sqlx::postgres::PgPoolOptions::new()
                    .max_connections(2)
                    .connect(pg_url)
                    .await?,
            )
        } else {
            None
        };

        let coordinator = if let Some(pool) = postgres {
            DualMigrationCoordinator::new(sqlite, pool)
        } else {
            DualMigrationCoordinator::new_sqlite_only(sqlite)
        };

        coordinator.run_down().await
    }

    /// Create a new migration file.
    async fn create_migration(&self, name: &str) -> Result<()> {
        let migrations_path = self.migrations_path();

        // Ensure migrations directory exists
        if !migrations_path.exists() {
            fs::create_dir_all(&migrations_path)?;
            println!("Created migrations directory: {:?}", migrations_path);
        }

        // Generate version number (timestamp)
        let version = Utc::now().format("%Y%m%d%H%M%S").to_string();
        let version_num: i64 = version.parse().unwrap();

        // Generate migration templates
        let (up_content, down_content) = generate_migration_template(name, version_num);

        // Create file paths
        let up_path = migrations_path.join(format!("{}_{}.up.sql", version, name));
        let down_path = migrations_path.join(format!("{}_{}.down.sql", version, name));

        // Write files
        fs::write(&up_path, up_content)?;
        fs::write(&down_path, down_content)?;

        println!("Created migration {}_{}", version, name);
        println!("  Up:   {:?}", up_path);
        println!("  Down: {:?}", down_path);
        println!("\nEdit these files to add your migration SQL.");

        Ok(())
    }

    /// Sync databases to consistent state.
    async fn sync_databases(&self) -> Result<()> {
        let Some(ref pg_url) = self.postgres_url else {
            println!("PostgreSQL not configured. Nothing to sync.");
            return Ok(());
        };

        println!("Synchronizing databases...\n");

        let sqlite = Arc::new(SqliteDb::open(&self.sqlite_path)?);
        let pool = sqlx::postgres::PgPoolOptions::new()
            .max_connections(2)
            .connect(pg_url)
            .await?;

        let coordinator = DualMigrationCoordinator::new(sqlite, pool);
        let result = coordinator.sync().await?;

        self.print_coordination_result(&result);

        Ok(())
    }

    /// Reset migration state (dangerous).
    async fn reset_migrations(&self, confirm: bool) -> Result<()> {
        if !confirm {
            println!("WARNING: This will reset all migration tracking!");
            println!("Data in tables will NOT be deleted, but migration history will be cleared.");
            println!("\nRun with --confirm to proceed.");
            return Ok(());
        }

        if !self.force {
            println!("Are you sure? This operation cannot be undone.");
            println!("Run with --force to proceed.");
            return Ok(());
        }

        println!("Resetting migration state...\n");

        // Reset SQLite
        let sqlite_db = SqliteDb::open(&self.sqlite_path)?;
        sqlite_db
            .execute_batch(
                "DROP TABLE IF EXISTS schema_version;
                 DROP TABLE IF EXISTS migration_lock;
                 DROP TABLE IF EXISTS migration_checkpoint;",
            )
            .await?;
        println!("SQLite migration state cleared.");

        // Reset PostgreSQL if configured
        if let Some(ref pg_url) = self.postgres_url {
            let pool = sqlx::postgres::PgPoolOptions::new()
                .max_connections(1)
                .connect(pg_url)
                .await?;

            sqlx::query(
                "DROP TABLE IF EXISTS schema_version;
                 DROP TABLE IF EXISTS migration_lock;
                 DROP TABLE IF EXISTS migration_checkpoint;",
            )
            .execute(&pool)
            .await?;

            println!("PostgreSQL migration state cleared.");
        }

        println!("\nMigration state has been reset. Run `migrate up` to re-apply migrations.");

        Ok(())
    }

    /// Migrate linear rewards to Bayesian quality scores.
    async fn migrate_rewards(&self, args: &RewardsArgs) -> Result<()> {
        use crate::reasoning_bank::pattern::BetaParams;

        let db = SqliteDb::open(&args.db_path)?;

        println!("Rewards Migration: Linear reward -> Bayesian quality scores");
        println!("{:-<60}", "");
        println!("  Database: {}", args.db_path);
        if args.dry_run {
            println!("  Mode: DRY RUN (no changes will be made)");
        }
        println!();

        // Ensure quality columns exist (idempotent ALTER TABLE)
        db.with_connection(|conn| {
            let _ = conn.execute_batch(
                "ALTER TABLE reasoning_patterns ADD COLUMN quality_alpha REAL DEFAULT 1.0;",
            );
            let _ = conn.execute_batch(
                "ALTER TABLE reasoning_patterns ADD COLUMN quality_beta REAL DEFAULT 1.0;",
            );
            Ok(())
        })
        .await?;

        // Query all patterns that still have the default Beta(1,1) prior
        // and have a non-zero reward value
        let candidates: Vec<(String, f64)> = db
            .with_connection(|conn| {
                let mut stmt = conn.prepare(
                    "SELECT id, reward FROM reasoning_patterns \
                     WHERE quality_alpha = 1.0 AND quality_beta = 1.0 AND reward != 0.0",
                )?;
                let rows = stmt
                    .query_map([], |row| {
                        Ok((row.get::<_, String>(0)?, row.get::<_, f64>(1)?))
                    })?
                    .collect::<std::result::Result<Vec<_>, _>>()?;
                Ok(rows)
            })
            .await?;

        if candidates.is_empty() {
            println!(
                "No patterns need migration \
                 (all already have Bayesian scores or zero reward)."
            );
            return Ok(());
        }

        println!("Found {} pattern(s) to migrate:\n", candidates.len());

        // Show a sample of what would change
        let preview_count = candidates.len().min(10);
        for (id, reward) in candidates.iter().take(preview_count) {
            let beta = BetaParams::from_reward(*reward as f32);
            println!(
                "  {} | reward={:.3} -> alpha={:.1}, beta={:.1} (mean={:.3})",
                &id[..id.len().min(12)],
                reward,
                beta.alpha(),
                beta.beta(),
                beta.mean(),
            );
        }
        if candidates.len() > preview_count {
            println!("  ... and {} more", candidates.len() - preview_count);
        }
        println!();

        if args.dry_run {
            println!("DRY RUN complete. Run without --dry-run to apply changes.");
            return Ok(());
        }

        // Apply the migration
        let mut migrated = 0usize;
        for (id, reward) in &candidates {
            let beta = BetaParams::from_reward(*reward as f32);
            db.execute(
                "UPDATE reasoning_patterns \
                 SET quality_alpha = ?, quality_beta = ? WHERE id = ?",
                &[
                    &beta.alpha() as &dyn rusqlite::ToSql,
                    &beta.beta(),
                    id,
                ],
            )
            .await?;
            migrated += 1;
        }

        println!(
            "Migrated {} pattern(s) from linear reward to Bayesian quality scores.",
            migrated
        );

        Ok(())
    }

    /// Re-embed patterns using the hash embedder.
    async fn migrate_embeddings(&self, args: &EmbeddingsArgs) -> Result<()> {
        use crate::ml::HashEmbedder;

        let db = SqliteDb::open(&args.db_path)?;

        println!("Embeddings Migration: Hash-based embedding generation");
        println!("{:-<60}", "");
        println!("  Database: {}", args.db_path);
        println!("  Missing only: {}", args.missing_only);
        if args.dry_run {
            println!("  Mode: DRY RUN (no changes will be made)");
        }
        println!();

        // Ensure embedding_method column exists (idempotent ALTER TABLE)
        db.with_connection(|conn| {
            let _ = conn.execute_batch(
                "ALTER TABLE reasoning_patterns ADD COLUMN embedding_method TEXT;",
            );
            Ok(())
        })
        .await?;

        // Query patterns based on --missing-only flag
        let query = if args.missing_only {
            "SELECT id, problem FROM reasoning_patterns \
             WHERE embedding IS NULL OR embedding = ''"
        } else {
            "SELECT id, problem FROM reasoning_patterns"
        };

        let patterns: Vec<(String, String)> = db
            .with_connection(|conn| {
                let mut stmt = conn.prepare(query)?;
                let rows = stmt
                    .query_map([], |row| {
                        Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
                    })?
                    .collect::<std::result::Result<Vec<_>, _>>()?;
                Ok(rows)
            })
            .await?;

        if patterns.is_empty() {
            println!("No patterns need embedding generation.");
            return Ok(());
        }

        println!("Found {} pattern(s) to embed.", patterns.len());

        if args.dry_run {
            // Show a preview
            let preview_count = patterns.len().min(10);
            println!("\nPreview:");
            for (id, problem) in patterns.iter().take(preview_count) {
                let preview: String = problem.chars().take(60).collect();
                println!(
                    "  {} | {}{}",
                    &id[..id.len().min(12)],
                    preview,
                    if problem.len() > 60 { "..." } else { "" }
                );
            }
            if patterns.len() > preview_count {
                println!("  ... and {} more", patterns.len() - preview_count);
            }
            println!("\nDRY RUN complete. Run without --dry-run to apply changes.");
            return Ok(());
        }

        // Generate and store embeddings
        let embedder = HashEmbedder::new();
        let mut embedded = 0usize;
        let mut errors = 0usize;

        for (id, problem) in &patterns {
            match embedder.embed(problem) {
                Ok(result) => {
                    let embedding_json =
                        serde_json::to_string(&result.embedding).unwrap_or_default();
                    let method = "hash".to_string();
                    match db
                        .execute(
                            "UPDATE reasoning_patterns SET embedding = ?, embedding_method = ? WHERE id = ?",
                            &[&embedding_json as &dyn rusqlite::ToSql, &method as &dyn rusqlite::ToSql, id],
                        )
                        .await
                    {
                        Ok(_) => embedded += 1,
                        Err(e) => {
                            eprintln!(
                                "  Failed to update {}: {}",
                                &id[..id.len().min(12)],
                                e
                            );
                            errors += 1;
                        }
                    }
                }
                Err(e) => {
                    eprintln!(
                        "  Failed to embed {}: {}",
                        &id[..id.len().min(12)],
                        e
                    );
                    errors += 1;
                }
            }

            // Progress indicator every 100 patterns
            if (embedded + errors) % 100 == 0 && (embedded + errors) > 0 {
                println!(
                    "  Progress: {}/{} patterns processed...",
                    embedded + errors,
                    patterns.len()
                );
            }
        }

        println!();
        println!("Embedding migration complete:");
        println!("  Embedded: {} pattern(s)", embedded);
        if errors > 0 {
            println!("  Errors:   {} pattern(s)", errors);
        }
        println!("  Method:   hash (SHAKE-256, 128-dim)");

        Ok(())
    }

    /// Print a coordination result.
    fn print_coordination_result(&self, result: &CoordinationResult) {
        println!("SQLite:");
        self.print_outcome("  ", &result.sqlite);

        if let Some(ref pg) = result.postgres {
            println!("\nPostgreSQL:");
            self.print_outcome("  ", pg);
        }

        if !result.warnings.is_empty() {
            println!("\nWarnings:");
            for warning in &result.warnings {
                println!("  - {}", warning);
            }
        }

        if result.consistent {
            println!("\nDatabases are consistent.");
        } else {
            println!("\nWARNING: Databases are NOT consistent!");
        }
    }

    /// Print a single migration outcome.
    fn print_outcome(&self, prefix: &str, outcome: &MigrationOutcome) {
        match outcome {
            MigrationOutcome::Success {
                applied,
                final_version,
            } => {
                println!("{}Applied {} migration(s)", prefix, applied.len());
                for m in applied {
                    println!(
                        "{}  {} - {} ({}ms)",
                        prefix,
                        m.version,
                        m.description,
                        m.execution_time_ms.unwrap_or(0)
                    );
                }
                if let Some(v) = final_version {
                    println!("{}Final version: {}", prefix, v);
                }
            }
            MigrationOutcome::NoChanges { current_version } => {
                println!("{}No changes needed", prefix);
                if let Some(v) = current_version {
                    println!("{}Current version: {}", prefix, v);
                }
            }
            MigrationOutcome::Failed {
                error,
                current_version,
            } => {
                println!("{}FAILED: {}", prefix, error);
                if let Some(v) = current_version {
                    println!("{}Current version: {}", prefix, v);
                }
            }
            MigrationOutcome::Skipped { reason } => {
                println!("{}Skipped: {}", prefix, reason);
            }
        }
    }

}

/// Mask password in a database URL for logging.
fn mask_password(url: &str) -> String {
    if let Some(at_pos) = url.find('@') {
        if let Some(colon_pos) = url[..at_pos].rfind(':') {
            let prefix = &url[..colon_pos + 1];
            let suffix = &url[at_pos..];
            return format!("{}****{}", prefix, suffix);
        }
    }
    url.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mask_password() {
        assert_eq!(
            mask_password("postgres://user:secret@localhost/db"),
            "postgres://user:****@localhost/db"
        );
        assert_eq!(
            mask_password("postgres://localhost/db"),
            "postgres://localhost/db"
        );
    }

    #[test]
    fn test_migrate_command_defaults() {
        let cmd = MigrateCommand {
            action: None,
            up: false,
            down: false,
            steps: 1,
            target: None,
            status: false,
            force: false,
            dry_run: false,
            sqlite_path: "test.db".to_string(),
            postgres_url: None,
            migrations_path: None,
        };
        assert_eq!(cmd.steps, 1);
        assert!(cmd.target.is_none());
    }

    #[test]
    fn test_rewards_args() {
        let args = RewardsArgs {
            db_path: "nagual.db".to_string(),
            dry_run: true,
        };
        assert_eq!(args.db_path, "nagual.db");
        assert!(args.dry_run);
    }

    #[test]
    fn test_embeddings_args() {
        let args = EmbeddingsArgs {
            db_path: "nagual.db".to_string(),
            missing_only: true,
            dry_run: false,
        };
        assert_eq!(args.db_path, "nagual.db");
        assert!(args.missing_only);
        assert!(!args.dry_run);
    }

    #[test]
    fn test_embeddings_args_defaults() {
        let args = EmbeddingsArgs {
            db_path: "nagual.db".to_string(),
            missing_only: false,
            dry_run: false,
        };
        assert!(!args.missing_only);
        assert!(!args.dry_run);
    }
}
