//! Migration runner implementation.
//!
//! Provides the core migration execution logic including:
//! - Migration discovery and loading
//! - Up/down execution with transaction support
//! - Checkpoint system for resumable migrations
//! - Status reporting

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sqlx::postgres::PgPool;
use sqlx::Row;

use crate::db::SqliteDb;
use crate::error::{MigrationError, Result};
use crate::migration::{
    calculate_checksum, migrations_dir, validate_migration_script, MigrationLock,
    POSTGRES_CHECKPOINT_TABLE, POSTGRES_SCHEMA_VERSION_TABLE, SQLITE_CHECKPOINT_TABLE,
    SQLITE_SCHEMA_VERSION_TABLE,
};

/// Represents a single migration with up and down scripts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Migration {
    /// Unique version number (timestamp-based recommended).
    pub version: i64,
    /// Human-readable name.
    pub name: String,
    /// SQL script for applying the migration.
    pub up_script: String,
    /// SQL script for rolling back the migration.
    pub down_script: Option<String>,
    /// SHA-256 checksum of the up script.
    pub checksum: String,
    /// Optional description.
    pub description: String,
}

impl Migration {
    /// Create a new migration from up and optional down scripts.
    pub fn new(version: i64, name: &str, up_script: &str, down_script: Option<&str>) -> Self {
        let checksum = calculate_checksum(up_script);
        Self {
            version,
            name: name.to_string(),
            up_script: up_script.to_string(),
            down_script: down_script.map(|s| s.to_string()),
            checksum,
            description: name.to_string(),
        }
    }

    /// Load a migration from file system.
    /// Expects files in format: {version}_{name}.up.sql and {version}_{name}.down.sql
    pub fn load_from_files(base_path: &Path) -> Result<Vec<Migration>> {
        let mut migrations = BTreeMap::new();

        if !base_path.exists() {
            return Ok(vec![]);
        }

        for entry in fs::read_dir(base_path).map_err(MigrationError::FileError)? {
            let entry = entry.map_err(MigrationError::FileError)?;
            let path = entry.path();

            if let Some(filename) = path.file_name().and_then(|n| n.to_str()) {
                // Parse filename: {version}_{name}.up.sql or {version}_{name}.down.sql
                if let Some(parsed) = parse_migration_filename(filename) {
                    let content = fs::read_to_string(&path).map_err(MigrationError::FileError)?;

                    let entry = migrations.entry(parsed.version).or_insert_with(|| {
                        (parsed.name.clone(), None, None)
                    });

                    match parsed.direction {
                        MigrationDirection::Up => entry.1 = Some(content),
                        MigrationDirection::Down => entry.2 = Some(content),
                    }
                }
            }
        }

        let mut result: Vec<Migration> = migrations
            .into_iter()
            .filter_map(|(version, (name, up, down))| {
                up.map(|up_script| Migration::new(version, &name, &up_script, down.as_deref()))
            })
            .collect();

        result.sort_by_key(|m| m.version);
        Ok(result)
    }
}

/// Parsed migration filename components.
struct ParsedMigrationFilename {
    version: i64,
    name: String,
    direction: MigrationDirection,
}

/// Direction of migration script.
enum MigrationDirection {
    Up,
    Down,
}

/// Parse a migration filename into components.
fn parse_migration_filename(filename: &str) -> Option<ParsedMigrationFilename> {
    // Expected format: {version}_{name}.up.sql or {version}_{name}.down.sql
    let direction = if filename.ends_with(".up.sql") {
        MigrationDirection::Up
    } else if filename.ends_with(".down.sql") {
        MigrationDirection::Down
    } else {
        return None;
    };

    let base = filename
        .trim_end_matches(".up.sql")
        .trim_end_matches(".down.sql");

    let parts: Vec<&str> = base.splitn(2, '_').collect();
    if parts.len() != 2 {
        return None;
    }

    let version = parts[0].parse::<i64>().ok()?;
    let name = parts[1].to_string();

    Some(ParsedMigrationFilename {
        version,
        name,
        direction,
    })
}

/// Schema version record stored in the database.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchemaVersion {
    /// Migration version number.
    pub version: i64,
    /// When the migration was applied.
    pub applied_at: DateTime<Utc>,
    /// Checksum of the migration script.
    pub checksum: String,
    /// Description of what this migration does.
    pub description: String,
    /// How long the migration took to execute.
    pub execution_time_ms: Option<i64>,
    /// If rolled back, when it was rolled back.
    pub rolled_back_at: Option<DateTime<Utc>>,
}

/// Checkpoint for resumable migrations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Checkpoint {
    /// Migration version being executed.
    pub version: i64,
    /// Current step index (0-based).
    pub step_index: i32,
    /// Total number of steps.
    pub total_steps: i32,
    /// Description of last completed step.
    pub last_completed_step: Option<String>,
    /// When checkpoint was created.
    pub created_at: DateTime<Utc>,
    /// When checkpoint was last updated.
    pub updated_at: DateTime<Utc>,
}

/// Migration status for a database.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationStatus {
    /// Current schema version (highest applied migration).
    pub current_version: Option<i64>,
    /// List of applied migrations.
    pub applied: Vec<SchemaVersion>,
    /// List of pending migrations (not yet applied).
    pub pending: Vec<Migration>,
    /// Any active checkpoint (interrupted migration).
    pub checkpoint: Option<Checkpoint>,
    /// Whether migrations are currently locked.
    pub is_locked: bool,
}

/// Migration runner for SQLite databases.
pub struct MigrationRunner {
    /// SQLite database connection.
    db: SqliteDb,
    /// Directory containing migration files.
    migrations_path: PathBuf,
    /// Whether we hold the migration lock.
    has_lock: bool,
}

impl MigrationRunner {
    /// Create a new migration runner.
    pub fn new(db: SqliteDb) -> Self {
        Self {
            db,
            migrations_path: migrations_dir(),
            has_lock: false,
        }
    }

    /// Create a migration runner with a custom migrations path.
    pub fn with_migrations_path(db: SqliteDb, path: impl Into<PathBuf>) -> Self {
        Self {
            db,
            migrations_path: path.into(),
            has_lock: false,
        }
    }

    /// Initialize the schema version table if it doesn't exist.
    pub async fn initialize(&self) -> Result<()> {
        self.db.execute_batch(SQLITE_SCHEMA_VERSION_TABLE).await?;
        self.db.execute_batch(SQLITE_CHECKPOINT_TABLE).await?;
        Ok(())
    }

    /// Acquire the migration lock.
    pub async fn acquire_lock(&mut self) -> Result<()> {
        self.db
            .with_connection(|conn| {
                MigrationLock::try_acquire_sqlite(conn)?;
                Ok(())
            })
            .await?;
        self.has_lock = true;
        Ok(())
    }

    /// Release the migration lock.
    pub async fn release_lock(&mut self) -> Result<()> {
        if self.has_lock {
            self.db
                .with_connection(|conn| {
                    MigrationLock::release_sqlite(conn)?;
                    Ok(())
                })
                .await?;
            self.has_lock = false;
        }
        Ok(())
    }

    /// Get the current migration status.
    pub async fn status(&self) -> Result<MigrationStatus> {
        // Ensure tables exist
        self.initialize().await?;

        // Get applied migrations
        let applied = self
            .db
            .query(
                "SELECT version, applied_at, checksum, description, execution_time_ms, rolled_back_at
                 FROM schema_version
                 WHERE rolled_back_at IS NULL
                 ORDER BY version",
                &[],
                |row| {
                    Ok(SchemaVersion {
                        version: row.get(0)?,
                        applied_at: DateTime::parse_from_rfc3339(&row.get::<_, String>(1)?)
                            .map(|dt| dt.with_timezone(&Utc))
                            .unwrap_or_else(|_| Utc::now()),
                        checksum: row.get(2)?,
                        description: row.get(3)?,
                        execution_time_ms: row.get(4).ok(),
                        rolled_back_at: row.get::<_, Option<String>>(5).ok().flatten().and_then(|s| {
                            DateTime::parse_from_rfc3339(&s)
                                .ok()
                                .map(|dt| dt.with_timezone(&Utc))
                        }),
                    })
                },
            )
            .await?;

        let current_version = applied.last().map(|m| m.version);

        // Load available migrations from files
        let available = Migration::load_from_files(&self.migrations_path)?;
        let applied_versions: std::collections::HashSet<i64> =
            applied.iter().map(|m| m.version).collect();
        let pending: Vec<Migration> = available
            .into_iter()
            .filter(|m| !applied_versions.contains(&m.version))
            .collect();

        // Check for active checkpoint
        let checkpoint = self
            .db
            .query_one(
                "SELECT version, step_index, total_steps, last_completed_step, created_at, updated_at
                 FROM migration_checkpoint
                 LIMIT 1",
                &[],
                |row| {
                    Ok(Checkpoint {
                        version: row.get(0)?,
                        step_index: row.get(1)?,
                        total_steps: row.get(2)?,
                        last_completed_step: row.get(3).ok(),
                        created_at: DateTime::parse_from_rfc3339(&row.get::<_, String>(4)?)
                            .map(|dt| dt.with_timezone(&Utc))
                            .unwrap_or_else(|_| Utc::now()),
                        updated_at: DateTime::parse_from_rfc3339(&row.get::<_, String>(5)?)
                            .map(|dt| dt.with_timezone(&Utc))
                            .unwrap_or_else(|_| Utc::now()),
                    })
                },
            )
            .await?;

        // Check lock status
        let is_locked = self
            .db
            .query_one(
                "SELECT 1 FROM migration_lock WHERE id = 1",
                &[],
                |_| Ok(true),
            )
            .await?
            .unwrap_or(false);

        Ok(MigrationStatus {
            current_version,
            applied,
            pending,
            checkpoint,
            is_locked,
        })
    }

    /// Execute all pending migrations.
    pub async fn run_up(&mut self) -> Result<Vec<SchemaVersion>> {
        self.initialize().await?;
        self.acquire_lock().await?;

        let result = self.run_up_internal().await;

        // Always release lock, even on error
        let _ = self.release_lock().await;

        result
    }

    /// Internal implementation of run_up.
    async fn run_up_internal(&self) -> Result<Vec<SchemaVersion>> {
        let status = self.status().await?;

        if status.pending.is_empty() {
            return Err(MigrationError::NoPendingMigrations.into());
        }

        let mut applied = Vec::new();

        for migration in status.pending {
            // Validate the migration script
            validate_migration_script(&migration.up_script, &migration.name)?;

            // Check if there's a checkpoint for this migration
            if let Some(ref checkpoint) = status.checkpoint {
                if checkpoint.version == migration.version {
                    tracing::info!(
                        "Resuming migration {} from step {}",
                        migration.version,
                        checkpoint.step_index
                    );
                    // For now, we don't support partial resume - clear checkpoint and restart
                    self.clear_checkpoint(migration.version).await?;
                }
            }

            let schema_version = self.apply_migration(&migration).await?;
            applied.push(schema_version);

            tracing::info!(
                "Applied migration {} ({})",
                migration.version,
                migration.name
            );
        }

        Ok(applied)
    }

    /// Apply a single migration.
    async fn apply_migration(&self, migration: &Migration) -> Result<SchemaVersion> {
        let start = Instant::now();

        // Execute migration in a transaction
        self.db
            .with_connection_mut(|conn| {
                let tx = conn.transaction().map_err(|e| {
                    MigrationError::ExecutionFailed {
                        version: migration.version,
                        message: e.to_string(),
                    }
                })?;

                // Execute migration script
                tx.execute_batch(&migration.up_script).map_err(|e| {
                    MigrationError::ExecutionFailed {
                        version: migration.version,
                        message: e.to_string(),
                    }
                })?;

                // Record in schema_version
                let now = Utc::now();
                let execution_time = start.elapsed().as_millis() as i64;

                tx.execute(
                    "INSERT INTO schema_version (version, applied_at, checksum, description, execution_time_ms)
                     VALUES (?, ?, ?, ?, ?)",
                    rusqlite::params![
                        migration.version,
                        now.to_rfc3339(),
                        &migration.checksum,
                        &migration.description,
                        execution_time,
                    ],
                ).map_err(|e| MigrationError::ExecutionFailed {
                    version: migration.version,
                    message: e.to_string(),
                })?;

                tx.commit().map_err(|e| {
                    MigrationError::ExecutionFailed {
                        version: migration.version,
                        message: e.to_string(),
                    }
                })?;

                Ok(SchemaVersion {
                    version: migration.version,
                    applied_at: now,
                    checksum: migration.checksum.clone(),
                    description: migration.description.clone(),
                    execution_time_ms: Some(execution_time),
                    rolled_back_at: None,
                })
            })
            .await
    }

    /// Rollback the last applied migration.
    pub async fn run_down(&mut self) -> Result<SchemaVersion> {
        self.initialize().await?;
        self.acquire_lock().await?;

        let result = self.run_down_internal().await;

        let _ = self.release_lock().await;

        result
    }

    /// Internal implementation of run_down.
    async fn run_down_internal(&self) -> Result<SchemaVersion> {
        let status = self.status().await?;

        let last_applied = status.applied.last().ok_or(MigrationError::NotFound {
            version: 0,
        })?;

        // Find the migration file to get the down script
        let migrations = Migration::load_from_files(&self.migrations_path)?;
        let migration = migrations
            .iter()
            .find(|m| m.version == last_applied.version)
            .ok_or(MigrationError::NotFound {
                version: last_applied.version,
            })?;

        let down_script = migration.down_script.as_ref().ok_or(MigrationError::RollbackFailed {
            version: migration.version,
            reason: "No down script available".to_string(),
        })?;

        // Verify checksum matches
        if migration.checksum != last_applied.checksum {
            return Err(MigrationError::ChecksumMismatch {
                version: migration.version,
                expected: last_applied.checksum.clone(),
                found: migration.checksum.clone(),
            }
            .into());
        }

        self.rollback_migration(migration, down_script).await
    }

    /// Rollback a single migration.
    async fn rollback_migration(
        &self,
        migration: &Migration,
        down_script: &str,
    ) -> Result<SchemaVersion> {
        let start = Instant::now();

        self.db
            .with_connection_mut(|conn| {
                let tx = conn.transaction().map_err(|e| MigrationError::RollbackFailed {
                    version: migration.version,
                    reason: e.to_string(),
                })?;

                // Execute down script
                tx.execute_batch(down_script).map_err(|e| MigrationError::RollbackFailed {
                    version: migration.version,
                    reason: e.to_string(),
                })?;

                // Mark as rolled back
                let now = Utc::now();
                tx.execute(
                    "UPDATE schema_version SET rolled_back_at = ? WHERE version = ?",
                    rusqlite::params![now.to_rfc3339(), migration.version],
                )
                .map_err(|e| MigrationError::RollbackFailed {
                    version: migration.version,
                    reason: e.to_string(),
                })?;

                tx.commit().map_err(|e| MigrationError::RollbackFailed {
                    version: migration.version,
                    reason: e.to_string(),
                })?;

                let execution_time = start.elapsed().as_millis() as i64;

                Ok(SchemaVersion {
                    version: migration.version,
                    applied_at: Utc::now(),
                    checksum: migration.checksum.clone(),
                    description: migration.description.clone(),
                    execution_time_ms: Some(execution_time),
                    rolled_back_at: Some(now),
                })
            })
            .await
    }

    /// Create a checkpoint for long-running migrations.
    pub async fn create_checkpoint(
        &self,
        version: i64,
        step_index: i32,
        total_steps: i32,
        step_description: Option<&str>,
    ) -> Result<Checkpoint> {
        let now = Utc::now();

        self.db
            .execute(
                "INSERT OR REPLACE INTO migration_checkpoint
                 (version, step_index, total_steps, last_completed_step, created_at, updated_at)
                 VALUES (?, ?, ?, ?, ?, ?)",
                &[
                    &version as &dyn rusqlite::ToSql,
                    &step_index,
                    &total_steps,
                    &step_description.map(|s| s.to_string()),
                    &now.to_rfc3339(),
                    &now.to_rfc3339(),
                ],
            )
            .await?;

        Ok(Checkpoint {
            version,
            step_index,
            total_steps,
            last_completed_step: step_description.map(|s| s.to_string()),
            created_at: now,
            updated_at: now,
        })
    }

    /// Update an existing checkpoint.
    pub async fn update_checkpoint(
        &self,
        version: i64,
        step_index: i32,
        step_description: Option<&str>,
    ) -> Result<()> {
        let now = Utc::now();

        self.db
            .execute(
                "UPDATE migration_checkpoint
                 SET step_index = ?, last_completed_step = ?, updated_at = ?
                 WHERE version = ?",
                &[
                    &step_index as &dyn rusqlite::ToSql,
                    &step_description.map(|s| s.to_string()),
                    &now.to_rfc3339(),
                    &version,
                ],
            )
            .await?;

        Ok(())
    }

    /// Clear a checkpoint after successful completion.
    pub async fn clear_checkpoint(&self, version: i64) -> Result<()> {
        self.db
            .execute(
                "DELETE FROM migration_checkpoint WHERE version = ?",
                &[&version as &dyn rusqlite::ToSql],
            )
            .await?;
        Ok(())
    }
}

/// Migration runner for PostgreSQL databases.
pub struct PostgresMigrationRunner {
    /// PostgreSQL connection pool.
    pool: PgPool,
    /// Directory containing migration files.
    migrations_path: PathBuf,
}

impl PostgresMigrationRunner {
    /// Create a new PostgreSQL migration runner.
    pub fn new(pool: PgPool) -> Self {
        Self {
            pool,
            migrations_path: migrations_dir(),
        }
    }

    /// Create with custom migrations path.
    pub fn with_migrations_path(pool: PgPool, path: impl Into<PathBuf>) -> Self {
        Self {
            pool,
            migrations_path: path.into(),
        }
    }

    /// Initialize the schema version table.
    pub async fn initialize(&self) -> Result<()> {
        sqlx::query(POSTGRES_SCHEMA_VERSION_TABLE)
            .execute(&self.pool)
            .await
            .map_err(|e| MigrationError::SqlError(e.to_string()))?;

        sqlx::query(POSTGRES_CHECKPOINT_TABLE)
            .execute(&self.pool)
            .await
            .map_err(|e| MigrationError::SqlError(e.to_string()))?;

        Ok(())
    }

    /// Try to acquire an advisory lock.
    pub async fn try_acquire_lock(&self) -> Result<bool> {
        let row = sqlx::query("SELECT pg_try_advisory_lock(12345678)")
            .fetch_one(&self.pool)
            .await
            .map_err(|e| MigrationError::LockFailed {
                reason: e.to_string(),
            })?;

        Ok(row.get::<bool, _>(0))
    }

    /// Release the advisory lock.
    pub async fn release_lock(&self) -> Result<()> {
        sqlx::query("SELECT pg_advisory_unlock(12345678)")
            .execute(&self.pool)
            .await
            .map_err(|e| MigrationError::LockFailed {
                reason: e.to_string(),
            })?;
        Ok(())
    }

    /// Get the current schema version.
    pub async fn current_version(&self) -> Result<Option<i64>> {
        self.initialize().await?;

        let row = sqlx::query(
            "SELECT version FROM schema_version
             WHERE rolled_back_at IS NULL
             ORDER BY version DESC
             LIMIT 1",
        )
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| MigrationError::SqlError(e.to_string()))?;

        Ok(row.map(|r| r.get::<i64, _>(0)))
    }

    /// Get full migration status.
    pub async fn status(&self) -> Result<MigrationStatus> {
        self.initialize().await?;

        let rows = sqlx::query(
            "SELECT version, applied_at, checksum, description, execution_time_ms, rolled_back_at
             FROM schema_version
             WHERE rolled_back_at IS NULL
             ORDER BY version",
        )
        .fetch_all(&self.pool)
        .await
        .map_err(|e| MigrationError::SqlError(e.to_string()))?;

        let applied: Vec<SchemaVersion> = rows
            .iter()
            .map(|row| SchemaVersion {
                version: row.get("version"),
                applied_at: row.get("applied_at"),
                checksum: row.get("checksum"),
                description: row.get("description"),
                execution_time_ms: row.get("execution_time_ms"),
                rolled_back_at: row.get("rolled_back_at"),
            })
            .collect();

        let current_version = applied.last().map(|m| m.version);

        let available = Migration::load_from_files(&self.migrations_path)?;
        let applied_versions: std::collections::HashSet<i64> =
            applied.iter().map(|m| m.version).collect();
        let pending: Vec<Migration> = available
            .into_iter()
            .filter(|m| !applied_versions.contains(&m.version))
            .collect();

        // Check for checkpoint
        let checkpoint_row = sqlx::query(
            "SELECT version, step_index, total_steps, last_completed_step, created_at, updated_at
             FROM migration_checkpoint
             LIMIT 1",
        )
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| MigrationError::SqlError(e.to_string()))?;

        let checkpoint = checkpoint_row.map(|row| Checkpoint {
            version: row.get("version"),
            step_index: row.get("step_index"),
            total_steps: row.get("total_steps"),
            last_completed_step: row.get("last_completed_step"),
            created_at: row.get("created_at"),
            updated_at: row.get("updated_at"),
        });

        // Check lock status (simplified - just check if we can acquire)
        let is_locked = !self.try_acquire_lock().await.unwrap_or(false);
        if !is_locked {
            let _ = self.release_lock().await;
        }

        Ok(MigrationStatus {
            current_version,
            applied,
            pending,
            checkpoint,
            is_locked,
        })
    }

    /// Execute all pending migrations.
    pub async fn run_up(&self) -> Result<Vec<SchemaVersion>> {
        self.initialize().await?;

        if !self.try_acquire_lock().await? {
            return Err(MigrationError::LockFailed {
                reason: "Could not acquire advisory lock".to_string(),
            }
            .into());
        }

        let result = self.run_up_internal().await;

        let _ = self.release_lock().await;

        result
    }

    async fn run_up_internal(&self) -> Result<Vec<SchemaVersion>> {
        let status = self.status().await?;

        if status.pending.is_empty() {
            return Err(MigrationError::NoPendingMigrations.into());
        }

        let mut applied = Vec::new();

        for migration in status.pending {
            validate_migration_script(&migration.up_script, &migration.name)?;

            let schema_version = self.apply_migration(&migration).await?;
            applied.push(schema_version);

            tracing::info!(
                "Applied PostgreSQL migration {} ({})",
                migration.version,
                migration.name
            );
        }

        Ok(applied)
    }

    async fn apply_migration(&self, migration: &Migration) -> Result<SchemaVersion> {
        let start = Instant::now();

        // Start transaction
        let mut tx = self
            .pool
            .begin()
            .await
            .map_err(|e| MigrationError::ExecutionFailed {
                version: migration.version,
                message: e.to_string(),
            })?;

        // Execute migration
        sqlx::query(&migration.up_script)
            .execute(&mut *tx)
            .await
            .map_err(|e| MigrationError::ExecutionFailed {
                version: migration.version,
                message: e.to_string(),
            })?;

        let execution_time = start.elapsed().as_millis() as i64;
        let now = Utc::now();

        // Record in schema_version
        sqlx::query(
            "INSERT INTO schema_version (version, applied_at, checksum, description, execution_time_ms)
             VALUES ($1, $2, $3, $4, $5)",
        )
        .bind(migration.version)
        .bind(now)
        .bind(&migration.checksum)
        .bind(&migration.description)
        .bind(execution_time)
        .execute(&mut *tx)
        .await
        .map_err(|e| MigrationError::ExecutionFailed {
            version: migration.version,
            message: e.to_string(),
        })?;

        tx.commit()
            .await
            .map_err(|e| MigrationError::ExecutionFailed {
                version: migration.version,
                message: e.to_string(),
            })?;

        Ok(SchemaVersion {
            version: migration.version,
            applied_at: now,
            checksum: migration.checksum.clone(),
            description: migration.description.clone(),
            execution_time_ms: Some(execution_time),
            rolled_back_at: None,
        })
    }

    /// Rollback the last migration.
    pub async fn run_down(&self) -> Result<SchemaVersion> {
        self.initialize().await?;

        if !self.try_acquire_lock().await? {
            return Err(MigrationError::LockFailed {
                reason: "Could not acquire advisory lock".to_string(),
            }
            .into());
        }

        let result = self.run_down_internal().await;

        let _ = self.release_lock().await;

        result
    }

    async fn run_down_internal(&self) -> Result<SchemaVersion> {
        let status = self.status().await?;

        let last_applied = status.applied.last().ok_or(MigrationError::NotFound {
            version: 0,
        })?;

        let migrations = Migration::load_from_files(&self.migrations_path)?;
        let migration = migrations
            .iter()
            .find(|m| m.version == last_applied.version)
            .ok_or(MigrationError::NotFound {
                version: last_applied.version,
            })?;

        let down_script = migration.down_script.as_ref().ok_or(MigrationError::RollbackFailed {
            version: migration.version,
            reason: "No down script available".to_string(),
        })?;

        if migration.checksum != last_applied.checksum {
            return Err(MigrationError::ChecksumMismatch {
                version: migration.version,
                expected: last_applied.checksum.clone(),
                found: migration.checksum.clone(),
            }
            .into());
        }

        let start = Instant::now();

        let mut tx = self
            .pool
            .begin()
            .await
            .map_err(|e| MigrationError::RollbackFailed {
                version: migration.version,
                reason: e.to_string(),
            })?;

        sqlx::query(down_script)
            .execute(&mut *tx)
            .await
            .map_err(|e| MigrationError::RollbackFailed {
                version: migration.version,
                reason: e.to_string(),
            })?;

        let now = Utc::now();

        sqlx::query("UPDATE schema_version SET rolled_back_at = $1 WHERE version = $2")
            .bind(now)
            .bind(migration.version)
            .execute(&mut *tx)
            .await
            .map_err(|e| MigrationError::RollbackFailed {
                version: migration.version,
                reason: e.to_string(),
            })?;

        tx.commit()
            .await
            .map_err(|e| MigrationError::RollbackFailed {
                version: migration.version,
                reason: e.to_string(),
            })?;

        Ok(SchemaVersion {
            version: migration.version,
            applied_at: Utc::now(),
            checksum: migration.checksum.clone(),
            description: migration.description.clone(),
            execution_time_ms: Some(start.elapsed().as_millis() as i64),
            rolled_back_at: Some(now),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_migration_filename_up() {
        let parsed = parse_migration_filename("20240101120000_create_users.up.sql").unwrap();
        assert_eq!(parsed.version, 20240101120000);
        assert_eq!(parsed.name, "create_users");
        assert!(matches!(parsed.direction, MigrationDirection::Up));
    }

    #[test]
    fn test_parse_migration_filename_down() {
        let parsed = parse_migration_filename("20240101120000_create_users.down.sql").unwrap();
        assert_eq!(parsed.version, 20240101120000);
        assert_eq!(parsed.name, "create_users");
        assert!(matches!(parsed.direction, MigrationDirection::Down));
    }

    #[test]
    fn test_parse_migration_filename_invalid() {
        assert!(parse_migration_filename("invalid.sql").is_none());
        assert!(parse_migration_filename("no_version.up.sql").is_none());
    }

    #[test]
    fn test_migration_new() {
        let m = Migration::new(1, "test", "CREATE TABLE test (id INT);", None);
        assert_eq!(m.version, 1);
        assert_eq!(m.name, "test");
        assert!(!m.checksum.is_empty());
    }
}
