//! Dual-database migration coordinator.
//!
//! Coordinates migrations across SQLite (local) and PostgreSQL (cloud) databases,
//! handling partial failure scenarios and maintaining consistency.

use std::path::PathBuf;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use sqlx::postgres::PgPool;

use crate::db::SqliteDb;
use crate::error::{CoordinationError, Result};
use crate::migration::runner::{MigrationRunner, MigrationStatus, PostgresMigrationRunner, SchemaVersion};
use crate::migration::migrations_dir;

/// Result of a coordinated migration operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoordinationResult {
    /// Outcome for SQLite database.
    pub sqlite: MigrationOutcome,
    /// Outcome for PostgreSQL database (if configured).
    pub postgres: Option<MigrationOutcome>,
    /// Whether both databases are now consistent.
    pub consistent: bool,
    /// Any warnings or notes about the operation.
    pub warnings: Vec<String>,
}

/// Outcome of a migration operation on a single database.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MigrationOutcome {
    /// Migration(s) applied successfully.
    Success {
        applied: Vec<SchemaVersion>,
        final_version: Option<i64>,
    },
    /// No changes were needed.
    NoChanges {
        current_version: Option<i64>,
    },
    /// Migration failed.
    Failed {
        error: String,
        current_version: Option<i64>,
    },
    /// Migration was skipped (e.g., database not configured).
    Skipped {
        reason: String,
    },
}

impl MigrationOutcome {
    /// Check if the outcome represents success.
    pub fn is_success(&self) -> bool {
        matches!(self, MigrationOutcome::Success { .. } | MigrationOutcome::NoChanges { .. })
    }

    /// Get the final version after this outcome.
    pub fn final_version(&self) -> Option<i64> {
        match self {
            MigrationOutcome::Success { final_version, .. } => *final_version,
            MigrationOutcome::NoChanges { current_version } => *current_version,
            MigrationOutcome::Failed { current_version, .. } => *current_version,
            MigrationOutcome::Skipped { .. } => None,
        }
    }
}

/// Strategy for handling partial failures during dual-database migration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FailureStrategy {
    /// Stop immediately on first failure, don't rollback successful operations.
    StopOnFirstFailure,
    /// Try to rollback successful operations if one fails (best-effort).
    RollbackOnFailure,
    /// Continue with remaining operations even if one fails.
    ContinueOnFailure,
    /// SQLite is primary - only apply to PostgreSQL if SQLite succeeds.
    SqlitePrimary,
    /// PostgreSQL is primary - only apply to SQLite if PostgreSQL succeeds.
    PostgresPrimary,
}

/// Configuration for the dual migration coordinator.
#[derive(Debug, Clone)]
pub struct CoordinatorConfig {
    /// How to handle partial failures.
    pub failure_strategy: FailureStrategy,
    /// Directory containing migration files.
    pub migrations_path: PathBuf,
    /// Whether to verify consistency after operations.
    pub verify_consistency: bool,
    /// Maximum time to wait for locks (in seconds).
    pub lock_timeout_secs: u64,
}

impl Default for CoordinatorConfig {
    fn default() -> Self {
        Self {
            failure_strategy: FailureStrategy::SqlitePrimary,
            migrations_path: migrations_dir(),
            verify_consistency: true,
            lock_timeout_secs: 30,
        }
    }
}

/// Coordinates migrations across SQLite and PostgreSQL databases.
pub struct DualMigrationCoordinator {
    /// SQLite database.
    sqlite: Arc<SqliteDb>,
    /// PostgreSQL connection pool (optional).
    postgres: Option<PgPool>,
    /// Coordinator configuration.
    config: CoordinatorConfig,
}

impl DualMigrationCoordinator {
    /// Create a new coordinator with SQLite only.
    pub fn new_sqlite_only(sqlite: Arc<SqliteDb>) -> Self {
        Self {
            sqlite,
            postgres: None,
            config: CoordinatorConfig::default(),
        }
    }

    /// Create a new coordinator with both databases.
    pub fn new(sqlite: Arc<SqliteDb>, postgres: PgPool) -> Self {
        Self {
            sqlite,
            postgres: Some(postgres),
            config: CoordinatorConfig::default(),
        }
    }

    /// Create with custom configuration.
    pub fn with_config(
        sqlite: Arc<SqliteDb>,
        postgres: Option<PgPool>,
        config: CoordinatorConfig,
    ) -> Self {
        Self {
            sqlite,
            postgres,
            config,
        }
    }

    /// Get the current status of both databases.
    pub async fn status(&self) -> Result<(MigrationStatus, Option<MigrationStatus>)> {
        // Get SQLite status
        let sqlite_db = SqliteDb::open(self.sqlite.path())?;
        let sqlite_runner = MigrationRunner::with_migrations_path(
            sqlite_db,
            &self.config.migrations_path,
        );
        let sqlite_status = sqlite_runner.status().await?;

        // Get PostgreSQL status if available
        let postgres_status = if let Some(ref pool) = self.postgres {
            let pg_runner = PostgresMigrationRunner::with_migrations_path(
                pool.clone(),
                &self.config.migrations_path,
            );
            Some(pg_runner.status().await?)
        } else {
            None
        };

        Ok((sqlite_status, postgres_status))
    }

    /// Check if both databases are at the same version.
    pub async fn is_consistent(&self) -> Result<bool> {
        let (sqlite_status, postgres_status) = self.status().await?;

        match postgres_status {
            Some(pg_status) => {
                Ok(sqlite_status.current_version == pg_status.current_version)
            }
            None => Ok(true), // SQLite-only mode is always "consistent"
        }
    }

    /// Get version difference between databases.
    pub async fn version_diff(&self) -> Result<Option<(i64, i64)>> {
        let (sqlite_status, postgres_status) = self.status().await?;

        match postgres_status {
            Some(pg_status) => {
                let sqlite_v = sqlite_status.current_version.unwrap_or(0);
                let postgres_v = pg_status.current_version.unwrap_or(0);
                if sqlite_v != postgres_v {
                    Ok(Some((sqlite_v, postgres_v)))
                } else {
                    Ok(None)
                }
            }
            None => Ok(None),
        }
    }

    /// Run all pending migrations on both databases.
    pub async fn run_up(&self) -> Result<CoordinationResult> {
        match self.config.failure_strategy {
            FailureStrategy::SqlitePrimary => self.run_up_sqlite_primary().await,
            FailureStrategy::PostgresPrimary => self.run_up_postgres_primary().await,
            FailureStrategy::StopOnFirstFailure => self.run_up_stop_on_failure().await,
            FailureStrategy::ContinueOnFailure => self.run_up_continue_on_failure().await,
            FailureStrategy::RollbackOnFailure => self.run_up_with_rollback().await,
        }
    }

    /// Run migrations with SQLite as primary (default strategy).
    async fn run_up_sqlite_primary(&self) -> Result<CoordinationResult> {
        let mut warnings = Vec::new();

        // First, apply to SQLite
        let sqlite_outcome = self.run_sqlite_up().await;

        // Only proceed to PostgreSQL if SQLite succeeded
        let postgres_outcome = match &sqlite_outcome {
            MigrationOutcome::Success { .. } | MigrationOutcome::NoChanges { .. } => {
                if self.postgres.is_some() {
                    let outcome = self.run_postgres_up().await;
                    if !outcome.is_success() {
                        warnings.push(
                            "PostgreSQL migration failed after SQLite succeeded. Databases may be inconsistent.".to_string()
                        );
                    }
                    Some(outcome)
                } else {
                    Some(MigrationOutcome::Skipped {
                        reason: "PostgreSQL not configured".to_string(),
                    })
                }
            }
            MigrationOutcome::Failed { .. } => {
                warnings.push("Skipping PostgreSQL migration due to SQLite failure".to_string());
                Some(MigrationOutcome::Skipped {
                    reason: "SQLite migration failed".to_string(),
                })
            }
            MigrationOutcome::Skipped { .. } => None,
        };

        let consistent = self.check_consistency(&sqlite_outcome, &postgres_outcome);

        Ok(CoordinationResult {
            sqlite: sqlite_outcome,
            postgres: postgres_outcome,
            consistent,
            warnings,
        })
    }

    /// Run migrations with PostgreSQL as primary.
    async fn run_up_postgres_primary(&self) -> Result<CoordinationResult> {
        let mut warnings = Vec::new();

        // First, apply to PostgreSQL
        let postgres_outcome = if self.postgres.is_some() {
            self.run_postgres_up().await
        } else {
            return Err(CoordinationError::PostgresFailed {
                message: "PostgreSQL not configured but set as primary".to_string(),
            }
            .into());
        };

        // Only proceed to SQLite if PostgreSQL succeeded
        let sqlite_outcome = match &postgres_outcome {
            MigrationOutcome::Success { .. } | MigrationOutcome::NoChanges { .. } => {
                let outcome = self.run_sqlite_up().await;
                if !outcome.is_success() {
                    warnings.push(
                        "SQLite migration failed after PostgreSQL succeeded. Databases may be inconsistent.".to_string()
                    );
                }
                outcome
            }
            MigrationOutcome::Failed { .. } => {
                warnings.push("Skipping SQLite migration due to PostgreSQL failure".to_string());
                MigrationOutcome::Skipped {
                    reason: "PostgreSQL migration failed".to_string(),
                }
            }
            MigrationOutcome::Skipped { .. } => MigrationOutcome::Skipped {
                reason: "PostgreSQL was skipped".to_string(),
            },
        };

        let consistent = self.check_consistency(&sqlite_outcome, &Some(postgres_outcome.clone()));

        Ok(CoordinationResult {
            sqlite: sqlite_outcome,
            postgres: Some(postgres_outcome),
            consistent,
            warnings,
        })
    }

    /// Run migrations, stopping on first failure.
    async fn run_up_stop_on_failure(&self) -> Result<CoordinationResult> {
        let mut warnings = Vec::new();

        // Try SQLite first
        let sqlite_outcome = self.run_sqlite_up().await;
        if let MigrationOutcome::Failed { ref error, .. } = sqlite_outcome {
            let error_msg = error.clone();
            return Ok(CoordinationResult {
                sqlite: sqlite_outcome,
                postgres: Some(MigrationOutcome::Skipped {
                    reason: format!("Stopped due to SQLite failure: {}", error_msg),
                }),
                consistent: false,
                warnings,
            });
        }

        // Try PostgreSQL
        let postgres_outcome = if self.postgres.is_some() {
            let outcome = self.run_postgres_up().await;
            if let MigrationOutcome::Failed { .. } = &outcome {
                warnings.push("PostgreSQL failed. Databases are inconsistent.".to_string());
            }
            Some(outcome)
        } else {
            Some(MigrationOutcome::Skipped {
                reason: "PostgreSQL not configured".to_string(),
            })
        };

        let consistent = self.check_consistency(&sqlite_outcome, &postgres_outcome);

        Ok(CoordinationResult {
            sqlite: sqlite_outcome,
            postgres: postgres_outcome,
            consistent,
            warnings,
        })
    }

    /// Run migrations on both databases, continuing even on failure.
    async fn run_up_continue_on_failure(&self) -> Result<CoordinationResult> {
        let mut warnings = Vec::new();

        // Run both independently
        let sqlite_outcome = self.run_sqlite_up().await;
        if !sqlite_outcome.is_success() {
            warnings.push("SQLite migration failed".to_string());
        }

        let postgres_outcome = if self.postgres.is_some() {
            let outcome = self.run_postgres_up().await;
            if !outcome.is_success() {
                warnings.push("PostgreSQL migration failed".to_string());
            }
            Some(outcome)
        } else {
            Some(MigrationOutcome::Skipped {
                reason: "PostgreSQL not configured".to_string(),
            })
        };

        let consistent = self.check_consistency(&sqlite_outcome, &postgres_outcome);
        if !consistent {
            warnings.push("Databases are in inconsistent state".to_string());
        }

        Ok(CoordinationResult {
            sqlite: sqlite_outcome,
            postgres: postgres_outcome,
            consistent,
            warnings,
        })
    }

    /// Run migrations with rollback on failure (best-effort).
    async fn run_up_with_rollback(&self) -> Result<CoordinationResult> {
        let mut warnings = Vec::new();

        // Get initial versions
        let (sqlite_status, postgres_status) = self.status().await?;
        let initial_sqlite_version = sqlite_status.current_version;
        let _initial_postgres_version = postgres_status.as_ref().and_then(|s| s.current_version);

        // Run SQLite
        let sqlite_outcome = self.run_sqlite_up().await;

        // Run PostgreSQL
        let postgres_outcome = if self.postgres.is_some() {
            let outcome = self.run_postgres_up().await;

            // If PostgreSQL failed but SQLite succeeded, try to rollback SQLite
            if !outcome.is_success() && sqlite_outcome.is_success() {
                warnings.push("PostgreSQL failed. Attempting SQLite rollback.".to_string());

                if let Err(e) = self.rollback_sqlite_to(initial_sqlite_version).await {
                    warnings.push(format!("SQLite rollback failed: {}", e));
                } else {
                    warnings.push("SQLite rolled back successfully".to_string());
                }
            }

            Some(outcome)
        } else {
            Some(MigrationOutcome::Skipped {
                reason: "PostgreSQL not configured".to_string(),
            })
        };

        let consistent = self.check_consistency(&sqlite_outcome, &postgres_outcome);

        Ok(CoordinationResult {
            sqlite: sqlite_outcome,
            postgres: postgres_outcome,
            consistent,
            warnings,
        })
    }

    /// Rollback the last migration on both databases.
    pub async fn run_down(&self) -> Result<CoordinationResult> {
        let mut warnings = Vec::new();

        // Rollback SQLite
        let sqlite_outcome = self.run_sqlite_down().await;
        if !sqlite_outcome.is_success() {
            warnings.push("SQLite rollback issue".to_string());
        }

        // Rollback PostgreSQL
        let postgres_outcome = if self.postgres.is_some() {
            let outcome = self.run_postgres_down().await;
            if !outcome.is_success() {
                warnings.push("PostgreSQL rollback issue".to_string());
            }
            Some(outcome)
        } else {
            Some(MigrationOutcome::Skipped {
                reason: "PostgreSQL not configured".to_string(),
            })
        };

        let consistent = self.check_consistency(&sqlite_outcome, &postgres_outcome);

        Ok(CoordinationResult {
            sqlite: sqlite_outcome,
            postgres: postgres_outcome,
            consistent,
            warnings,
        })
    }

    /// Sync databases to a consistent state.
    /// This will bring the lagging database up to match the other.
    pub async fn sync(&self) -> Result<CoordinationResult> {
        let (sqlite_status, postgres_status) = self.status().await?;

        let Some(pg_status) = postgres_status else {
            // SQLite-only mode - nothing to sync
            return Ok(CoordinationResult {
                sqlite: MigrationOutcome::NoChanges {
                    current_version: sqlite_status.current_version,
                },
                postgres: Some(MigrationOutcome::Skipped {
                    reason: "PostgreSQL not configured".to_string(),
                }),
                consistent: true,
                warnings: vec![],
            });
        };

        let sqlite_v = sqlite_status.current_version.unwrap_or(0);
        let postgres_v = pg_status.current_version.unwrap_or(0);

        if sqlite_v == postgres_v {
            return Ok(CoordinationResult {
                sqlite: MigrationOutcome::NoChanges {
                    current_version: Some(sqlite_v),
                },
                postgres: Some(MigrationOutcome::NoChanges {
                    current_version: Some(postgres_v),
                }),
                consistent: true,
                warnings: vec![],
            });
        }

        let mut warnings = Vec::new();

        // Determine which database is behind and needs catching up
        if sqlite_v > postgres_v {
            warnings.push(format!(
                "PostgreSQL is behind (v{}) - syncing to SQLite version (v{})",
                postgres_v, sqlite_v
            ));
            let postgres_outcome = self.run_postgres_up().await;
            Ok(CoordinationResult {
                sqlite: MigrationOutcome::NoChanges {
                    current_version: Some(sqlite_v),
                },
                postgres: Some(postgres_outcome),
                consistent: self
                    .check_consistency(
                        &MigrationOutcome::NoChanges {
                            current_version: Some(sqlite_v),
                        },
                        &Some(MigrationOutcome::Success {
                            applied: vec![],
                            final_version: Some(sqlite_v),
                        }),
                    ),
                warnings,
            })
        } else {
            warnings.push(format!(
                "SQLite is behind (v{}) - syncing to PostgreSQL version (v{})",
                sqlite_v, postgres_v
            ));
            let sqlite_outcome = self.run_sqlite_up().await;
            Ok(CoordinationResult {
                sqlite: sqlite_outcome,
                postgres: Some(MigrationOutcome::NoChanges {
                    current_version: Some(postgres_v),
                }),
                consistent: true,
                warnings,
            })
        }
    }

    // Helper methods for individual database operations

    async fn run_sqlite_up(&self) -> MigrationOutcome {
        let db = match SqliteDb::open(self.sqlite.path()) {
            Ok(db) => db,
            Err(e) => {
                return MigrationOutcome::Failed {
                    error: e.to_string(),
                    current_version: None,
                }
            }
        };

        let mut runner =
            MigrationRunner::with_migrations_path(db, &self.config.migrations_path);

        match runner.run_up().await {
            Ok(applied) => {
                let final_version = applied.last().map(|m| m.version);
                MigrationOutcome::Success {
                    applied,
                    final_version,
                }
            }
            Err(e) => {
                // Check if it's "no pending migrations" which is actually fine
                if e.to_string().contains("No pending migrations") {
                    let status = runner.status().await.ok();
                    MigrationOutcome::NoChanges {
                        current_version: status.and_then(|s| s.current_version),
                    }
                } else {
                    let status = runner.status().await.ok();
                    MigrationOutcome::Failed {
                        error: e.to_string(),
                        current_version: status.and_then(|s| s.current_version),
                    }
                }
            }
        }
    }

    async fn run_sqlite_down(&self) -> MigrationOutcome {
        let db = match SqliteDb::open(self.sqlite.path()) {
            Ok(db) => db,
            Err(e) => {
                return MigrationOutcome::Failed {
                    error: e.to_string(),
                    current_version: None,
                }
            }
        };

        let mut runner =
            MigrationRunner::with_migrations_path(db, &self.config.migrations_path);

        match runner.run_down().await {
            Ok(rolled_back) => MigrationOutcome::Success {
                applied: vec![rolled_back.clone()],
                final_version: Some(rolled_back.version - 1).filter(|&v| v > 0),
            },
            Err(e) => {
                let status = runner.status().await.ok();
                MigrationOutcome::Failed {
                    error: e.to_string(),
                    current_version: status.and_then(|s| s.current_version),
                }
            }
        }
    }

    async fn run_postgres_up(&self) -> MigrationOutcome {
        let Some(ref pool) = self.postgres else {
            return MigrationOutcome::Skipped {
                reason: "PostgreSQL not configured".to_string(),
            };
        };

        let runner =
            PostgresMigrationRunner::with_migrations_path(pool.clone(), &self.config.migrations_path);

        match runner.run_up().await {
            Ok(applied) => {
                let final_version = applied.last().map(|m| m.version);
                MigrationOutcome::Success {
                    applied,
                    final_version,
                }
            }
            Err(e) => {
                if e.to_string().contains("No pending migrations") {
                    let status = runner.status().await.ok();
                    MigrationOutcome::NoChanges {
                        current_version: status.and_then(|s| s.current_version),
                    }
                } else {
                    let status = runner.status().await.ok();
                    MigrationOutcome::Failed {
                        error: e.to_string(),
                        current_version: status.and_then(|s| s.current_version),
                    }
                }
            }
        }
    }

    async fn run_postgres_down(&self) -> MigrationOutcome {
        let Some(ref pool) = self.postgres else {
            return MigrationOutcome::Skipped {
                reason: "PostgreSQL not configured".to_string(),
            };
        };

        let runner =
            PostgresMigrationRunner::with_migrations_path(pool.clone(), &self.config.migrations_path);

        match runner.run_down().await {
            Ok(rolled_back) => MigrationOutcome::Success {
                applied: vec![rolled_back.clone()],
                final_version: Some(rolled_back.version - 1).filter(|&v| v > 0),
            },
            Err(e) => {
                let status = runner.status().await.ok();
                MigrationOutcome::Failed {
                    error: e.to_string(),
                    current_version: status.and_then(|s| s.current_version),
                }
            }
        }
    }

    async fn rollback_sqlite_to(&self, target_version: Option<i64>) -> Result<()> {
        let db = SqliteDb::open(self.sqlite.path())?;
        let mut runner =
            MigrationRunner::with_migrations_path(db, &self.config.migrations_path);

        let target = target_version.unwrap_or(0);

        loop {
            let status = runner.status().await?;
            let current = status.current_version.unwrap_or(0);

            if current <= target {
                break;
            }

            runner.run_down().await?;
        }

        Ok(())
    }

    fn check_consistency(
        &self,
        sqlite: &MigrationOutcome,
        postgres: &Option<MigrationOutcome>,
    ) -> bool {
        match postgres {
            Some(pg) => {
                let sqlite_v = sqlite.final_version();
                let postgres_v = pg.final_version();

                match (sqlite_v, postgres_v) {
                    (Some(sv), Some(pv)) => sv == pv,
                    (None, None) => true,
                    _ => {
                        // One is None (skipped/failed), the other has a version
                        // This is only consistent if postgres was skipped
                        matches!(pg, MigrationOutcome::Skipped { .. })
                    }
                }
            }
            None => true, // No postgres = always consistent
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_migration_outcome_is_success() {
        assert!(MigrationOutcome::Success {
            applied: vec![],
            final_version: Some(1)
        }
        .is_success());

        assert!(MigrationOutcome::NoChanges {
            current_version: Some(1)
        }
        .is_success());

        assert!(!MigrationOutcome::Failed {
            error: "test".to_string(),
            current_version: None
        }
        .is_success());

        assert!(!MigrationOutcome::Skipped {
            reason: "test".to_string()
        }
        .is_success());
    }

    #[test]
    fn test_coordinator_config_default() {
        let config = CoordinatorConfig::default();
        assert_eq!(config.failure_strategy, FailureStrategy::SqlitePrimary);
        assert!(config.verify_consistency);
    }
}
