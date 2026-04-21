//! Plan persistence and storage
//!
//! Supports both in-memory storage (for tests) and SQLite persistence (for production).

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use tracing::{debug, info, warn};

use super::types::*;
use crate::db::SqliteDb;
use crate::error::NagualError;

/// Storage backend for plans
enum StorageBackend {
    /// In-memory storage (for tests)
    Memory(RwLock<HashMap<String, Plan>>),
    /// SQLite-backed storage (for production)
    Sqlite(Arc<SqliteDb>),
}

/// Storage for GOAP plans
pub struct PlanStorage {
    backend: StorageBackend,
}

impl PlanStorage {
    /// Create an in-memory storage (for tests)
    pub fn in_memory() -> Self {
        Self {
            backend: StorageBackend::Memory(RwLock::new(HashMap::new())),
        }
    }

    /// Create SQLite-backed storage
    pub async fn sqlite(db: Arc<SqliteDb>) -> Result<Self, NagualError> {
        // Initialize schema
        Self::init_schema(&db).await?;
        Ok(Self {
            backend: StorageBackend::Sqlite(db),
        })
    }

    /// Initialize the database schema for plans
    async fn init_schema(db: &SqliteDb) -> Result<(), NagualError> {
        db.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS plans (
                id TEXT PRIMARY KEY,
                goal_json TEXT NOT NULL,
                actions_json TEXT NOT NULL,
                total_cost REAL NOT NULL,
                estimated_duration_seconds INTEGER,
                status TEXT NOT NULL DEFAULT 'ready',
                current_step INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS plan_steps (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                plan_id TEXT NOT NULL,
                step_number INTEGER NOT NULL,
                action_id TEXT NOT NULL,
                action_name TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                result_json TEXT,
                started_at TEXT,
                completed_at TEXT,
                UNIQUE(plan_id, step_number)
            );

            CREATE INDEX IF NOT EXISTS idx_plans_status ON plans(status);
            CREATE INDEX IF NOT EXISTS idx_plan_steps_plan_id ON plan_steps(plan_id);
            "#,
        )
        .await?;

        debug!("Plan storage schema initialized");
        Ok(())
    }

    /// Save a plan
    pub async fn save(&self, plan: &Plan) -> Result<(), NagualError> {
        match &self.backend {
            StorageBackend::Memory(plans) => {
                let mut plans = plans.write().map_err(|_| NagualError::Internal {
                    message: "Failed to acquire write lock".into(),
                })?;
                plans.insert(plan.id.clone(), plan.clone());
                debug!("Plan {} saved to memory", plan.id);
                Ok(())
            }
            StorageBackend::Sqlite(db) => {
                let goal_json = serde_json::to_string(&plan.goal).map_err(|e| {
                    NagualError::Internal {
                        message: format!("Failed to serialize goal: {}", e),
                    }
                })?;

                let actions_json = serde_json::to_string(&plan.actions).map_err(|e| {
                    NagualError::Internal {
                        message: format!("Failed to serialize actions: {}", e),
                    }
                })?;

                let status_str = format!("{}", plan.status);
                let created_at_str = plan.created_at.to_rfc3339();
                let estimated_duration: Option<i64> =
                    plan.estimated_duration_seconds.map(|d| d as i64);

                // Upsert the plan
                db.execute(
                    r#"
                    INSERT INTO plans (id, goal_json, actions_json, total_cost, estimated_duration_seconds,
                                       status, current_step, created_at, updated_at)
                    VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, CURRENT_TIMESTAMP)
                    ON CONFLICT(id) DO UPDATE SET
                        goal_json = excluded.goal_json,
                        actions_json = excluded.actions_json,
                        total_cost = excluded.total_cost,
                        estimated_duration_seconds = excluded.estimated_duration_seconds,
                        status = excluded.status,
                        current_step = excluded.current_step,
                        updated_at = CURRENT_TIMESTAMP
                    "#,
                    &[
                        &plan.id as &dyn rusqlite::ToSql,
                        &goal_json,
                        &actions_json,
                        &plan.total_cost,
                        &estimated_duration,
                        &status_str,
                        &(plan.current_step as i64),
                        &created_at_str,
                    ],
                )
                .await?;

                debug!("Plan {} saved to SQLite", plan.id);
                Ok(())
            }
        }
    }

    /// Load a plan by ID
    pub async fn load(&self, plan_id: &str) -> Result<Option<Plan>, NagualError> {
        match &self.backend {
            StorageBackend::Memory(plans) => {
                let plans = plans.read().map_err(|_| NagualError::Internal {
                    message: "Failed to acquire read lock".into(),
                })?;
                Ok(plans.get(plan_id).cloned())
            }
            StorageBackend::Sqlite(db) => {
                let result = db
                    .query_one(
                        r#"
                    SELECT id, goal_json, actions_json, total_cost, estimated_duration_seconds,
                           status, current_step, created_at
                    FROM plans WHERE id = ?1
                    "#,
                        &[&plan_id as &dyn rusqlite::ToSql],
                        |row| {
                            Ok((
                                row.get::<_, String>(0)?,
                                row.get::<_, String>(1)?,
                                row.get::<_, String>(2)?,
                                row.get::<_, f64>(3)?,
                                row.get::<_, Option<i64>>(4)?,
                                row.get::<_, String>(5)?,
                                row.get::<_, i64>(6)?,
                                row.get::<_, String>(7)?,
                            ))
                        },
                    )
                    .await?;

                match result {
                    Some((
                        id,
                        goal_json,
                        actions_json,
                        total_cost,
                        estimated_duration_seconds,
                        status_str,
                        current_step,
                        created_at_str,
                    )) => {
                        let goal: Goal = serde_json::from_str(&goal_json).map_err(|e| {
                            NagualError::Internal {
                                message: format!("Failed to deserialize goal: {}", e),
                            }
                        })?;

                        let actions: Vec<PlannedAction> =
                            serde_json::from_str(&actions_json).map_err(|e| {
                                NagualError::Internal {
                                    message: format!("Failed to deserialize actions: {}", e),
                                }
                            })?;

                        let status = Self::parse_status(&status_str);
                        let created_at = chrono::DateTime::parse_from_rfc3339(&created_at_str)
                            .map(|dt| dt.with_timezone(&chrono::Utc))
                            .unwrap_or_else(|_| chrono::Utc::now());

                        Ok(Some(Plan {
                            id,
                            goal,
                            actions,
                            total_cost,
                            estimated_duration_seconds: estimated_duration_seconds.map(|d| d as u64),
                            status,
                            current_step: current_step as usize,
                            created_at,
                        }))
                    }
                    None => Ok(None),
                }
            }
        }
    }

    /// List all plans
    pub async fn list(&self) -> Result<Vec<Plan>, NagualError> {
        match &self.backend {
            StorageBackend::Memory(plans) => {
                let plans = plans.read().map_err(|_| NagualError::Internal {
                    message: "Failed to acquire read lock".into(),
                })?;
                Ok(plans.values().cloned().collect())
            }
            StorageBackend::Sqlite(db) => {
                let rows = db
                    .query(
                        r#"
                    SELECT id, goal_json, actions_json, total_cost, estimated_duration_seconds,
                           status, current_step, created_at
                    FROM plans
                    ORDER BY created_at DESC
                    "#,
                        &[],
                        |row| {
                            Ok((
                                row.get::<_, String>(0)?,
                                row.get::<_, String>(1)?,
                                row.get::<_, String>(2)?,
                                row.get::<_, f64>(3)?,
                                row.get::<_, Option<i64>>(4)?,
                                row.get::<_, String>(5)?,
                                row.get::<_, i64>(6)?,
                                row.get::<_, String>(7)?,
                            ))
                        },
                    )
                    .await?;

                let mut plans = Vec::new();
                for (
                    id,
                    goal_json,
                    actions_json,
                    total_cost,
                    estimated_duration_seconds,
                    status_str,
                    current_step,
                    created_at_str,
                ) in rows
                {
                    if let (Ok(goal), Ok(actions)) = (
                        serde_json::from_str::<Goal>(&goal_json),
                        serde_json::from_str::<Vec<PlannedAction>>(&actions_json),
                    ) {
                        let status = Self::parse_status(&status_str);
                        let created_at = chrono::DateTime::parse_from_rfc3339(&created_at_str)
                            .map(|dt| dt.with_timezone(&chrono::Utc))
                            .unwrap_or_else(|_| chrono::Utc::now());

                        plans.push(Plan {
                            id,
                            goal,
                            actions,
                            total_cost,
                            estimated_duration_seconds: estimated_duration_seconds.map(|d| d as u64),
                            status,
                            current_step: current_step as usize,
                            created_at,
                        });
                    }
                }

                Ok(plans)
            }
        }
    }

    /// List plans by status
    pub async fn list_by_status(&self, status: PlanStatus) -> Result<Vec<Plan>, NagualError> {
        match &self.backend {
            StorageBackend::Memory(plans) => {
                let plans = plans.read().map_err(|_| NagualError::Internal {
                    message: "Failed to acquire read lock".into(),
                })?;
                Ok(plans
                    .values()
                    .filter(|p| p.status == status)
                    .cloned()
                    .collect())
            }
            StorageBackend::Sqlite(db) => {
                let status_str = format!("{}", status);
                let rows = db
                    .query(
                        r#"
                    SELECT id, goal_json, actions_json, total_cost, estimated_duration_seconds,
                           status, current_step, created_at
                    FROM plans
                    WHERE status = ?1
                    ORDER BY created_at DESC
                    "#,
                        &[&status_str as &dyn rusqlite::ToSql],
                        |row| {
                            Ok((
                                row.get::<_, String>(0)?,
                                row.get::<_, String>(1)?,
                                row.get::<_, String>(2)?,
                                row.get::<_, f64>(3)?,
                                row.get::<_, Option<i64>>(4)?,
                                row.get::<_, String>(5)?,
                                row.get::<_, i64>(6)?,
                                row.get::<_, String>(7)?,
                            ))
                        },
                    )
                    .await?;

                let mut plans = Vec::new();
                for (
                    id,
                    goal_json,
                    actions_json,
                    total_cost,
                    estimated_duration_seconds,
                    status_str,
                    current_step,
                    created_at_str,
                ) in rows
                {
                    if let (Ok(goal), Ok(actions)) = (
                        serde_json::from_str::<Goal>(&goal_json),
                        serde_json::from_str::<Vec<PlannedAction>>(&actions_json),
                    ) {
                        let status = Self::parse_status(&status_str);
                        let created_at = chrono::DateTime::parse_from_rfc3339(&created_at_str)
                            .map(|dt| dt.with_timezone(&chrono::Utc))
                            .unwrap_or_else(|_| chrono::Utc::now());

                        plans.push(Plan {
                            id,
                            goal,
                            actions,
                            total_cost,
                            estimated_duration_seconds: estimated_duration_seconds.map(|d| d as u64),
                            status,
                            current_step: current_step as usize,
                            created_at,
                        });
                    }
                }

                Ok(plans)
            }
        }
    }

    /// Delete a plan
    pub async fn delete(&self, plan_id: &str) -> Result<bool, NagualError> {
        match &self.backend {
            StorageBackend::Memory(plans) => {
                let mut plans = plans.write().map_err(|_| NagualError::Internal {
                    message: "Failed to acquire write lock".into(),
                })?;
                let existed = plans.remove(plan_id).is_some();
                if existed {
                    info!("Plan {} deleted from memory", plan_id);
                }
                Ok(existed)
            }
            StorageBackend::Sqlite(db) => {
                // Delete steps first
                db.execute(
                    "DELETE FROM plan_steps WHERE plan_id = ?1",
                    &[&plan_id as &dyn rusqlite::ToSql],
                )
                .await?;

                // Delete plan
                let rows_affected = db
                    .execute(
                        "DELETE FROM plans WHERE id = ?1",
                        &[&plan_id as &dyn rusqlite::ToSql],
                    )
                    .await?;

                let existed = rows_affected > 0;
                if existed {
                    info!("Plan {} deleted from SQLite", plan_id);
                }
                Ok(existed)
            }
        }
    }

    /// Get the most recent active plan
    pub async fn get_current(&self) -> Result<Option<Plan>, NagualError> {
        match &self.backend {
            StorageBackend::Memory(plans) => {
                let plans = plans.read().map_err(|_| NagualError::Internal {
                    message: "Failed to acquire read lock".into(),
                })?;
                Ok(plans
                    .values()
                    .filter(|p| !matches!(p.status, PlanStatus::Completed | PlanStatus::Cancelled))
                    .max_by_key(|p| p.created_at)
                    .cloned())
            }
            StorageBackend::Sqlite(db) => {
                let result = db
                    .query_one(
                        r#"
                    SELECT id, goal_json, actions_json, total_cost, estimated_duration_seconds,
                           status, current_step, created_at
                    FROM plans
                    WHERE status NOT IN ('completed', 'cancelled')
                    ORDER BY created_at DESC
                    LIMIT 1
                    "#,
                        &[],
                        |row| {
                            Ok((
                                row.get::<_, String>(0)?,
                                row.get::<_, String>(1)?,
                                row.get::<_, String>(2)?,
                                row.get::<_, f64>(3)?,
                                row.get::<_, Option<i64>>(4)?,
                                row.get::<_, String>(5)?,
                                row.get::<_, i64>(6)?,
                                row.get::<_, String>(7)?,
                            ))
                        },
                    )
                    .await?;

                match result {
                    Some((
                        id,
                        goal_json,
                        actions_json,
                        total_cost,
                        estimated_duration_seconds,
                        status_str,
                        current_step,
                        created_at_str,
                    )) => {
                        let goal: Goal = serde_json::from_str(&goal_json).map_err(|e| {
                            NagualError::Internal {
                                message: format!("Failed to deserialize goal: {}", e),
                            }
                        })?;

                        let actions: Vec<PlannedAction> =
                            serde_json::from_str(&actions_json).map_err(|e| {
                                NagualError::Internal {
                                    message: format!("Failed to deserialize actions: {}", e),
                                }
                            })?;

                        let status = Self::parse_status(&status_str);
                        let created_at = chrono::DateTime::parse_from_rfc3339(&created_at_str)
                            .map(|dt| dt.with_timezone(&chrono::Utc))
                            .unwrap_or_else(|_| chrono::Utc::now());

                        Ok(Some(Plan {
                            id,
                            goal,
                            actions,
                            total_cost,
                            estimated_duration_seconds: estimated_duration_seconds.map(|d| d as u64),
                            status,
                            current_step: current_step as usize,
                            created_at,
                        }))
                    }
                    None => Ok(None),
                }
            }
        }
    }

    /// Count plans by status
    pub async fn count_by_status(&self) -> Result<HashMap<PlanStatus, usize>, NagualError> {
        match &self.backend {
            StorageBackend::Memory(plans) => {
                let plans = plans.read().map_err(|_| NagualError::Internal {
                    message: "Failed to acquire read lock".into(),
                })?;
                let mut counts = HashMap::new();
                for plan in plans.values() {
                    *counts.entry(plan.status).or_insert(0) += 1;
                }
                Ok(counts)
            }
            StorageBackend::Sqlite(db) => {
                let rows = db
                    .query(
                        "SELECT status, COUNT(*) FROM plans GROUP BY status",
                        &[],
                        |row| {
                            let status_str: String = row.get(0)?;
                            let count: i64 = row.get(1)?;
                            Ok((status_str, count as usize))
                        },
                    )
                    .await?;

                let mut counts = HashMap::new();
                for (status_str, count) in rows {
                    let status = Self::parse_status(&status_str);
                    counts.insert(status, count);
                }
                Ok(counts)
            }
        }
    }

    /// Parse status string to enum
    fn parse_status(s: &str) -> PlanStatus {
        match s.to_lowercase().as_str() {
            "planning" => PlanStatus::Planning,
            "ready" => PlanStatus::Ready,
            "executing" => PlanStatus::Executing,
            "paused" => PlanStatus::Paused,
            "completed" => PlanStatus::Completed,
            "failed" => PlanStatus::Failed,
            "cancelled" => PlanStatus::Cancelled,
            "replanning" => PlanStatus::Replanning,
            _ => {
                warn!("Unknown plan status: {}, defaulting to Ready", s);
                PlanStatus::Ready
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_save_and_load_memory() {
        let storage = PlanStorage::in_memory();

        let goal = Goal::new("Test", "Test goal");
        let plan = Plan::new(goal, vec![]);

        storage.save(&plan).await.unwrap();

        let loaded = storage.load(&plan.id).await.unwrap();
        assert!(loaded.is_some());
        assert_eq!(loaded.unwrap().id, plan.id);
    }

    #[tokio::test]
    async fn test_list_by_status_memory() {
        let storage = PlanStorage::in_memory();

        // Create plans with different statuses
        let goal1 = Goal::new("Ready", "");
        let plan1 = Plan::new(goal1, vec![]);
        storage.save(&plan1).await.unwrap();

        let goal2 = Goal::new("Completed", "");
        let mut plan2 = Plan::new(goal2, vec![]);
        plan2.status = PlanStatus::Completed;
        storage.save(&plan2).await.unwrap();

        let ready_plans = storage.list_by_status(PlanStatus::Ready).await.unwrap();
        assert_eq!(ready_plans.len(), 1);

        let completed_plans = storage.list_by_status(PlanStatus::Completed).await.unwrap();
        assert_eq!(completed_plans.len(), 1);
    }

    #[tokio::test]
    async fn test_sqlite_storage() {
        let db = Arc::new(SqliteDb::open_in_memory().unwrap());
        let storage = PlanStorage::sqlite(db).await.unwrap();

        let goal = Goal::new("SQLite Test", "Test SQLite persistence")
            .with_condition(Condition::is_true("test_complete"));
        let plan = Plan::new(goal, vec![]);
        let plan_id = plan.id.clone();

        // Save
        storage.save(&plan).await.unwrap();

        // Load
        let loaded = storage.load(&plan_id).await.unwrap();
        assert!(loaded.is_some());
        let loaded = loaded.unwrap();
        assert_eq!(loaded.id, plan_id);
        assert_eq!(loaded.goal.name, "SQLite Test");
        assert_eq!(loaded.goal.conditions.len(), 1);

        // Update status and save again
        let mut updated = loaded;
        updated.status = PlanStatus::Executing;
        storage.save(&updated).await.unwrap();

        // Verify update
        let reloaded = storage.load(&plan_id).await.unwrap().unwrap();
        assert_eq!(reloaded.status, PlanStatus::Executing);

        // List
        let all_plans = storage.list().await.unwrap();
        assert_eq!(all_plans.len(), 1);

        // List by status
        let executing = storage.list_by_status(PlanStatus::Executing).await.unwrap();
        assert_eq!(executing.len(), 1);

        // Delete
        let deleted = storage.delete(&plan_id).await.unwrap();
        assert!(deleted);

        // Verify deleted
        let after_delete = storage.load(&plan_id).await.unwrap();
        assert!(after_delete.is_none());
    }

    #[tokio::test]
    async fn test_get_current() {
        let db = Arc::new(SqliteDb::open_in_memory().unwrap());
        let storage = PlanStorage::sqlite(db).await.unwrap();

        // No plans yet
        let current = storage.get_current().await.unwrap();
        assert!(current.is_none());

        // Add a ready plan
        let goal1 = Goal::new("Plan 1", "");
        let plan1 = Plan::new(goal1, vec![]);
        storage.save(&plan1).await.unwrap();

        // Should return it
        let current = storage.get_current().await.unwrap();
        assert!(current.is_some());
        assert_eq!(current.unwrap().goal.name, "Plan 1");

        // Add a completed plan (should not be returned as current)
        let goal2 = Goal::new("Plan 2", "");
        let mut plan2 = Plan::new(goal2, vec![]);
        plan2.status = PlanStatus::Completed;
        storage.save(&plan2).await.unwrap();

        // Should still return plan1
        let current = storage.get_current().await.unwrap();
        assert!(current.is_some());
        assert_eq!(current.unwrap().goal.name, "Plan 1");
    }
}
