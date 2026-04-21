//! Prediction Storage with database persistence.
//!
//! This module provides storage operations for predictions including:
//! - Creating, reading, updating, and deleting predictions
//! - Listing predictions with filters
//! - Resolving predictions with Brier score calculation
//! - Calibration bucket management

use std::sync::Arc;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};

use super::{
    bucket_index_for_probability, CalibrationBucket, Prediction,
    PredictionBuilder, PredictionError, PredictionId, PredictionResult, PredictionStatus,
    SQLITE_CALIBRATION_BUCKETS_TABLE, SQLITE_PREDICTIONS_TABLE,
};
use crate::db::SqliteDb;

/// Filter options for listing predictions.
#[derive(Debug, Clone, Default)]
pub struct PredictionFilter {
    /// Filter by status
    pub status: Option<PredictionStatus>,
    /// Filter by domain
    pub domain: Option<String>,
    /// Filter by minimum probability
    pub min_probability: Option<f64>,
    /// Filter by maximum probability
    pub max_probability: Option<f64>,
    /// Filter by creation date (after)
    pub created_after: Option<DateTime<Utc>>,
    /// Filter by creation date (before)
    pub created_before: Option<DateTime<Utc>>,
    /// Limit number of results
    pub limit: Option<usize>,
    /// Offset for pagination
    pub offset: Option<usize>,
}

impl PredictionFilter {
    /// Create a new filter.
    pub fn new() -> Self {
        Self::default()
    }

    /// Filter by status.
    pub fn with_status(mut self, status: PredictionStatus) -> Self {
        self.status = Some(status);
        self
    }

    /// Filter by domain.
    pub fn with_domain(mut self, domain: impl Into<String>) -> Self {
        self.domain = Some(domain.into());
        self
    }

    /// Filter by probability range.
    pub fn with_probability_range(mut self, min: f64, max: f64) -> Self {
        self.min_probability = Some(min);
        self.max_probability = Some(max);
        self
    }

    /// Set limit.
    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = Some(limit);
        self
    }

    /// Set offset.
    pub fn with_offset(mut self, offset: usize) -> Self {
        self.offset = Some(offset);
        self
    }
}

/// Update operations for predictions.
#[derive(Debug, Clone, Default)]
pub struct PredictionUpdate {
    /// New probability value
    pub probability: Option<f64>,
    /// New calibrated probability
    pub calibrated_probability: Option<f64>,
    /// New confidence value
    pub confidence: Option<f64>,
    /// New domain
    pub domain: Option<String>,
    /// New context
    pub context: Option<String>,
    /// Tags to add
    pub add_tags: Vec<String>,
    /// Tags to remove
    pub remove_tags: Vec<String>,
}

/// Storage statistics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StorageStats {
    /// Total number of predictions
    pub total_predictions: usize,
    /// Number of pending predictions
    pub pending_count: usize,
    /// Number of resolved predictions
    pub resolved_count: usize,
    /// Number of expired predictions
    pub expired_count: usize,
    /// Number of cancelled predictions
    pub cancelled_count: usize,
    /// Average Brier score for resolved predictions
    pub avg_brier_score: Option<f64>,
    /// Number of calibration buckets
    pub bucket_count: usize,
}

/// Prediction storage with SQLite backend.
pub struct PredictionStorage {
    /// Database connection
    db: Arc<SqliteDb>,
    /// Default domain
    domain: String,
}

impl PredictionStorage {
    /// Create a new prediction storage.
    pub async fn new(db: Arc<SqliteDb>) -> PredictionResult<Self> {
        let storage = Self {
            db,
            domain: "general".to_string(),
        };
        storage.init_schema().await?;
        Ok(storage)
    }

    /// Create a new prediction storage with a specific domain.
    pub async fn with_domain(
        db: Arc<SqliteDb>,
        domain: impl Into<String>,
    ) -> PredictionResult<Self> {
        let storage = Self {
            db,
            domain: domain.into(),
        };
        storage.init_schema().await?;
        Ok(storage)
    }

    /// Initialize the database schema.
    async fn init_schema(&self) -> PredictionResult<()> {
        self.db
            .execute_batch(SQLITE_PREDICTIONS_TABLE)
            .await
            .map_err(|e| PredictionError::Database(e.to_string()))?;

        self.db
            .execute_batch(SQLITE_CALIBRATION_BUCKETS_TABLE)
            .await
            .map_err(|e| PredictionError::Database(e.to_string()))?;

        // Initialize calibration buckets
        self.init_calibration_buckets().await?;

        info!("Prediction storage schema initialized");
        Ok(())
    }

    /// Initialize calibration buckets.
    async fn init_calibration_buckets(&self) -> PredictionResult<()> {
        let now = Utc::now().to_rfc3339();

        for i in 0..10 {
            let lower = i as f64 * 0.1;
            let upper = (i + 1) as f64 * 0.1;
            let bucket_id = format!("{}_{}", self.domain, i);

            let sql = r#"
                INSERT OR IGNORE INTO calibration_buckets
                (bucket_id, lower_bound, upper_bound, prediction_count,
                 actual_positive_count, total_brier_score, domain, updated_at)
                VALUES (?, ?, ?, 0, 0, 0.0, ?, ?)
            "#;

            self.db
                .execute(sql, &[&bucket_id, &lower, &upper, &self.domain, &now])
                .await
                .map_err(|e| PredictionError::Database(e.to_string()))?;
        }

        Ok(())
    }

    /// Store a new prediction.
    pub async fn store_prediction(&self, prediction: &Prediction) -> PredictionResult<PredictionId> {
        let tags_json =
            serde_json::to_string(prediction.tags()).unwrap_or_else(|_| "[]".to_string());
        let metadata_json =
            serde_json::to_string(prediction.metadata()).unwrap_or_else(|_| "{}".to_string());
        let _evidence_json = serde_json::to_string(prediction.evidence_pattern_ids())
            .unwrap_or_else(|_| "[]".to_string());

        let sql = r#"
            INSERT INTO predictions (
                id, description, probability, calibrated_probability, confidence,
                timeline_min_days, timeline_max_days, status, actual_outcome,
                brier_score, domain, context, created_at, updated_at, resolved_at,
                session_id, agent_id, tags, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        "#;

        let id_str = prediction.id().to_string();
        let status_str = prediction.status().to_string();
        let created_at_str = prediction.created_at().to_rfc3339();
        let updated_at_str = prediction.updated_at().to_rfc3339();
        let resolved_at_str = prediction.resolved_at().map(|dt| dt.to_rfc3339());
        let actual_outcome: Option<i32> = prediction.actual_outcome().map(|b| if b { 1 } else { 0 });

        self.db
            .execute(
                sql,
                &[
                    &id_str as &dyn rusqlite::ToSql,
                    &prediction.description(),
                    &prediction.probability(),
                    &prediction.calibrated_probability(),
                    &prediction.confidence(),
                    &(prediction.timeline_min_days() as i32),
                    &(prediction.timeline_max_days() as i32),
                    &status_str,
                    &actual_outcome,
                    &prediction.brier_score(),
                    &prediction.domain(),
                    &prediction.context(),
                    &created_at_str,
                    &updated_at_str,
                    &resolved_at_str,
                    &prediction.session_id(),
                    &prediction.agent_id(),
                    &tags_json,
                    &metadata_json,
                ],
            )
            .await
            .map_err(|e| PredictionError::Database(e.to_string()))?;

        debug!(prediction_id = %id_str, "Prediction stored");
        Ok(prediction.id().clone())
    }

    /// Get a prediction by ID.
    pub async fn get_prediction(&self, id: &PredictionId) -> PredictionResult<Option<Prediction>> {
        let sql = "SELECT * FROM predictions WHERE id = ?";
        let id_str = id.to_string();

        let prediction = self
            .db
            .query_one(sql, &[&id_str], |row| Self::prediction_from_row(row))
            .await
            .map_err(|e| PredictionError::Database(e.to_string()))?;

        Ok(prediction)
    }

    /// List predictions with optional filters.
    pub async fn list_predictions(
        &self,
        filter: &PredictionFilter,
    ) -> PredictionResult<Vec<Prediction>> {
        let mut sql = "SELECT * FROM predictions WHERE 1=1".to_string();
        let mut params: Vec<Box<dyn rusqlite::ToSql + Send + Sync>> = Vec::new();

        if let Some(status) = &filter.status {
            sql.push_str(" AND status = ?");
            params.push(Box::new(status.to_string()));
        }

        if let Some(domain) = &filter.domain {
            sql.push_str(" AND domain = ?");
            params.push(Box::new(domain.clone()));
        }

        if let Some(min_prob) = filter.min_probability {
            sql.push_str(" AND probability >= ?");
            params.push(Box::new(min_prob));
        }

        if let Some(max_prob) = filter.max_probability {
            sql.push_str(" AND probability <= ?");
            params.push(Box::new(max_prob));
        }

        sql.push_str(" ORDER BY created_at DESC");

        if let Some(limit) = filter.limit {
            sql.push_str(&format!(" LIMIT {}", limit));
        }

        if let Some(offset) = filter.offset {
            sql.push_str(&format!(" OFFSET {}", offset));
        }

        // Convert params for the query
        let param_refs: Vec<&dyn rusqlite::ToSql> = params
            .iter()
            .map(|p| p.as_ref() as &dyn rusqlite::ToSql)
            .collect();

        let predictions = self
            .db
            .query(&sql, param_refs.as_slice(), |row| {
                Self::prediction_from_row(row)
            })
            .await
            .map_err(|e| PredictionError::Database(e.to_string()))?;

        Ok(predictions)
    }

    /// List pending predictions.
    pub async fn list_pending(&self, limit: usize) -> PredictionResult<Vec<Prediction>> {
        self.list_predictions(
            &PredictionFilter::new()
                .with_status(PredictionStatus::Pending)
                .with_limit(limit),
        )
        .await
    }

    /// List resolved predictions.
    pub async fn list_resolved(&self, limit: usize) -> PredictionResult<Vec<Prediction>> {
        self.list_predictions(
            &PredictionFilter::new()
                .with_status(PredictionStatus::Resolved)
                .with_limit(limit),
        )
        .await
    }

    /// Resolve a prediction with an actual outcome.
    ///
    /// This calculates the Brier score and updates calibration buckets.
    pub async fn resolve_prediction(
        &self,
        id: &PredictionId,
        outcome: bool,
    ) -> PredictionResult<Prediction> {
        // Get the prediction
        let mut prediction = self
            .get_prediction(id)
            .await?
            .ok_or_else(|| PredictionError::NotFound { id: id.to_string() })?;

        // Check if already resolved
        if prediction.status() == PredictionStatus::Resolved {
            return Err(PredictionError::AlreadyResolved {
                id: id.to_string(),
                resolved_at: prediction.resolved_at().unwrap_or_else(Utc::now),
            });
        }

        // Resolve the prediction (calculates Brier score)
        prediction.resolve(outcome)?;

        // Update in database
        let now = Utc::now();
        let resolved_at_str = now.to_rfc3339();
        let updated_at_str = now.to_rfc3339();
        let outcome_int: i32 = if outcome { 1 } else { 0 };
        let brier_score = prediction.brier_score().unwrap_or(0.0);

        let update_sql = r#"
            UPDATE predictions SET
                status = 'resolved',
                actual_outcome = ?,
                brier_score = ?,
                resolved_at = ?,
                updated_at = ?
            WHERE id = ?
        "#;

        let id_str = id.to_string();
        self.db
            .execute(
                update_sql,
                &[
                    &outcome_int as &dyn rusqlite::ToSql,
                    &brier_score,
                    &resolved_at_str,
                    &updated_at_str,
                    &id_str,
                ],
            )
            .await
            .map_err(|e| PredictionError::Database(e.to_string()))?;

        // Update calibration bucket
        self.update_calibration_bucket(prediction.probability(), outcome, brier_score)
            .await?;

        info!(
            prediction_id = %id,
            brier_score = %brier_score,
            "Prediction resolved"
        );

        Ok(prediction)
    }

    /// Update calibration bucket for a resolved prediction.
    async fn update_calibration_bucket(
        &self,
        probability: f64,
        outcome: bool,
        brier_score: f64,
    ) -> PredictionResult<()> {
        let bucket_idx = bucket_index_for_probability(probability);
        let bucket_id = format!("{}_{}", self.domain, bucket_idx);
        let now = Utc::now().to_rfc3339();
        let positive_increment: i32 = if outcome { 1 } else { 0 };

        let sql = r#"
            UPDATE calibration_buckets SET
                prediction_count = prediction_count + 1,
                actual_positive_count = actual_positive_count + ?,
                total_brier_score = total_brier_score + ?,
                updated_at = ?
            WHERE bucket_id = ?
        "#;

        self.db
            .execute(
                sql,
                &[
                    &positive_increment as &dyn rusqlite::ToSql,
                    &brier_score,
                    &now,
                    &bucket_id,
                ],
            )
            .await
            .map_err(|e| PredictionError::Database(e.to_string()))?;

        debug!(
            bucket_id = %bucket_id,
            probability = %probability,
            outcome = %outcome,
            "Calibration bucket updated"
        );

        Ok(())
    }

    /// Cancel a prediction.
    pub async fn cancel_prediction(&self, id: &PredictionId) -> PredictionResult<()> {
        let prediction = self
            .get_prediction(id)
            .await?
            .ok_or_else(|| PredictionError::NotFound { id: id.to_string() })?;

        if !prediction.is_pending() {
            warn!(
                prediction_id = %id,
                "Cannot cancel non-pending prediction"
            );
            return Ok(());
        }

        let now = Utc::now().to_rfc3339();
        let id_str = id.to_string();
        let sql = "UPDATE predictions SET status = 'cancelled', updated_at = ? WHERE id = ?";

        self.db
            .execute(sql, &[&now, &id_str])
            .await
            .map_err(|e| PredictionError::Database(e.to_string()))?;

        info!(prediction_id = %id, "Prediction cancelled");
        Ok(())
    }

    /// Delete a prediction.
    pub async fn delete_prediction(&self, id: &PredictionId) -> PredictionResult<()> {
        let id_str = id.to_string();
        let sql = "DELETE FROM predictions WHERE id = ?";

        self.db
            .execute(sql, &[&id_str])
            .await
            .map_err(|e| PredictionError::Database(e.to_string()))?;

        info!(prediction_id = %id, "Prediction deleted");
        Ok(())
    }

    /// Get calibration buckets.
    pub async fn get_calibration_buckets(&self) -> PredictionResult<Vec<CalibrationBucket>> {
        let sql = r#"
            SELECT bucket_id, lower_bound, upper_bound, prediction_count,
                   actual_positive_count, total_brier_score, domain, updated_at
            FROM calibration_buckets
            WHERE domain = ?
            ORDER BY lower_bound ASC
        "#;

        let buckets = self
            .db
            .query(sql, &[&self.domain], |row| {
                let bucket_id: String = row.get("bucket_id")?;
                let lower_bound: f64 = row.get("lower_bound")?;
                let upper_bound: f64 = row.get("upper_bound")?;
                let prediction_count: i32 = row.get("prediction_count")?;
                let actual_positive_count: i32 = row.get("actual_positive_count")?;
                let total_brier_score: f64 = row.get("total_brier_score")?;
                let domain: String = row.get("domain")?;
                let updated_at_str: String = row.get("updated_at")?;

                let updated_at = DateTime::parse_from_rfc3339(&updated_at_str)
                    .map(|dt| dt.with_timezone(&Utc))
                    .unwrap_or_else(|_| Utc::now());

                Ok(CalibrationBucket {
                    id: bucket_id,
                    lower_bound,
                    upper_bound,
                    prediction_count: prediction_count as u32,
                    actual_positive_count: actual_positive_count as u32,
                    total_brier_score,
                    domain,
                    updated_at,
                })
            })
            .await
            .map_err(|e| PredictionError::Database(e.to_string()))?;

        Ok(buckets)
    }

    /// Get storage statistics.
    pub async fn get_stats(&self) -> PredictionResult<StorageStats> {
        let count_sql = r#"
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END) as pending,
                SUM(CASE WHEN status = 'resolved' THEN 1 ELSE 0 END) as resolved,
                SUM(CASE WHEN status = 'expired' THEN 1 ELSE 0 END) as expired,
                SUM(CASE WHEN status = 'cancelled' THEN 1 ELSE 0 END) as cancelled
            FROM predictions
        "#;

        let counts = self
            .db
            .query_one(count_sql, &[], |row| {
                Ok((
                    row.get::<_, i64>(0)? as usize,
                    row.get::<_, i64>(1)? as usize,
                    row.get::<_, i64>(2)? as usize,
                    row.get::<_, i64>(3)? as usize,
                    row.get::<_, i64>(4)? as usize,
                ))
            })
            .await
            .map_err(|e| PredictionError::Database(e.to_string()))?
            .unwrap_or((0, 0, 0, 0, 0));

        let avg_sql = r#"
            SELECT AVG(brier_score) FROM predictions
            WHERE status = 'resolved' AND brier_score IS NOT NULL
        "#;

        let avg_brier = self
            .db
            .query_one(avg_sql, &[], |row| row.get::<_, Option<f64>>(0))
            .await
            .map_err(|e| PredictionError::Database(e.to_string()))?
            .flatten();

        let buckets = self.get_calibration_buckets().await?;

        Ok(StorageStats {
            total_predictions: counts.0,
            pending_count: counts.1,
            resolved_count: counts.2,
            expired_count: counts.3,
            cancelled_count: counts.4,
            avg_brier_score: avg_brier,
            bucket_count: buckets.len(),
        })
    }

    /// Get the domain.
    pub fn domain(&self) -> &str {
        &self.domain
    }

    /// Convert a database row to a Prediction.
    fn prediction_from_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<Prediction> {
        let id: String = row.get("id")?;
        let description: String = row.get("description")?;
        let probability: f64 = row.get("probability")?;
        let calibrated_probability: Option<f64> = row.get("calibrated_probability")?;
        let confidence: f64 = row.get("confidence")?;
        let timeline_min_days: i32 = row.get("timeline_min_days")?;
        let timeline_max_days: i32 = row.get("timeline_max_days")?;
        let status_str: String = row.get("status")?;
        let actual_outcome: Option<i32> = row.get("actual_outcome")?;
        let brier_score: Option<f64> = row.get("brier_score")?;
        let domain: String = row.get("domain")?;
        let context: Option<String> = row.get("context")?;
        let created_at_str: String = row.get("created_at")?;
        let _updated_at_str: String = row.get("updated_at")?;
        let resolved_at_str: Option<String> = row.get("resolved_at")?;
        let session_id: Option<String> = row.get("session_id")?;
        let agent_id: Option<String> = row.get("agent_id")?;
        let tags_json: String = row
            .get::<_, Option<String>>("tags")?
            .unwrap_or_else(|| "[]".to_string());
        let metadata_json: String = row
            .get::<_, Option<String>>("metadata")?
            .unwrap_or_else(|| "{}".to_string());

        let status = status_str
            .parse::<PredictionStatus>()
            .unwrap_or(PredictionStatus::Pending);

        let created_at = DateTime::parse_from_rfc3339(&created_at_str)
            .map(|dt| dt.with_timezone(&Utc))
            .unwrap_or_else(|_| Utc::now());

        let resolved_at = resolved_at_str.and_then(|s| {
            DateTime::parse_from_rfc3339(&s)
                .map(|dt| dt.with_timezone(&Utc))
                .ok()
        });

        let tags: Vec<String> = serde_json::from_str(&tags_json).unwrap_or_default();
        let metadata: std::collections::HashMap<String, serde_json::Value> =
            serde_json::from_str(&metadata_json).unwrap_or_default();

        let mut builder = PredictionBuilder::new()
            .id(id)
            .created_at(created_at)
            .description(description)
            .probability(probability)
            .confidence(confidence)
            .timeline_min_days(timeline_min_days as u32)
            .timeline_max_days(timeline_max_days as u32)
            .status(status)
            .domain(domain)
            .tags(tags)
            .metadata(metadata);

        if let Some(cp) = calibrated_probability {
            builder = builder.calibrated_probability(cp);
        }

        if let Some(outcome) = actual_outcome {
            builder = builder.actual_outcome(outcome != 0);
        }

        if let Some(bs) = brier_score {
            builder = builder.brier_score(bs);
        }

        if let Some(ra) = resolved_at {
            builder = builder.resolved_at(ra);
        }

        if let Some(ctx) = context {
            builder = builder.context(ctx);
        }

        if let Some(sid) = session_id {
            builder = builder.session_id(sid);
        }

        if let Some(aid) = agent_id {
            builder = builder.agent_id(aid);
        }

        builder.build().map_err(|e| {
            rusqlite::Error::FromSqlConversionFailure(0, rusqlite::types::Type::Text, Box::new(e))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn test_db() -> Arc<SqliteDb> {
        Arc::new(SqliteDb::open_in_memory().unwrap())
    }

    #[tokio::test]
    async fn test_storage_creation() {
        let db = test_db().await;
        let storage = PredictionStorage::new(db).await.unwrap();
        assert_eq!(storage.domain(), "general");
    }

    #[tokio::test]
    async fn test_store_and_get_prediction() {
        let db = test_db().await;
        let storage = PredictionStorage::new(db).await.unwrap();

        let prediction = Prediction::new("Test prediction", 0.75).unwrap();
        let id = storage.store_prediction(&prediction).await.unwrap();

        let retrieved = storage.get_prediction(&id).await.unwrap().unwrap();
        assert_eq!(retrieved.description(), "Test prediction");
    }

    #[tokio::test]
    async fn test_resolve_prediction() {
        let db = test_db().await;
        let storage = PredictionStorage::new(db).await.unwrap();

        let prediction = Prediction::new("Test", 0.8).unwrap();
        let id = storage.store_prediction(&prediction).await.unwrap();

        let resolved = storage.resolve_prediction(&id, true).await.unwrap();

        assert!(resolved.is_resolved());
        assert_eq!(resolved.actual_outcome(), Some(true));

        // Brier score should be (0.8 - 1.0)^2 = 0.04
        let brier = resolved.brier_score().unwrap();
        assert!((brier - 0.04).abs() < 0.001);
    }

    #[tokio::test]
    async fn test_list_predictions_with_filter() {
        let db = test_db().await;
        let storage = PredictionStorage::new(db).await.unwrap();

        // Store some predictions
        for i in 0..5 {
            let p = Prediction::new(format!("Prediction {}", i), 0.5).unwrap();
            storage.store_prediction(&p).await.unwrap();
        }

        let filter = PredictionFilter::new()
            .with_status(PredictionStatus::Pending)
            .with_limit(3);

        let predictions = storage.list_predictions(&filter).await.unwrap();
        assert_eq!(predictions.len(), 3);
    }

    #[tokio::test]
    async fn test_calibration_buckets() {
        let db = test_db().await;
        let storage = PredictionStorage::new(db).await.unwrap();

        let buckets = storage.get_calibration_buckets().await.unwrap();
        assert_eq!(buckets.len(), 10);
    }

    #[tokio::test]
    async fn test_storage_stats() {
        let db = test_db().await;
        let storage = PredictionStorage::new(db).await.unwrap();

        let p1 = Prediction::new("P1", 0.7).unwrap();
        let id1 = storage.store_prediction(&p1).await.unwrap();

        let p2 = Prediction::new("P2", 0.6).unwrap();
        storage.store_prediction(&p2).await.unwrap();

        // Resolve one
        storage.resolve_prediction(&id1, true).await.unwrap();

        let stats = storage.get_stats().await.unwrap();
        assert_eq!(stats.total_predictions, 2);
        assert_eq!(stats.pending_count, 1);
        assert_eq!(stats.resolved_count, 1);
    }
}
