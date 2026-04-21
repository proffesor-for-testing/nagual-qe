//! Prediction resolution with Brier score calculation.
//!
//! This module provides functionality for:
//! - Creating and storing predictions
//! - Resolving predictions with actual outcomes
//! - Calculating Brier scores for calibration tracking

use std::sync::Arc;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};
use uuid::Uuid;

use super::{
    bucket_index_for_probability, calculate_brier_score, update_bucket, CalibrationBucket,
    PredictionError, PredictionResult, PredictionStatus, SQLITE_CALIBRATION_BUCKETS_TABLE,
    SQLITE_PREDICTIONS_TABLE,
};
use crate::db::SqliteDb;

/// A prediction with probability estimate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Prediction {
    /// Unique identifier
    pub id: String,

    /// Description of what is being predicted
    pub description: String,

    /// Predicted probability (0.0 to 1.0)
    pub probability: f64,

    /// Calibrated probability after adjustment (optional)
    pub calibrated_probability: Option<f64>,

    /// Current status
    pub status: PredictionStatus,

    /// Actual outcome (true/false) after resolution
    pub actual_outcome: Option<bool>,

    /// Brier score after resolution
    pub brier_score: Option<f64>,

    /// Domain/category for the prediction
    pub domain: String,

    /// Additional context
    pub context: Option<String>,

    /// When the prediction was created
    pub created_at: DateTime<Utc>,

    /// When the prediction was resolved
    pub resolved_at: Option<DateTime<Utc>>,

    /// Who created the prediction
    pub created_by: Option<String>,

    /// Tags for categorization
    pub tags: Vec<String>,
}

impl Prediction {
    /// Create a new prediction with the given description.
    pub fn new(description: impl Into<String>) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            description: description.into(),
            probability: 0.5, // Default to maximum uncertainty
            calibrated_probability: None,
            status: PredictionStatus::Pending,
            actual_outcome: None,
            brier_score: None,
            domain: "general".to_string(),
            context: None,
            created_at: Utc::now(),
            resolved_at: None,
            created_by: None,
            tags: Vec::new(),
        }
    }

    /// Set the predicted probability.
    pub fn with_probability(mut self, probability: f64) -> Self {
        self.probability = probability.clamp(0.0, 1.0);
        self
    }

    /// Set the calibrated probability.
    pub fn with_calibrated_probability(mut self, probability: f64) -> Self {
        self.calibrated_probability = Some(probability.clamp(0.0, 1.0));
        self
    }

    /// Set the domain.
    pub fn with_domain(mut self, domain: impl Into<String>) -> Self {
        self.domain = domain.into();
        self
    }

    /// Set the context.
    pub fn with_context(mut self, context: impl Into<String>) -> Self {
        self.context = Some(context.into());
        self
    }

    /// Set the creator.
    pub fn with_created_by(mut self, created_by: impl Into<String>) -> Self {
        self.created_by = Some(created_by.into());
        self
    }

    /// Add tags.
    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.tags = tags;
        self
    }

    /// Add a single tag.
    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.push(tag.into());
        self
    }

    /// Check if the prediction is still pending.
    pub fn is_pending(&self) -> bool {
        self.status == PredictionStatus::Pending
    }

    /// Check if the prediction has been resolved.
    pub fn is_resolved(&self) -> bool {
        self.status == PredictionStatus::Resolved
    }

    /// Get the effective probability (calibrated if available, otherwise raw).
    pub fn effective_probability(&self) -> f64 {
        self.calibrated_probability.unwrap_or(self.probability)
    }
}

impl Default for Prediction {
    fn default() -> Self {
        Self::new("Unnamed prediction")
    }
}

/// Result of resolving a prediction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolutionResult {
    /// The prediction ID
    pub prediction_id: String,

    /// The original predicted probability
    pub predicted_probability: f64,

    /// The calibrated probability (if any)
    pub calibrated_probability: Option<f64>,

    /// The actual outcome
    pub actual_outcome: bool,

    /// The calculated Brier score
    pub brier_score: f64,

    /// When the resolution occurred
    pub resolved_at: DateTime<Utc>,

    /// The calibration bucket this prediction falls into
    pub bucket_index: usize,

    /// Summary message
    pub message: String,
}

impl ResolutionResult {
    /// Create a new resolution result.
    pub fn new(
        prediction_id: String,
        predicted_probability: f64,
        calibrated_probability: Option<f64>,
        actual_outcome: bool,
    ) -> Self {
        let brier_score = calculate_brier_score(predicted_probability, actual_outcome);
        let bucket_index = bucket_index_for_probability(predicted_probability);
        let resolved_at = Utc::now();

        let outcome_str = if actual_outcome { "true" } else { "false" };
        let message = format!(
            "Prediction {} resolved: probability {:.2}%, outcome {}, Brier score {:.4}",
            prediction_id,
            predicted_probability * 100.0,
            outcome_str,
            brier_score
        );

        Self {
            prediction_id,
            predicted_probability,
            calibrated_probability,
            actual_outcome,
            brier_score,
            resolved_at,
            bucket_index,
            message,
        }
    }

    /// Check if this was a good prediction (Brier score < 0.25).
    pub fn is_good(&self) -> bool {
        self.brier_score < 0.25
    }

    /// Get a qualitative assessment of the prediction quality.
    pub fn quality(&self) -> &'static str {
        match self.brier_score {
            s if s < 0.1 => "Excellent",
            s if s < 0.2 => "Good",
            s if s < 0.3 => "Fair",
            s if s < 0.5 => "Poor",
            _ => "Very Poor",
        }
    }
}

/// Store and manage predictions in the database.
pub struct PredictionStore {
    /// SQLite database connection
    db: Arc<SqliteDb>,

    /// Domain for this store
    domain: String,
}

impl PredictionStore {
    /// Create a new prediction store.
    pub async fn new(db: Arc<SqliteDb>) -> PredictionResult<Self> {
        let store = Self {
            db,
            domain: "general".to_string(),
        };
        store.init_schema().await?;
        Ok(store)
    }

    /// Create a new prediction store with a specific domain.
    pub async fn new_with_domain(
        db: Arc<SqliteDb>,
        domain: impl Into<String>,
    ) -> PredictionResult<Self> {
        let store = Self {
            db,
            domain: domain.into(),
        };
        store.init_schema().await?;
        Ok(store)
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

        // Initialize calibration buckets if they don't exist
        self.init_calibration_buckets().await?;

        info!("Prediction store schema initialized");
        Ok(())
    }

    /// Initialize calibration buckets for all domains.
    async fn init_calibration_buckets(&self) -> PredictionResult<()> {
        let now = Utc::now().to_rfc3339();

        for i in 0..10 {
            let lower = i as f64 * 0.1;
            let upper = (i + 1) as f64 * 0.1;
            let bucket_id = format!("{}_{}", self.domain, i);

            let sql = r#"
                INSERT OR IGNORE INTO calibration_buckets
                (bucket_id, lower_bound, upper_bound, prediction_count, actual_positive_count, total_brier_score, domain, updated_at)
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
    pub async fn store(&self, prediction: &Prediction) -> PredictionResult<String> {
        // Validate probability
        if !(0.0..=1.0).contains(&prediction.probability) {
            return Err(PredictionError::InvalidProbability {
                value: prediction.probability,
            });
        }

        let tags_json =
            serde_json::to_string(&prediction.tags).unwrap_or_else(|_| "[]".to_string());

        let sql = r#"
            INSERT INTO predictions (
                id, description, probability, calibrated_probability, status,
                actual_outcome, brier_score, domain, context, created_at,
                resolved_at, created_by, tags
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        "#;

        let status_str = prediction.status.to_string();
        let created_at_str = prediction.created_at.to_rfc3339();
        let resolved_at_str = prediction.resolved_at.map(|dt| dt.to_rfc3339());

        self.db
            .execute(
                sql,
                &[
                    &prediction.id as &dyn rusqlite::ToSql,
                    &prediction.description,
                    &prediction.probability,
                    &prediction.calibrated_probability,
                    &status_str,
                    &prediction.actual_outcome,
                    &prediction.brier_score,
                    &prediction.domain,
                    &prediction.context,
                    &created_at_str,
                    &resolved_at_str,
                    &prediction.created_by,
                    &tags_json,
                ],
            )
            .await
            .map_err(|e| PredictionError::Database(e.to_string()))?;

        debug!(prediction_id = %prediction.id, "Prediction stored");
        Ok(prediction.id.clone())
    }

    /// Get a prediction by ID.
    pub async fn get(&self, id: &str) -> PredictionResult<Option<Prediction>> {
        let sql = "SELECT * FROM predictions WHERE id = ?";

        let prediction = self
            .db
            .query_one(sql, &[&id], |row| Self::prediction_from_row(row))
            .await
            .map_err(|e| PredictionError::Database(e.to_string()))?;

        Ok(prediction)
    }

    /// List pending predictions.
    pub async fn list_pending(&self, limit: usize) -> PredictionResult<Vec<Prediction>> {
        let sql = r#"
            SELECT * FROM predictions
            WHERE status = 'pending'
            ORDER BY created_at DESC
            LIMIT ?
        "#;

        let limit_i64 = limit as i64;
        let predictions = self
            .db
            .query(sql, &[&limit_i64], |row| Self::prediction_from_row(row))
            .await
            .map_err(|e| PredictionError::Database(e.to_string()))?;

        Ok(predictions)
    }

    /// List resolved predictions.
    pub async fn list_resolved(&self, limit: usize) -> PredictionResult<Vec<Prediction>> {
        let sql = r#"
            SELECT * FROM predictions
            WHERE status = 'resolved'
            ORDER BY resolved_at DESC
            LIMIT ?
        "#;

        let limit_i64 = limit as i64;
        let predictions = self
            .db
            .query(sql, &[&limit_i64], |row| Self::prediction_from_row(row))
            .await
            .map_err(|e| PredictionError::Database(e.to_string()))?;

        Ok(predictions)
    }

    /// List all predictions.
    pub async fn list_all(&self, limit: usize) -> PredictionResult<Vec<Prediction>> {
        let sql = r#"
            SELECT * FROM predictions
            ORDER BY created_at DESC
            LIMIT ?
        "#;

        let limit_i64 = limit as i64;
        let predictions = self
            .db
            .query(sql, &[&limit_i64], |row| Self::prediction_from_row(row))
            .await
            .map_err(|e| PredictionError::Database(e.to_string()))?;

        Ok(predictions)
    }

    /// Resolve a prediction with an actual outcome.
    ///
    /// This is the main function for Task 3.E.1:
    /// - Takes prediction_id and actual_outcome (true/false)
    /// - Calculates Brier score: (probability - outcome)^2
    /// - Updates prediction record with brier_score and resolved_at
    /// - Updates calibration buckets
    /// - Returns ResolutionResult with score
    pub async fn resolve(
        &self,
        prediction_id: &str,
        actual_outcome: bool,
    ) -> PredictionResult<ResolutionResult> {
        // Get the prediction
        let prediction = self
            .get(prediction_id)
            .await?
            .ok_or_else(|| PredictionError::NotFound {
                id: prediction_id.to_string(),
            })?;

        // Check if already resolved
        if prediction.status == PredictionStatus::Resolved {
            return Err(PredictionError::AlreadyResolved {
                id: prediction_id.to_string(),
            });
        }

        // Create resolution result (calculates Brier score)
        let result = ResolutionResult::new(
            prediction_id.to_string(),
            prediction.probability,
            prediction.calibrated_probability,
            actual_outcome,
        );

        // Update the prediction record
        let now = Utc::now();
        let resolved_at_str = now.to_rfc3339();
        let outcome_int: i32 = if actual_outcome { 1 } else { 0 };

        let update_sql = r#"
            UPDATE predictions SET
                status = 'resolved',
                actual_outcome = ?,
                brier_score = ?,
                resolved_at = ?
            WHERE id = ?
        "#;

        self.db
            .execute(
                update_sql,
                &[
                    &outcome_int as &dyn rusqlite::ToSql,
                    &result.brier_score,
                    &resolved_at_str,
                    &prediction_id,
                ],
            )
            .await
            .map_err(|e| PredictionError::Database(e.to_string()))?;

        // Update calibration bucket
        self.update_calibration_bucket(
            prediction.probability,
            actual_outcome,
            result.brier_score,
        )
        .await?;

        info!(
            prediction_id = %prediction_id,
            brier_score = %result.brier_score,
            quality = %result.quality(),
            "Prediction resolved"
        );

        Ok(result)
    }

    /// Update the calibration bucket for a resolved prediction.
    async fn update_calibration_bucket(
        &self,
        probability: f64,
        actual_outcome: bool,
        brier_score: f64,
    ) -> PredictionResult<()> {
        let bucket_index = bucket_index_for_probability(probability);
        let bucket_id = format!("{}_{}", self.domain, bucket_index);
        let now = Utc::now().to_rfc3339();
        let positive_increment: i32 = if actual_outcome { 1 } else { 0 };

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
            outcome = %actual_outcome,
            "Calibration bucket updated"
        );

        Ok(())
    }

    /// Cancel a pending prediction.
    pub async fn cancel(&self, prediction_id: &str) -> PredictionResult<()> {
        let prediction = self
            .get(prediction_id)
            .await?
            .ok_or_else(|| PredictionError::NotFound {
                id: prediction_id.to_string(),
            })?;

        if !prediction.is_pending() {
            warn!(
                prediction_id = %prediction_id,
                "Cannot cancel non-pending prediction"
            );
            return Ok(());
        }

        let sql = "UPDATE predictions SET status = 'cancelled' WHERE id = ?";
        self.db
            .execute(sql, &[&prediction_id])
            .await
            .map_err(|e| PredictionError::Database(e.to_string()))?;

        info!(prediction_id = %prediction_id, "Prediction cancelled");
        Ok(())
    }

    /// Get calibration buckets for the current domain.
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
                    bucket_id,
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

    /// Get prediction count by status.
    pub async fn count_by_status(&self, status: PredictionStatus) -> PredictionResult<usize> {
        let sql = "SELECT COUNT(*) FROM predictions WHERE status = ?";
        let status_str = status.to_string();

        let count = self
            .db
            .query_one(sql, &[&status_str], |row| row.get::<_, i64>(0))
            .await
            .map_err(|e| PredictionError::Database(e.to_string()))?
            .unwrap_or(0);

        Ok(count as usize)
    }

    /// Get total prediction count.
    pub async fn count_total(&self) -> PredictionResult<usize> {
        let sql = "SELECT COUNT(*) FROM predictions";

        let count = self
            .db
            .query_one(sql, &[], |row| row.get::<_, i64>(0))
            .await
            .map_err(|e| PredictionError::Database(e.to_string()))?
            .unwrap_or(0);

        Ok(count as usize)
    }

    /// Get average Brier score for resolved predictions.
    pub async fn average_brier_score(&self) -> PredictionResult<Option<f64>> {
        let sql = r#"
            SELECT AVG(brier_score) FROM predictions
            WHERE status = 'resolved' AND brier_score IS NOT NULL
        "#;

        let avg = self
            .db
            .query_one(sql, &[], |row| row.get::<_, Option<f64>>(0))
            .await
            .map_err(|e| PredictionError::Database(e.to_string()))?
            .flatten();

        Ok(avg)
    }

    /// Convert a database row to a Prediction.
    fn prediction_from_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<Prediction> {
        let id: String = row.get("id")?;
        let description: String = row.get("description")?;
        let probability: f64 = row.get("probability")?;
        let calibrated_probability: Option<f64> = row.get("calibrated_probability")?;
        let status_str: String = row.get("status")?;
        let actual_outcome: Option<i32> = row.get("actual_outcome")?;
        let brier_score: Option<f64> = row.get("brier_score")?;
        let domain: String = row.get("domain")?;
        let context: Option<String> = row.get("context")?;
        let created_at_str: String = row.get("created_at")?;
        let resolved_at_str: Option<String> = row.get("resolved_at")?;
        let created_by: Option<String> = row.get("created_by")?;
        let tags_json: String = row.get::<_, Option<String>>("tags")?.unwrap_or_else(|| "[]".to_string());

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

        Ok(Prediction {
            id,
            description,
            probability,
            calibrated_probability,
            status,
            actual_outcome: actual_outcome.map(|v| v != 0),
            brier_score,
            domain,
            context,
            created_at,
            resolved_at,
            created_by,
            tags,
        })
    }

    /// Get the domain for this store.
    pub fn domain(&self) -> &str {
        &self.domain
    }
}

/// Resolve a prediction with an actual outcome (standalone function).
///
/// This is the main entry point for Task 3.E.1.
pub async fn resolve_prediction(
    db: Arc<SqliteDb>,
    prediction_id: &str,
    actual_outcome: bool,
) -> PredictionResult<ResolutionResult> {
    let store = PredictionStore::new(db).await?;
    store.resolve(prediction_id, actual_outcome).await
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn test_db() -> Arc<SqliteDb> {
        Arc::new(SqliteDb::open_in_memory().unwrap())
    }

    #[tokio::test]
    async fn test_prediction_creation() {
        let prediction = Prediction::new("Test prediction")
            .with_probability(0.7)
            .with_domain("testing")
            .with_tag("test");

        assert_eq!(prediction.description, "Test prediction");
        assert!((prediction.probability - 0.7).abs() < f64::EPSILON);
        assert_eq!(prediction.domain, "testing");
        assert!(prediction.is_pending());
    }

    #[tokio::test]
    async fn test_prediction_store_creation() {
        let db = test_db().await;
        let store = PredictionStore::new(db).await.unwrap();
        assert_eq!(store.domain(), "general");
    }

    #[tokio::test]
    async fn test_store_and_retrieve_prediction() {
        let db = test_db().await;
        let store = PredictionStore::new(db).await.unwrap();

        let prediction = Prediction::new("Will it rain tomorrow?").with_probability(0.65);

        let id = store.store(&prediction).await.unwrap();
        let retrieved = store.get(&id).await.unwrap().unwrap();

        assert_eq!(retrieved.description, "Will it rain tomorrow?");
        assert!((retrieved.probability - 0.65).abs() < f64::EPSILON);
        assert!(retrieved.is_pending());
    }

    #[tokio::test]
    async fn test_resolve_prediction() {
        let db = test_db().await;
        let store = PredictionStore::new(db).await.unwrap();

        let prediction = Prediction::new("Test").with_probability(0.8);
        let id = store.store(&prediction).await.unwrap();

        // Resolve with outcome = true
        let result = store.resolve(&id, true).await.unwrap();

        // Brier score should be (0.8 - 1.0)^2 = 0.04
        assert!((result.brier_score - 0.04).abs() < 1e-10);
        assert!(result.actual_outcome);

        // Verify the prediction was updated
        let resolved = store.get(&id).await.unwrap().unwrap();
        assert!(resolved.is_resolved());
        assert_eq!(resolved.actual_outcome, Some(true));
        assert!(resolved.brier_score.is_some());
    }

    #[tokio::test]
    async fn test_resolve_prediction_false_outcome() {
        let db = test_db().await;
        let store = PredictionStore::new(db).await.unwrap();

        let prediction = Prediction::new("Test").with_probability(0.3);
        let id = store.store(&prediction).await.unwrap();

        // Resolve with outcome = false
        let result = store.resolve(&id, false).await.unwrap();

        // Brier score should be (0.3 - 0.0)^2 = 0.09
        assert!((result.brier_score - 0.09).abs() < 1e-10);
        assert!(!result.actual_outcome);
    }

    #[tokio::test]
    async fn test_cannot_resolve_twice() {
        let db = test_db().await;
        let store = PredictionStore::new(db).await.unwrap();

        let prediction = Prediction::new("Test").with_probability(0.5);
        let id = store.store(&prediction).await.unwrap();

        // First resolution should succeed
        store.resolve(&id, true).await.unwrap();

        // Second resolution should fail
        let result = store.resolve(&id, false).await;
        assert!(matches!(result, Err(PredictionError::AlreadyResolved { .. })));
    }

    #[tokio::test]
    async fn test_list_pending_predictions() {
        let db = test_db().await;
        let store = PredictionStore::new(db).await.unwrap();

        // Create some predictions
        for i in 0..5 {
            let prediction = Prediction::new(format!("Prediction {}", i)).with_probability(0.5);
            store.store(&prediction).await.unwrap();
        }

        let pending = store.list_pending(10).await.unwrap();
        assert_eq!(pending.len(), 5);
    }

    #[tokio::test]
    async fn test_calibration_buckets() {
        let db = test_db().await;
        let store = PredictionStore::new(db).await.unwrap();

        let buckets = store.get_calibration_buckets().await.unwrap();
        assert_eq!(buckets.len(), 10);

        // Check bucket ranges
        assert!((buckets[0].lower_bound - 0.0).abs() < f64::EPSILON);
        assert!((buckets[0].upper_bound - 0.1).abs() < f64::EPSILON);
        assert!((buckets[9].lower_bound - 0.9).abs() < f64::EPSILON);
        assert!((buckets[9].upper_bound - 1.0).abs() < f64::EPSILON);
    }

    #[tokio::test]
    async fn test_calibration_bucket_updates() {
        let db = test_db().await;
        let store = PredictionStore::new(db).await.unwrap();

        // Create and resolve a prediction with probability 0.75 (bucket 7)
        let prediction = Prediction::new("Test").with_probability(0.75);
        let id = store.store(&prediction).await.unwrap();
        store.resolve(&id, true).await.unwrap();

        // Check bucket 7 was updated
        let buckets = store.get_calibration_buckets().await.unwrap();
        let bucket_7 = &buckets[7];

        assert_eq!(bucket_7.prediction_count, 1);
        assert_eq!(bucket_7.actual_positive_count, 1);
    }

    #[tokio::test]
    async fn test_resolution_result_quality() {
        let result = ResolutionResult::new("test".to_string(), 0.9, None, true);
        assert_eq!(result.quality(), "Excellent"); // Brier = 0.01

        let result = ResolutionResult::new("test".to_string(), 0.7, None, true);
        assert_eq!(result.quality(), "Excellent"); // Brier = 0.09

        let result = ResolutionResult::new("test".to_string(), 0.5, None, true);
        assert_eq!(result.quality(), "Fair"); // Brier = 0.25

        let result = ResolutionResult::new("test".to_string(), 0.2, None, true);
        assert_eq!(result.quality(), "Very Poor"); // Brier = 0.64
    }

    #[tokio::test]
    async fn test_average_brier_score() {
        let db = test_db().await;
        let store = PredictionStore::new(db).await.unwrap();

        // No resolved predictions yet
        let avg = store.average_brier_score().await.unwrap();
        assert!(avg.is_none());

        // Add some resolved predictions
        let p1 = Prediction::new("P1").with_probability(0.8);
        let id1 = store.store(&p1).await.unwrap();
        store.resolve(&id1, true).await.unwrap(); // Brier = 0.04

        let p2 = Prediction::new("P2").with_probability(0.6);
        let id2 = store.store(&p2).await.unwrap();
        store.resolve(&id2, true).await.unwrap(); // Brier = 0.16

        // Average should be (0.04 + 0.16) / 2 = 0.10
        let avg = store.average_brier_score().await.unwrap().unwrap();
        assert!((avg - 0.10).abs() < 1e-10);
    }

    #[tokio::test]
    async fn test_invalid_probability() {
        let db = test_db().await;
        let store = PredictionStore::new(db).await.unwrap();

        // Probability is clamped, so this should work
        let prediction = Prediction::new("Test").with_probability(1.5);
        assert!((prediction.probability - 1.0).abs() < f64::EPSILON);

        let prediction = Prediction::new("Test").with_probability(-0.5);
        assert!((prediction.probability - 0.0).abs() < f64::EPSILON);
    }
}
