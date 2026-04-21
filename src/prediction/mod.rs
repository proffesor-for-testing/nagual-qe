//! Prediction Engine Core for Nagual.
//!
//! This module implements a comprehensive prediction system with:
//! - Probabilistic predictions with Brier Score calibration
//! - Pattern-based probability calculation using Bayesian methods
//! - Timeline estimation with percentile-based analysis
//! - Evidence chain tracking for explainability
//!
//! # Architecture
//!
//! The prediction engine consists of several components:
//!
//! - **Prediction Storage**: Persistent storage for predictions and outcomes
//! - **Generator**: Creates predictions from patterns and context
//! - **Probability Calculator**: Computes probability using Bayesian methods
//! - **Timeline Estimator**: Estimates when predictions may resolve
//! - **Evidence Linker**: Connects predictions to supporting patterns
//! - **Calibration**: Tracks and adjusts for systematic biases
//!
//! # Brier Score
//!
//! The Brier score measures the accuracy of probabilistic predictions:
//! - Score = (predicted_probability - actual_outcome)^2
//! - Range: 0.0 (perfect) to 1.0 (worst)
//! - Lower is better
//!
//! # Example
//!
//! ```ignore
//! use nagual::prediction::{
//!     PredictionGenerator, PredictionStorage, GenerationContext, Prediction,
//! };
//!
//! // Create a prediction from patterns
//! let generator = PredictionGenerator::new();
//! let context = GenerationContext::new("Will the deployment succeed?")
//!     .with_domain("devops.deployment");
//!
//! let prediction = generator.generate_prediction(&patterns, &context)?;
//!
//! // Store the prediction
//! let storage = PredictionStorage::new(db).await?;
//! storage.store_prediction(&prediction).await?;
//!
//! // Later, resolve the prediction
//! storage.resolve_prediction(&prediction.id(), true).await?;
//!
//! // Get calibration report
//! let report = storage.get_calibration_report().await?;
//! println!("Overall Brier: {}", report.overall_brier_score);
//! ```

mod evidence;
mod generator;
mod probability;
mod storage;
mod timeline;

pub use evidence::{
    get_prediction_evidence, link_evidence, ContributionType, EvidenceLink, EvidenceSummary,
    PredictionEvidence,
};
pub use generator::{
    generate_prediction, GeneratedPrediction, GenerationContext, GenerationMetadata,
    GenerationResult, InputPattern, PatternAnalysis, PredictionGenerator,
};
pub use probability::{
    calculate_probability, PriorPrediction, ProbabilityBreakdown, ProbabilityCalculator,
    ProbabilityConfig, ProbabilityResult, WeightedPattern,
};
pub use storage::{PredictionFilter, PredictionStorage, PredictionUpdate, StorageStats};
pub use timeline::{
    estimate_timeline, ResolutionTimeAnalysis, TimelineConfig, TimelineEstimate, TimelineEstimator,
};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;
use uuid::Uuid;

/// Errors specific to prediction operations.
#[derive(Error, Debug)]
pub enum PredictionError {
    /// Database error during prediction operations
    #[error("Database error: {0}")]
    Database(String),

    /// Prediction not found
    #[error("Prediction not found: {id}")]
    NotFound { id: String },

    /// Invalid probability value
    #[error("Invalid probability: {value} (must be 0.0-1.0)")]
    InvalidProbability { value: f64 },

    /// Invalid confidence value
    #[error("Invalid confidence: {value} (must be 0.0-1.0)")]
    InvalidConfidence { value: f64 },

    /// Insufficient patterns for prediction
    #[error("Insufficient patterns: need at least {required}, found {found}")]
    InsufficientPatterns { required: usize, found: usize },

    /// Invalid timeline
    #[error("Invalid timeline: min_days ({min}) > max_days ({max})")]
    InvalidTimeline { min: u32, max: u32 },

    /// Prediction already resolved
    #[error("Prediction '{id}' already resolved at {resolved_at}")]
    AlreadyResolved {
        id: String,
        resolved_at: DateTime<Utc>,
    },

    /// Evidence not found
    #[error("Evidence not found for pattern: {pattern_id}")]
    EvidenceNotFound { pattern_id: String },

    /// No calibration data available
    #[error("No calibration data available")]
    NoCalibrationData,

    /// Serialization error
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    /// Calculation error
    #[error("Calculation error: {reason}")]
    CalculationError { reason: String },
}

/// Result type for prediction operations.
pub type PredictionResult<T> = std::result::Result<T, PredictionError>;

/// Status of a prediction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PredictionStatus {
    /// Prediction is pending resolution
    Pending,
    /// Prediction has been resolved (outcome known)
    Resolved,
    /// Prediction has expired without resolution
    Expired,
    /// Prediction was cancelled/invalidated
    Cancelled,
}

impl Default for PredictionStatus {
    fn default() -> Self {
        Self::Pending
    }
}

impl std::fmt::Display for PredictionStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PredictionStatus::Pending => write!(f, "pending"),
            PredictionStatus::Resolved => write!(f, "resolved"),
            PredictionStatus::Expired => write!(f, "expired"),
            PredictionStatus::Cancelled => write!(f, "cancelled"),
        }
    }
}

impl std::str::FromStr for PredictionStatus {
    type Err = PredictionError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "pending" => Ok(PredictionStatus::Pending),
            "resolved" => Ok(PredictionStatus::Resolved),
            "expired" => Ok(PredictionStatus::Expired),
            "cancelled" | "canceled" => Ok(PredictionStatus::Cancelled),
            _ => Err(PredictionError::CalculationError {
                reason: format!("Invalid status: {}", s),
            }),
        }
    }
}

/// Unique identifier for a prediction.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PredictionId(pub String);

impl PredictionId {
    /// Create a new random prediction ID.
    pub fn new() -> Self {
        Self(Uuid::new_v4().to_string())
    }

    /// Create a prediction ID from an existing string.
    pub fn from_string(s: impl Into<String>) -> Self {
        Self(s.into())
    }

    /// Get the underlying string.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Default for PredictionId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for PredictionId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<String> for PredictionId {
    fn from(s: String) -> Self {
        Self(s)
    }
}

impl From<&str> for PredictionId {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

/// A prediction about a future outcome.
///
/// Predictions are the core unit of the prediction engine. Each prediction
/// captures a probabilistic forecast with confidence bounds, timeline estimates,
/// and links to supporting evidence patterns.
///
/// # Fields
///
/// - `id`: Unique identifier (UUID)
/// - `created_at`: When the prediction was created
/// - `updated_at`: Last modification timestamp
/// - `description`: Human-readable description of what is being predicted
/// - `probability`: Estimated probability (0.0-1.0)
/// - `confidence`: Confidence in the probability estimate (0.0-1.0)
/// - `timeline_min_days`: Minimum expected days until resolution
/// - `timeline_max_days`: Maximum expected days until resolution
/// - `evidence_pattern_ids`: IDs of patterns supporting this prediction
/// - `status`: Current status (Pending, Resolved, Expired, Cancelled)
/// - `actual_outcome`: The actual outcome once resolved (true = happened)
/// - `resolved_at`: When the prediction was resolved
/// - `brier_score`: Calibration score (lower is better, 0 = perfect)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Prediction {
    /// Unique identifier for this prediction
    id: PredictionId,

    /// Timestamp when the prediction was created
    created_at: DateTime<Utc>,

    /// Last update timestamp
    updated_at: DateTime<Utc>,

    /// Human-readable description of what is being predicted
    description: String,

    /// Estimated probability of the predicted outcome (0.0-1.0)
    probability: f64,

    /// Calibrated probability after adjusting for historical biases
    #[serde(skip_serializing_if = "Option::is_none")]
    calibrated_probability: Option<f64>,

    /// Confidence in the probability estimate (0.0-1.0)
    /// Higher confidence means narrower uncertainty bounds
    confidence: f64,

    /// Minimum expected days until resolution
    timeline_min_days: u32,

    /// Maximum expected days until resolution
    timeline_max_days: u32,

    /// IDs of patterns that support this prediction
    evidence_pattern_ids: Vec<String>,

    /// Current status of the prediction
    status: PredictionStatus,

    /// The actual outcome once resolved (true = event happened, false = did not happen)
    actual_outcome: Option<bool>,

    /// When the prediction was resolved
    resolved_at: Option<DateTime<Utc>>,

    /// Brier score for calibration (0 = perfect, 1 = worst)
    /// Computed as (probability - actual)^2
    brier_score: Option<f64>,

    /// Domain/category for the prediction
    #[serde(default)]
    domain: String,

    /// Additional context about the prediction
    #[serde(default)]
    context: String,

    /// Tags for categorization
    #[serde(default)]
    tags: Vec<String>,

    /// Session ID that created this prediction
    #[serde(skip_serializing_if = "Option::is_none")]
    session_id: Option<String>,

    /// Agent ID that created this prediction
    #[serde(skip_serializing_if = "Option::is_none")]
    agent_id: Option<String>,

    /// Additional metadata
    #[serde(default)]
    metadata: HashMap<String, serde_json::Value>,
}

impl Prediction {
    /// Create a new prediction builder.
    pub fn builder() -> PredictionBuilder {
        PredictionBuilder::new()
    }

    /// Create a new prediction with required fields.
    pub fn new(description: impl Into<String>, probability: f64) -> PredictionResult<Self> {
        Self::builder()
            .description(description)
            .probability(probability)
            .build()
    }

    // Getters

    /// Get the prediction ID.
    pub fn id(&self) -> &PredictionId {
        &self.id
    }

    /// Get the creation timestamp.
    pub fn created_at(&self) -> DateTime<Utc> {
        self.created_at
    }

    /// Get the last update timestamp.
    pub fn updated_at(&self) -> DateTime<Utc> {
        self.updated_at
    }

    /// Get the description.
    pub fn description(&self) -> &str {
        &self.description
    }

    /// Get the probability.
    pub fn probability(&self) -> f64 {
        self.probability
    }

    /// Get the calibrated probability if available.
    pub fn calibrated_probability(&self) -> Option<f64> {
        self.calibrated_probability
    }

    /// Get the effective probability (calibrated if available, otherwise raw).
    pub fn effective_probability(&self) -> f64 {
        self.calibrated_probability.unwrap_or(self.probability)
    }

    /// Get the confidence.
    pub fn confidence(&self) -> f64 {
        self.confidence
    }

    /// Get the minimum timeline in days.
    pub fn timeline_min_days(&self) -> u32 {
        self.timeline_min_days
    }

    /// Get the maximum timeline in days.
    pub fn timeline_max_days(&self) -> u32 {
        self.timeline_max_days
    }

    /// Get the evidence pattern IDs.
    pub fn evidence_pattern_ids(&self) -> &[String] {
        &self.evidence_pattern_ids
    }

    /// Get the status.
    pub fn status(&self) -> PredictionStatus {
        self.status
    }

    /// Get the actual outcome if resolved.
    pub fn actual_outcome(&self) -> Option<bool> {
        self.actual_outcome
    }

    /// Get when the prediction was resolved.
    pub fn resolved_at(&self) -> Option<DateTime<Utc>> {
        self.resolved_at
    }

    /// Get the Brier score if resolved.
    pub fn brier_score(&self) -> Option<f64> {
        self.brier_score
    }

    /// Get the domain.
    pub fn domain(&self) -> &str {
        &self.domain
    }

    /// Get the context.
    pub fn context(&self) -> &str {
        &self.context
    }

    /// Get the tags.
    pub fn tags(&self) -> &[String] {
        &self.tags
    }

    /// Get the session ID.
    pub fn session_id(&self) -> Option<&str> {
        self.session_id.as_deref()
    }

    /// Get the agent ID.
    pub fn agent_id(&self) -> Option<&str> {
        self.agent_id.as_deref()
    }

    /// Get the metadata.
    pub fn metadata(&self) -> &HashMap<String, serde_json::Value> {
        &self.metadata
    }

    // Setters

    /// Set the status.
    pub fn set_status(&mut self, status: PredictionStatus) {
        self.status = status;
        self.updated_at = Utc::now();
    }

    /// Set the calibrated probability.
    pub fn set_calibrated_probability(&mut self, probability: f64) {
        self.calibrated_probability = Some(probability.clamp(0.0, 1.0));
        self.updated_at = Utc::now();
    }

    /// Add an evidence pattern ID.
    pub fn add_evidence_pattern(&mut self, pattern_id: impl Into<String>) {
        let id = pattern_id.into();
        if !self.evidence_pattern_ids.contains(&id) {
            self.evidence_pattern_ids.push(id);
            self.updated_at = Utc::now();
        }
    }

    /// Add a tag.
    pub fn add_tag(&mut self, tag: impl Into<String>) {
        self.tags.push(tag.into());
        self.updated_at = Utc::now();
    }

    /// Set metadata value.
    pub fn set_metadata(&mut self, key: impl Into<String>, value: serde_json::Value) {
        self.metadata.insert(key.into(), value);
        self.updated_at = Utc::now();
    }

    /// Touch the updated_at timestamp.
    pub fn touch(&mut self) {
        self.updated_at = Utc::now();
    }

    // Resolution methods

    /// Resolve the prediction with an actual outcome.
    ///
    /// Calculates the Brier score automatically using the effective probability.
    pub fn resolve(&mut self, actual_outcome: bool) -> PredictionResult<()> {
        if self.status == PredictionStatus::Resolved {
            return Err(PredictionError::AlreadyResolved {
                id: self.id.to_string(),
                resolved_at: self.resolved_at.unwrap_or_else(Utc::now),
            });
        }

        let now = Utc::now();
        self.actual_outcome = Some(actual_outcome);
        self.resolved_at = Some(now);
        self.status = PredictionStatus::Resolved;
        self.updated_at = now;

        // Calculate Brier score using effective probability
        self.brier_score = Some(calculate_brier_score(
            self.effective_probability(),
            actual_outcome,
        ));

        Ok(())
    }

    /// Mark the prediction as expired.
    pub fn expire(&mut self) {
        if self.status == PredictionStatus::Pending {
            self.status = PredictionStatus::Expired;
            self.updated_at = Utc::now();
        }
    }

    /// Cancel the prediction.
    pub fn cancel(&mut self) {
        if self.status == PredictionStatus::Pending {
            self.status = PredictionStatus::Cancelled;
            self.updated_at = Utc::now();
        }
    }

    // Computed properties

    /// Check if the prediction is still pending.
    pub fn is_pending(&self) -> bool {
        self.status == PredictionStatus::Pending
    }

    /// Check if the prediction has been resolved.
    pub fn is_resolved(&self) -> bool {
        self.status == PredictionStatus::Resolved
    }

    /// Check if the prediction was correct (if resolved).
    pub fn is_correct(&self) -> Option<bool> {
        self.actual_outcome.map(|outcome| {
            let prob = self.effective_probability();
            // Consider correct if probability > 0.5 and outcome is true,
            // or probability < 0.5 and outcome is false
            (prob > 0.5 && outcome) || (prob < 0.5 && !outcome)
        })
    }

    /// Get the age of the prediction in days.
    pub fn age_days(&self) -> i64 {
        (Utc::now() - self.created_at).num_days()
    }

    /// Check if the prediction should be expired based on timeline.
    pub fn should_expire(&self) -> bool {
        self.is_pending() && self.age_days() as u32 > self.timeline_max_days
    }

    /// Get the expected timeline midpoint in days.
    pub fn timeline_midpoint(&self) -> u32 {
        (self.timeline_min_days + self.timeline_max_days) / 2
    }

    /// Get the timeline range.
    pub fn timeline_range(&self) -> (u32, u32) {
        (self.timeline_min_days, self.timeline_max_days)
    }

    /// Get the uncertainty interval for the probability.
    /// Returns (lower_bound, upper_bound) based on confidence.
    pub fn probability_interval(&self) -> (f64, f64) {
        // Higher confidence = narrower interval
        let half_width = (1.0 - self.confidence) * 0.5;
        let prob = self.effective_probability();
        let lower = (prob - half_width).max(0.0);
        let upper = (prob + half_width).min(1.0);
        (lower, upper)
    }

    /// Get the evidence count.
    pub fn evidence_count(&self) -> usize {
        self.evidence_pattern_ids.len()
    }
}

impl Default for Prediction {
    fn default() -> Self {
        Self::builder().build().unwrap_or_else(|_| {
            // Provide a minimal default
            let now = Utc::now();
            Self {
                id: PredictionId::new(),
                created_at: now,
                updated_at: now,
                description: String::new(),
                probability: 0.5,
                calibrated_probability: None,
                confidence: 0.5,
                timeline_min_days: 1,
                timeline_max_days: 30,
                evidence_pattern_ids: Vec::new(),
                status: PredictionStatus::Pending,
                actual_outcome: None,
                resolved_at: None,
                brier_score: None,
                domain: String::new(),
                context: String::new(),
                tags: Vec::new(),
                session_id: None,
                agent_id: None,
                metadata: HashMap::new(),
            }
        })
    }
}

/// Builder for creating Prediction instances.
#[derive(Debug, Default)]
pub struct PredictionBuilder {
    id: Option<PredictionId>,
    created_at: Option<DateTime<Utc>>,
    description: Option<String>,
    probability: Option<f64>,
    calibrated_probability: Option<f64>,
    confidence: Option<f64>,
    timeline_min_days: Option<u32>,
    timeline_max_days: Option<u32>,
    evidence_pattern_ids: Option<Vec<String>>,
    status: Option<PredictionStatus>,
    actual_outcome: Option<bool>,
    resolved_at: Option<DateTime<Utc>>,
    brier_score: Option<f64>,
    domain: Option<String>,
    context: Option<String>,
    tags: Option<Vec<String>>,
    session_id: Option<String>,
    agent_id: Option<String>,
    metadata: Option<HashMap<String, serde_json::Value>>,
}

impl PredictionBuilder {
    /// Create a new prediction builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the prediction ID.
    pub fn id(mut self, id: impl Into<PredictionId>) -> Self {
        self.id = Some(id.into());
        self
    }

    /// Set the creation timestamp.
    pub fn created_at(mut self, created_at: DateTime<Utc>) -> Self {
        self.created_at = Some(created_at);
        self
    }

    /// Set the description.
    pub fn description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Set the probability.
    pub fn probability(mut self, probability: f64) -> Self {
        self.probability = Some(probability.clamp(0.0, 1.0));
        self
    }

    /// Set the calibrated probability.
    pub fn calibrated_probability(mut self, probability: f64) -> Self {
        self.calibrated_probability = Some(probability.clamp(0.0, 1.0));
        self
    }

    /// Set the confidence.
    pub fn confidence(mut self, confidence: f64) -> Self {
        self.confidence = Some(confidence.clamp(0.0, 1.0));
        self
    }

    /// Set the minimum timeline in days.
    pub fn timeline_min_days(mut self, days: u32) -> Self {
        self.timeline_min_days = Some(days);
        self
    }

    /// Set the maximum timeline in days.
    pub fn timeline_max_days(mut self, days: u32) -> Self {
        self.timeline_max_days = Some(days);
        self
    }

    /// Set both timeline bounds.
    pub fn timeline(mut self, min_days: u32, max_days: u32) -> Self {
        self.timeline_min_days = Some(min_days);
        self.timeline_max_days = Some(max_days);
        self
    }

    /// Set the evidence pattern IDs.
    pub fn evidence_pattern_ids(mut self, ids: Vec<String>) -> Self {
        self.evidence_pattern_ids = Some(ids);
        self
    }

    /// Add a single evidence pattern ID.
    pub fn evidence_pattern(mut self, id: impl Into<String>) -> Self {
        self.evidence_pattern_ids
            .get_or_insert_with(Vec::new)
            .push(id.into());
        self
    }

    /// Set the status.
    pub fn status(mut self, status: PredictionStatus) -> Self {
        self.status = Some(status);
        self
    }

    /// Set the actual outcome.
    pub fn actual_outcome(mut self, outcome: bool) -> Self {
        self.actual_outcome = Some(outcome);
        self
    }

    /// Set the resolved timestamp.
    pub fn resolved_at(mut self, resolved_at: DateTime<Utc>) -> Self {
        self.resolved_at = Some(resolved_at);
        self
    }

    /// Set the Brier score.
    pub fn brier_score(mut self, score: f64) -> Self {
        self.brier_score = Some(score.clamp(0.0, 1.0));
        self
    }

    /// Set the domain.
    pub fn domain(mut self, domain: impl Into<String>) -> Self {
        self.domain = Some(domain.into());
        self
    }

    /// Set the context.
    pub fn context(mut self, context: impl Into<String>) -> Self {
        self.context = Some(context.into());
        self
    }

    /// Set the tags.
    pub fn tags(mut self, tags: Vec<String>) -> Self {
        self.tags = Some(tags);
        self
    }

    /// Add a single tag.
    pub fn tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.get_or_insert_with(Vec::new).push(tag.into());
        self
    }

    /// Set the session ID.
    pub fn session_id(mut self, session_id: impl Into<String>) -> Self {
        self.session_id = Some(session_id.into());
        self
    }

    /// Set the agent ID.
    pub fn agent_id(mut self, agent_id: impl Into<String>) -> Self {
        self.agent_id = Some(agent_id.into());
        self
    }

    /// Set the metadata.
    pub fn metadata(mut self, metadata: HashMap<String, serde_json::Value>) -> Self {
        self.metadata = Some(metadata);
        self
    }

    /// Add a single metadata entry.
    pub fn meta(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.metadata
            .get_or_insert_with(HashMap::new)
            .insert(key.into(), value);
        self
    }

    /// Build the prediction.
    pub fn build(self) -> PredictionResult<Prediction> {
        let now = Utc::now();
        let probability = self.probability.unwrap_or(0.5);
        let confidence = self.confidence.unwrap_or(0.5);
        let timeline_min = self.timeline_min_days.unwrap_or(1);
        let timeline_max = self.timeline_max_days.unwrap_or(30);

        // Validate probability
        if !(0.0..=1.0).contains(&probability) {
            return Err(PredictionError::InvalidProbability { value: probability });
        }

        // Validate confidence
        if !(0.0..=1.0).contains(&confidence) {
            return Err(PredictionError::InvalidConfidence { value: confidence });
        }

        // Validate timeline
        if timeline_min > timeline_max {
            return Err(PredictionError::InvalidTimeline {
                min: timeline_min,
                max: timeline_max,
            });
        }

        Ok(Prediction {
            id: self.id.unwrap_or_else(PredictionId::new),
            created_at: self.created_at.unwrap_or(now),
            updated_at: now,
            description: self.description.unwrap_or_default(),
            probability,
            calibrated_probability: self.calibrated_probability,
            confidence,
            timeline_min_days: timeline_min,
            timeline_max_days: timeline_max,
            evidence_pattern_ids: self.evidence_pattern_ids.unwrap_or_default(),
            status: self.status.unwrap_or_default(),
            actual_outcome: self.actual_outcome,
            resolved_at: self.resolved_at,
            brier_score: self.brier_score,
            domain: self.domain.unwrap_or_default(),
            context: self.context.unwrap_or_default(),
            tags: self.tags.unwrap_or_default(),
            session_id: self.session_id,
            agent_id: self.agent_id,
            metadata: self.metadata.unwrap_or_default(),
        })
    }
}

// ============================================================================
// SQL Table Definitions
// ============================================================================

/// SQL for creating the predictions table in SQLite.
pub const SQLITE_PREDICTIONS_TABLE: &str = r#"
CREATE TABLE IF NOT EXISTS predictions (
    id TEXT PRIMARY KEY,
    description TEXT NOT NULL,
    probability REAL NOT NULL,
    calibrated_probability REAL,
    confidence REAL NOT NULL DEFAULT 0.5,
    timeline_min_days INTEGER NOT NULL DEFAULT 1,
    timeline_max_days INTEGER NOT NULL DEFAULT 30,
    status TEXT NOT NULL DEFAULT 'pending',
    actual_outcome INTEGER,
    brier_score REAL,
    domain TEXT DEFAULT 'general',
    context TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    resolved_at TEXT,
    session_id TEXT,
    agent_id TEXT,
    tags TEXT DEFAULT '[]',
    metadata TEXT DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_predictions_status ON predictions(status);
CREATE INDEX IF NOT EXISTS idx_predictions_domain ON predictions(domain);
CREATE INDEX IF NOT EXISTS idx_predictions_created_at ON predictions(created_at);
CREATE INDEX IF NOT EXISTS idx_predictions_probability ON predictions(probability);
CREATE INDEX IF NOT EXISTS idx_predictions_timeline ON predictions(timeline_min_days, timeline_max_days);
"#;

/// SQL for creating the prediction_evidence junction table.
pub const SQLITE_PREDICTION_EVIDENCE_TABLE: &str = r#"
CREATE TABLE IF NOT EXISTS prediction_evidence (
    id TEXT PRIMARY KEY,
    prediction_id TEXT NOT NULL,
    pattern_id TEXT NOT NULL,
    relevance_score REAL NOT NULL DEFAULT 1.0,
    contribution_type TEXT DEFAULT 'supporting',
    created_at TEXT NOT NULL,
    FOREIGN KEY (prediction_id) REFERENCES predictions(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_prediction_evidence_prediction ON prediction_evidence(prediction_id);
CREATE INDEX IF NOT EXISTS idx_prediction_evidence_pattern ON prediction_evidence(pattern_id);
CREATE INDEX IF NOT EXISTS idx_prediction_evidence_relevance ON prediction_evidence(relevance_score);
"#;

/// SQL for creating the calibration_buckets table in SQLite.
pub const SQLITE_CALIBRATION_BUCKETS_TABLE: &str = r#"
CREATE TABLE IF NOT EXISTS calibration_buckets (
    bucket_id TEXT PRIMARY KEY,
    lower_bound REAL NOT NULL,
    upper_bound REAL NOT NULL,
    prediction_count INTEGER NOT NULL DEFAULT 0,
    actual_positive_count INTEGER NOT NULL DEFAULT 0,
    total_brier_score REAL NOT NULL DEFAULT 0.0,
    domain TEXT DEFAULT 'general',
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_calibration_buckets_domain ON calibration_buckets(domain);
"#;

// ============================================================================
// Calibration Bucket Types
// ============================================================================

/// A calibration bucket for tracking prediction accuracy in a probability range.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationBucket {
    /// Unique identifier for the bucket
    pub id: String,
    /// Lower bound of the probability range (inclusive)
    pub lower_bound: f64,
    /// Upper bound of the probability range (exclusive, except for 1.0)
    pub upper_bound: f64,
    /// Number of predictions in this bucket
    pub prediction_count: u32,
    /// Number of predictions where the outcome was positive
    pub actual_positive_count: u32,
    /// Sum of Brier scores for predictions in this bucket
    pub total_brier_score: f64,
    /// Domain this bucket belongs to
    pub domain: String,
    /// Last update timestamp
    pub updated_at: DateTime<Utc>,
}

impl CalibrationBucket {
    /// Create a new calibration bucket for a probability range.
    pub fn new(lower_bound: f64, upper_bound: f64) -> Self {
        Self {
            id: format!("{:.1}-{:.1}", lower_bound, upper_bound),
            lower_bound,
            upper_bound,
            prediction_count: 0,
            actual_positive_count: 0,
            total_brier_score: 0.0,
            domain: "general".to_string(),
            updated_at: Utc::now(),
        }
    }

    /// Create a bucket for a specific domain.
    pub fn with_domain(mut self, domain: impl Into<String>) -> Self {
        self.domain = domain.into();
        self.id = format!("{}-{:.1}-{:.1}", self.domain, self.lower_bound, self.upper_bound);
        self
    }

    /// Check if a probability falls within this bucket.
    pub fn contains(&self, probability: f64) -> bool {
        probability >= self.lower_bound && probability < self.upper_bound
            || (probability == 1.0 && self.upper_bound == 1.0)
    }

    /// Update the bucket with a resolved prediction.
    pub fn update(&mut self, probability: f64, outcome: bool) {
        self.prediction_count += 1;
        if outcome {
            self.actual_positive_count += 1;
        }
        self.total_brier_score += calculate_brier_score(probability, outcome);
        self.updated_at = Utc::now();
    }

    /// Get the actual positive rate (frequency of positive outcomes).
    pub fn actual_rate(&self) -> f64 {
        if self.prediction_count == 0 {
            return 0.0;
        }
        self.actual_positive_count as f64 / self.prediction_count as f64
    }

    /// Get the expected probability (midpoint of the range).
    pub fn expected_probability(&self) -> f64 {
        (self.lower_bound + self.upper_bound) / 2.0
    }

    /// Get the average Brier score for this bucket.
    pub fn avg_brier_score(&self) -> f64 {
        if self.prediction_count == 0 {
            return 0.0;
        }
        self.total_brier_score / self.prediction_count as f64
    }

    /// Get the calibration error (difference between expected and actual).
    pub fn calibration_error(&self) -> f64 {
        (self.expected_probability() - self.actual_rate()).abs()
    }
}

// ============================================================================
// Utility Functions
// ============================================================================

/// Initialize the default calibration buckets (0.0-0.1, 0.1-0.2, ..., 0.9-1.0).
pub fn init_default_buckets() -> Vec<CalibrationBucket> {
    (0..10)
        .map(|i| {
            let lower = i as f64 * 0.1;
            let upper = (i + 1) as f64 * 0.1;
            CalibrationBucket::new(lower, upper)
        })
        .collect()
}

/// Get the bucket index for a given probability (0-9).
pub fn bucket_index_for_probability(probability: f64) -> usize {
    let idx = (probability * 10.0).floor() as usize;
    idx.min(9) // Clamp to 9 for probability == 1.0
}

/// Calculate Brier score for a single prediction.
///
/// Brier score = (predicted_probability - actual_outcome)^2
///
/// # Arguments
///
/// * `probability` - The predicted probability (0.0 to 1.0)
/// * `outcome` - The actual outcome (true = 1.0, false = 0.0)
///
/// # Returns
///
/// The Brier score (0.0 = perfect, 1.0 = worst)
pub fn calculate_brier_score(probability: f64, outcome: bool) -> f64 {
    let outcome_val = if outcome { 1.0 } else { 0.0 };
    (probability - outcome_val).powi(2)
}

/// Summary statistics for a set of predictions.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PredictionStats {
    /// Total number of predictions
    pub total: usize,
    /// Number of pending predictions
    pub pending: usize,
    /// Number of resolved predictions
    pub resolved: usize,
    /// Number of expired predictions
    pub expired: usize,
    /// Number of cancelled predictions
    pub cancelled: usize,
    /// Number of correct predictions (among resolved)
    pub correct: usize,
    /// Accuracy rate (correct / resolved)
    pub accuracy: f64,
    /// Average Brier score (lower is better)
    pub avg_brier_score: f64,
    /// Average confidence
    pub avg_confidence: f64,
    /// Average evidence count per prediction
    pub avg_evidence_count: f64,
}

impl PredictionStats {
    /// Calculate statistics from a slice of predictions.
    pub fn from_predictions(predictions: &[Prediction]) -> Self {
        if predictions.is_empty() {
            return Self::default();
        }

        let total = predictions.len();
        let mut pending = 0;
        let mut resolved = 0;
        let mut expired = 0;
        let mut cancelled = 0;
        let mut correct = 0;
        let mut brier_sum = 0.0;
        let mut brier_count = 0;
        let mut confidence_sum = 0.0;
        let mut evidence_count_sum = 0;

        for p in predictions {
            match p.status {
                PredictionStatus::Pending => pending += 1,
                PredictionStatus::Resolved => {
                    resolved += 1;
                    if p.is_correct().unwrap_or(false) {
                        correct += 1;
                    }
                    if let Some(brier) = p.brier_score {
                        brier_sum += brier;
                        brier_count += 1;
                    }
                }
                PredictionStatus::Expired => expired += 1,
                PredictionStatus::Cancelled => cancelled += 1,
            }
            confidence_sum += p.confidence;
            evidence_count_sum += p.evidence_count();
        }

        let accuracy = if resolved > 0 {
            correct as f64 / resolved as f64
        } else {
            0.0
        };

        let avg_brier_score = if brier_count > 0 {
            brier_sum / brier_count as f64
        } else {
            0.0
        };

        Self {
            total,
            pending,
            resolved,
            expired,
            cancelled,
            correct,
            accuracy,
            avg_brier_score,
            avg_confidence: confidence_sum / total as f64,
            avg_evidence_count: evidence_count_sum as f64 / total as f64,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_brier_score_perfect_prediction() {
        // Predicted 1.0, outcome true -> score 0.0
        assert!((calculate_brier_score(1.0, true) - 0.0).abs() < f64::EPSILON);

        // Predicted 0.0, outcome false -> score 0.0
        assert!((calculate_brier_score(0.0, false) - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_brier_score_worst_prediction() {
        // Predicted 1.0, outcome false -> score 1.0
        assert!((calculate_brier_score(1.0, false) - 1.0).abs() < f64::EPSILON);

        // Predicted 0.0, outcome true -> score 1.0
        assert!((calculate_brier_score(0.0, true) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_brier_score_uncertain_prediction() {
        // Predicted 0.5 -> score 0.25 regardless of outcome
        assert!((calculate_brier_score(0.5, true) - 0.25).abs() < f64::EPSILON);
        assert!((calculate_brier_score(0.5, false) - 0.25).abs() < f64::EPSILON);
    }

    #[test]
    fn test_brier_score_typical_prediction() {
        // Predicted 0.7, outcome true -> (0.7 - 1.0)^2 = 0.09
        assert!((calculate_brier_score(0.7, true) - 0.09).abs() < 1e-10);

        // Predicted 0.7, outcome false -> (0.7 - 0.0)^2 = 0.49
        assert!((calculate_brier_score(0.7, false) - 0.49).abs() < 1e-10);
    }

    #[test]
    fn test_bucket_index() {
        assert_eq!(bucket_index_for_probability(0.0), 0);
        assert_eq!(bucket_index_for_probability(0.05), 0);
        assert_eq!(bucket_index_for_probability(0.1), 1);
        assert_eq!(bucket_index_for_probability(0.15), 1);
        assert_eq!(bucket_index_for_probability(0.5), 5);
        assert_eq!(bucket_index_for_probability(0.95), 9);
        assert_eq!(bucket_index_for_probability(1.0), 9); // Edge case
    }

    #[test]
    fn test_prediction_id() {
        let id1 = PredictionId::new();
        let id2 = PredictionId::new();
        assert_ne!(id1, id2);

        let id3 = PredictionId::from_string("test-123");
        assert_eq!(id3.as_str(), "test-123");
    }

    #[test]
    fn test_prediction_status_display() {
        assert_eq!(PredictionStatus::Pending.to_string(), "pending");
        assert_eq!(PredictionStatus::Resolved.to_string(), "resolved");
        assert_eq!(PredictionStatus::Expired.to_string(), "expired");
        assert_eq!(PredictionStatus::Cancelled.to_string(), "cancelled");
    }

    #[test]
    fn test_prediction_status_parse() {
        assert_eq!(
            "pending".parse::<PredictionStatus>().unwrap(),
            PredictionStatus::Pending
        );
        assert_eq!(
            "RESOLVED".parse::<PredictionStatus>().unwrap(),
            PredictionStatus::Resolved
        );
    }

    #[test]
    fn test_prediction_builder_minimal() {
        let prediction = Prediction::builder()
            .description("Test prediction")
            .probability(0.75)
            .build()
            .unwrap();

        assert_eq!(prediction.description(), "Test prediction");
        assert!((prediction.probability() - 0.75).abs() < 0.001);
        assert!(prediction.is_pending());
    }

    #[test]
    fn test_prediction_builder_full() {
        let prediction = Prediction::builder()
            .id("pred-123")
            .description("Deployment will succeed")
            .probability(0.85)
            .confidence(0.9)
            .timeline(7, 14)
            .evidence_pattern("pattern-1")
            .evidence_pattern("pattern-2")
            .domain("devops.deployment")
            .context("Production deployment")
            .tag("deployment")
            .tag("critical")
            .session_id("session-1")
            .agent_id("agent-1")
            .meta("version", serde_json::json!("1.0"))
            .build()
            .unwrap();

        assert_eq!(prediction.id().as_str(), "pred-123");
        assert_eq!(prediction.description(), "Deployment will succeed");
        assert!((prediction.probability() - 0.85).abs() < 0.001);
        assert!((prediction.confidence() - 0.9).abs() < 0.001);
        assert_eq!(prediction.timeline_min_days(), 7);
        assert_eq!(prediction.timeline_max_days(), 14);
        assert_eq!(prediction.evidence_pattern_ids().len(), 2);
        assert_eq!(prediction.domain(), "devops.deployment");
        assert_eq!(prediction.tags().len(), 2);
    }

    #[test]
    fn test_prediction_resolution() {
        let mut prediction = Prediction::new("Test", 0.8).unwrap();

        assert!(prediction.is_pending());
        assert!(prediction.actual_outcome().is_none());
        assert!(prediction.brier_score().is_none());

        // Resolve with success
        prediction.resolve(true).unwrap();

        assert!(prediction.is_resolved());
        assert_eq!(prediction.actual_outcome(), Some(true));
        assert!(prediction.brier_score().is_some());

        // Brier score should be (0.8 - 1.0)^2 = 0.04
        let brier = prediction.brier_score().unwrap();
        assert!((brier - 0.04).abs() < 0.001);

        // Should be marked as correct
        assert_eq!(prediction.is_correct(), Some(true));
    }

    #[test]
    fn test_prediction_resolution_failure() {
        let mut prediction = Prediction::new("Test", 0.8).unwrap();
        prediction.resolve(false).unwrap();

        // Brier score should be (0.8 - 0.0)^2 = 0.64
        let brier = prediction.brier_score().unwrap();
        assert!((brier - 0.64).abs() < 0.001);

        // Should be marked as incorrect (predicted high, got false)
        assert_eq!(prediction.is_correct(), Some(false));
    }

    #[test]
    fn test_prediction_double_resolve_error() {
        let mut prediction = Prediction::new("Test", 0.5).unwrap();
        prediction.resolve(true).unwrap();

        let result = prediction.resolve(false);
        assert!(matches!(result, Err(PredictionError::AlreadyResolved { .. })));
    }

    #[test]
    fn test_prediction_probability_clamping() {
        let prediction = Prediction::builder()
            .probability(1.5) // Should clamp to 1.0
            .confidence(-0.5) // Should clamp to 0.0
            .build()
            .unwrap();

        assert!((prediction.probability() - 1.0).abs() < 0.001);
        assert!((prediction.confidence() - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_prediction_invalid_timeline() {
        let result = Prediction::builder()
            .timeline_min_days(30)
            .timeline_max_days(7)
            .build();

        assert!(matches!(result, Err(PredictionError::InvalidTimeline { .. })));
    }

    #[test]
    fn test_prediction_probability_interval() {
        let prediction = Prediction::builder()
            .probability(0.7)
            .confidence(0.8) // High confidence = narrow interval
            .build()
            .unwrap();

        let (lower, upper) = prediction.probability_interval();
        assert!(lower < prediction.probability());
        assert!(upper > prediction.probability());
        assert!(lower >= 0.0);
        assert!(upper <= 1.0);
    }

    #[test]
    fn test_prediction_expire_and_cancel() {
        let mut prediction = Prediction::new("Test", 0.5).unwrap();

        prediction.expire();
        assert_eq!(prediction.status(), PredictionStatus::Expired);

        // Can't cancel after expiring
        prediction.cancel();
        assert_eq!(prediction.status(), PredictionStatus::Expired);
    }

    #[test]
    fn test_calibration_bucket() {
        let mut bucket = CalibrationBucket::new(0.7, 0.8);

        assert!(bucket.contains(0.75));
        assert!(!bucket.contains(0.69));
        assert!(!bucket.contains(0.8)); // Upper bound is exclusive

        bucket.update(0.75, true);
        bucket.update(0.72, false);

        assert_eq!(bucket.prediction_count, 2);
        assert_eq!(bucket.actual_positive_count, 1);
        assert!((bucket.actual_rate() - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_init_default_buckets() {
        let buckets = init_default_buckets();
        assert_eq!(buckets.len(), 10);

        // Check first bucket
        assert!((buckets[0].lower_bound - 0.0).abs() < f64::EPSILON);
        assert!((buckets[0].upper_bound - 0.1).abs() < f64::EPSILON);

        // Check last bucket
        assert!((buckets[9].lower_bound - 0.9).abs() < f64::EPSILON);
        assert!((buckets[9].upper_bound - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_prediction_stats() {
        let predictions = vec![
            {
                let mut p = Prediction::new("P1", 0.8).unwrap();
                p.resolve(true).unwrap();
                p
            },
            {
                let mut p = Prediction::new("P2", 0.3).unwrap();
                p.resolve(false).unwrap();
                p
            },
            Prediction::new("P3", 0.5).unwrap(), // Pending
        ];

        let stats = PredictionStats::from_predictions(&predictions);

        assert_eq!(stats.total, 3);
        assert_eq!(stats.pending, 1);
        assert_eq!(stats.resolved, 2);
        assert_eq!(stats.correct, 2); // Both predictions were correct
        assert!((stats.accuracy - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_prediction_serialization() {
        let prediction = Prediction::builder()
            .description("Test prediction")
            .probability(0.75)
            .domain("testing")
            .tag("important")
            .build()
            .unwrap();

        let json = serde_json::to_string(&prediction).unwrap();
        let deserialized: Prediction = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.description(), prediction.description());
        assert!((deserialized.probability() - prediction.probability()).abs() < 0.001);
        assert_eq!(deserialized.domain(), prediction.domain());
    }
}
