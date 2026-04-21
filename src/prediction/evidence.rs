//! Evidence linking for predictions.
//!
//! This module provides functionality for linking predictions to supporting
//! evidence patterns, tracking relevance scores, and generating evidence summaries.

use std::sync::Arc;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tracing::debug;
use uuid::Uuid;

use super::{PredictionError, PredictionId, PredictionResult};
use crate::db::SqliteDb;

/// A link between a prediction and a supporting pattern.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvidenceLink {
    /// Unique identifier for this link
    pub id: String,
    /// The prediction this evidence supports
    pub prediction_id: PredictionId,
    /// The pattern providing evidence
    pub pattern_id: String,
    /// How relevant this evidence is (0.0 to 1.0)
    pub relevance_score: f64,
    /// Type of contribution (supporting, contradicting, neutral)
    pub contribution_type: ContributionType,
    /// When this link was created
    pub created_at: DateTime<Utc>,
}

impl EvidenceLink {
    /// Create a new evidence link.
    pub fn new(prediction_id: PredictionId, pattern_id: impl Into<String>) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            prediction_id,
            pattern_id: pattern_id.into(),
            relevance_score: 1.0,
            contribution_type: ContributionType::Supporting,
            created_at: Utc::now(),
        }
    }

    /// Create an evidence link without a prediction ID (for deferred linking).
    /// The prediction ID will be set to a placeholder.
    pub fn for_pattern(pattern_id: impl Into<String>, relevance_score: f64) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            prediction_id: PredictionId::from_string("pending"),
            pattern_id: pattern_id.into(),
            relevance_score: relevance_score.clamp(0.0, 1.0),
            contribution_type: ContributionType::Supporting,
            created_at: Utc::now(),
        }
    }

    /// Set the prediction ID.
    pub fn with_prediction_id(mut self, prediction_id: PredictionId) -> Self {
        self.prediction_id = prediction_id;
        self
    }

    /// Set the relevance score.
    pub fn with_relevance(mut self, score: f64) -> Self {
        self.relevance_score = score.clamp(0.0, 1.0);
        self
    }

    /// Set the contribution type.
    pub fn with_contribution_type(mut self, contribution_type: ContributionType) -> Self {
        self.contribution_type = contribution_type;
        self
    }
}

/// Type of contribution an evidence pattern makes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ContributionType {
    /// Evidence supports the prediction
    Supporting,
    /// Evidence contradicts the prediction
    Contradicting,
    /// Evidence is neutral
    Neutral,
}

impl Default for ContributionType {
    fn default() -> Self {
        Self::Supporting
    }
}

impl std::fmt::Display for ContributionType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ContributionType::Supporting => write!(f, "supporting"),
            ContributionType::Contradicting => write!(f, "contradicting"),
            ContributionType::Neutral => write!(f, "neutral"),
        }
    }
}

impl std::str::FromStr for ContributionType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "supporting" => Ok(ContributionType::Supporting),
            "contradicting" => Ok(ContributionType::Contradicting),
            "neutral" => Ok(ContributionType::Neutral),
            _ => Err(format!("Unknown contribution type: {}", s)),
        }
    }
}

/// Evidence summary for a prediction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvidenceSummary {
    /// Total number of evidence patterns
    pub total_count: usize,
    /// Number of supporting patterns
    pub supporting_count: usize,
    /// Number of contradicting patterns
    pub contradicting_count: usize,
    /// Number of neutral patterns
    pub neutral_count: usize,
    /// Average relevance score
    pub avg_relevance: f64,
    /// Top pattern IDs by relevance
    pub top_patterns: Vec<String>,
}

impl Default for EvidenceSummary {
    fn default() -> Self {
        Self {
            total_count: 0,
            supporting_count: 0,
            contradicting_count: 0,
            neutral_count: 0,
            avg_relevance: 0.0,
            top_patterns: Vec::new(),
        }
    }
}

/// Full evidence information for a prediction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionEvidence {
    /// The prediction ID
    pub prediction_id: PredictionId,
    /// All evidence links
    pub links: Vec<EvidenceLink>,
    /// Summary statistics
    pub summary: EvidenceSummary,
}

impl PredictionEvidence {
    /// Create from a list of evidence links.
    pub fn from_links(prediction_id: PredictionId, links: Vec<EvidenceLink>) -> Self {
        let total_count = links.len();
        let mut supporting_count = 0;
        let mut contradicting_count = 0;
        let mut neutral_count = 0;
        let mut relevance_sum = 0.0;

        for link in &links {
            match link.contribution_type {
                ContributionType::Supporting => supporting_count += 1,
                ContributionType::Contradicting => contradicting_count += 1,
                ContributionType::Neutral => neutral_count += 1,
            }
            relevance_sum += link.relevance_score;
        }

        let avg_relevance = if total_count > 0 {
            relevance_sum / total_count as f64
        } else {
            0.0
        };

        // Get top patterns by relevance
        let mut sorted_links = links.clone();
        sorted_links.sort_by(|a, b| {
            b.relevance_score
                .partial_cmp(&a.relevance_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let top_patterns: Vec<String> = sorted_links
            .iter()
            .take(5)
            .map(|l| l.pattern_id.clone())
            .collect();

        let summary = EvidenceSummary {
            total_count,
            supporting_count,
            contradicting_count,
            neutral_count,
            avg_relevance,
            top_patterns,
        };

        Self {
            prediction_id,
            links,
            summary,
        }
    }
}

/// Create a simple evidence link (for use in prediction generation).
/// This creates an evidence link without a prediction ID, which should be set later.
pub fn link_evidence(pattern_id: impl Into<String>, relevance_score: f64) -> EvidenceLink {
    EvidenceLink::for_pattern(pattern_id, relevance_score)
}

/// Persist an evidence link to the database.
pub async fn store_evidence_link(
    db: Arc<SqliteDb>,
    link: &EvidenceLink,
) -> PredictionResult<()> {
    let sql = r#"
        INSERT INTO prediction_evidence (
            id, prediction_id, pattern_id, relevance_score,
            contribution_type, created_at
        ) VALUES (?, ?, ?, ?, ?, ?)
    "#;

    let contribution_str = link.contribution_type.to_string();
    let prediction_id_str = link.prediction_id.to_string();
    let created_at_str = link.created_at.to_rfc3339();

    db.execute(
        sql,
        &[
            &link.id as &dyn rusqlite::ToSql,
            &prediction_id_str,
            &link.pattern_id,
            &link.relevance_score,
            &contribution_str,
            &created_at_str,
        ],
    )
    .await
    .map_err(|e| PredictionError::Database(e.to_string()))?;

    debug!(
        prediction_id = %link.prediction_id,
        pattern_id = %link.pattern_id,
        "Evidence linked"
    );

    Ok(())
}

/// Get evidence for a prediction.
pub async fn get_prediction_evidence(
    db: Arc<SqliteDb>,
    prediction_id: &PredictionId,
) -> PredictionResult<PredictionEvidence> {
    let sql = r#"
        SELECT id, prediction_id, pattern_id, relevance_score,
               contribution_type, created_at
        FROM prediction_evidence
        WHERE prediction_id = ?
        ORDER BY relevance_score DESC
    "#;

    let prediction_id_str = prediction_id.to_string();

    let links: Vec<EvidenceLink> = db
        .query(sql, &[&prediction_id_str], |row| {
            let id: String = row.get("id")?;
            let pred_id_str: String = row.get("prediction_id")?;
            let pattern_id: String = row.get("pattern_id")?;
            let relevance_score: f64 = row.get("relevance_score")?;
            let contribution_str: String = row.get("contribution_type")?;
            let created_at_str: String = row.get("created_at")?;

            let contribution_type = contribution_str
                .parse::<ContributionType>()
                .unwrap_or(ContributionType::Supporting);

            let created_at = DateTime::parse_from_rfc3339(&created_at_str)
                .map(|dt| dt.with_timezone(&Utc))
                .unwrap_or_else(|_| Utc::now());

            Ok(EvidenceLink {
                id,
                prediction_id: PredictionId::from_string(pred_id_str),
                pattern_id,
                relevance_score,
                contribution_type,
                created_at,
            })
        })
        .await
        .map_err(|e| PredictionError::Database(e.to_string()))?;

    Ok(PredictionEvidence::from_links(prediction_id.clone(), links))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_evidence_link_creation() {
        let link = EvidenceLink::new(PredictionId::new(), "pattern-123")
            .with_relevance(0.85)
            .with_contribution_type(ContributionType::Supporting);

        assert!((link.relevance_score - 0.85).abs() < 0.001);
        assert_eq!(link.contribution_type, ContributionType::Supporting);
    }

    #[test]
    fn test_contribution_type_display() {
        assert_eq!(ContributionType::Supporting.to_string(), "supporting");
        assert_eq!(ContributionType::Contradicting.to_string(), "contradicting");
        assert_eq!(ContributionType::Neutral.to_string(), "neutral");
    }

    #[test]
    fn test_prediction_evidence_from_links() {
        let pred_id = PredictionId::new();
        let links = vec![
            EvidenceLink::new(pred_id.clone(), "p1").with_relevance(0.9),
            EvidenceLink::new(pred_id.clone(), "p2")
                .with_relevance(0.7)
                .with_contribution_type(ContributionType::Contradicting),
            EvidenceLink::new(pred_id.clone(), "p3").with_relevance(0.5),
        ];

        let evidence = PredictionEvidence::from_links(pred_id, links);

        assert_eq!(evidence.summary.total_count, 3);
        assert_eq!(evidence.summary.supporting_count, 2);
        assert_eq!(evidence.summary.contradicting_count, 1);
    }
}
