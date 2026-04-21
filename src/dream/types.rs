//! Dream Cycle Types
//!
//! Core types for the Dream Cycle background maintenance system.
//! Provides pattern consolidation, refresh, prediction calibration,
//! and spreading activation during idle periods.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::fmt;

/// Dream cycle configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DreamConfig {
    /// Whether dream cycle is enabled
    pub enabled: bool,
    /// Seconds of idle time before triggering dream cycle
    pub idle_threshold_seconds: u64,
    /// Maximum duration for a dream cycle
    pub max_duration_seconds: u64,
    /// Which phases are enabled
    pub phases: DreamPhases,
    /// Budget constraints
    pub budget: DreamBudget,
}

impl Default for DreamConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            idle_threshold_seconds: 300, // 5 minutes
            max_duration_seconds: 30,
            phases: DreamPhases::default(),
            budget: DreamBudget::default(),
        }
    }
}

/// Which dream phases are enabled
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DreamPhases {
    /// Pattern consolidation (merge similar, dedupe, archive low-quality)
    pub consolidate: bool,
    /// Pattern refresh (update stale patterns via research)
    pub refresh: bool,
    /// Prediction calibration (update Brier scores)
    pub calibrate: bool,
    /// Spreading activation (strengthen/create pattern connections)
    pub activate: bool,
}

impl Default for DreamPhases {
    fn default() -> Self {
        Self {
            consolidate: true,
            refresh: true,
            calibrate: true,
            activate: true,
        }
    }
}

/// Budget constraints for dream cycle
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DreamBudget {
    /// Maximum patterns to consolidate per cycle
    pub max_patterns_consolidated: usize,
    /// Maximum patterns to refresh per cycle
    pub max_patterns_refreshed: usize,
    /// Maximum predictions to calibrate per cycle
    pub max_predictions_calibrated: usize,
    /// Maximum tokens to use per cycle (for refresh research)
    pub max_tokens_per_cycle: usize,
}

impl Default for DreamBudget {
    fn default() -> Self {
        Self {
            max_patterns_consolidated: 20,
            max_patterns_refreshed: 5,
            max_predictions_calibrated: 10,
            max_tokens_per_cycle: 5000,
        }
    }
}

/// Result of a complete dream cycle
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DreamResult {
    /// Unique cycle identifier
    pub cycle_id: String,
    /// When the cycle started
    pub started_at: DateTime<Utc>,
    /// When the cycle completed
    pub completed_at: DateTime<Utc>,
    /// Results from each phase
    pub phases_completed: Vec<PhaseResult>,
    /// Total duration in milliseconds
    pub total_duration_ms: u64,
    /// Total tokens used (refresh phase)
    pub tokens_used: usize,
}

impl DreamResult {
    /// Check if cycle was successful (all phases succeeded)
    pub fn is_success(&self) -> bool {
        self.phases_completed.iter().all(|p| p.success)
    }

    /// Get total items processed across all phases
    pub fn total_items_processed(&self) -> usize {
        self.phases_completed.iter().map(|p| p.items_processed).sum()
    }
}

/// Result of a single dream phase
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseResult {
    /// Which phase
    pub phase: DreamPhase,
    /// Whether the phase succeeded
    pub success: bool,
    /// Number of items processed
    pub items_processed: usize,
    /// Duration in milliseconds
    pub duration_ms: u64,
    /// Phase-specific details
    pub details: PhaseDetails,
}

/// Dream cycle phases
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DreamPhase {
    /// Merge similar patterns, deduplicate, archive low-quality
    Consolidate,
    /// Research updates for stale patterns
    Refresh,
    /// Update prediction calibration scores
    Calibrate,
    /// Spreading activation through pattern graph
    Activate,
}

impl fmt::Display for DreamPhase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DreamPhase::Consolidate => write!(f, "consolidate"),
            DreamPhase::Refresh => write!(f, "refresh"),
            DreamPhase::Calibrate => write!(f, "calibrate"),
            DreamPhase::Activate => write!(f, "activate"),
        }
    }
}

/// Phase-specific details
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum PhaseDetails {
    /// Consolidation phase details
    Consolidate {
        patterns_merged: usize,
        patterns_archived: usize,
        duplicates_removed: usize,
    },
    /// Refresh phase details
    Refresh {
        patterns_refreshed: usize,
        research_triggered: usize,
        patterns_updated: usize,
    },
    /// Calibration phase details
    Calibrate {
        predictions_reviewed: usize,
        brier_score_before: f64,
        brier_score_after: f64,
    },
    /// Activation phase details
    Activate {
        connections_strengthened: usize,
        new_connections: usize,
        activation_spread: f64,
    },
}

impl Default for PhaseDetails {
    fn default() -> Self {
        PhaseDetails::Consolidate {
            patterns_merged: 0,
            patterns_archived: 0,
            duplicates_removed: 0,
        }
    }
}

/// Dream cycle status for monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DreamStatus {
    /// Whether dream cycle is enabled
    pub enabled: bool,
    /// Current state
    pub state: DreamState,
    /// Last completed cycle
    pub last_cycle: Option<DreamResult>,
    /// Time until next cycle (if idle)
    pub next_cycle_in_seconds: Option<u64>,
    /// Total cycles completed
    pub total_cycles: usize,
    /// Total items processed across all cycles
    pub total_items_processed: usize,
}

/// Current dream cycle state
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DreamState {
    /// Waiting for idle threshold
    Idle,
    /// Currently running a cycle
    Running,
    /// Disabled
    Disabled,
}

impl fmt::Display for DreamState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DreamState::Idle => write!(f, "idle"),
            DreamState::Running => write!(f, "running"),
            DreamState::Disabled => write!(f, "disabled"),
        }
    }
}

/// Consolidated pattern information
#[derive(Debug, Clone)]
pub struct ConsolidationCandidate {
    pub id: String,
    pub problem: String,
    pub solution: String,
    pub domain: String,
    pub reward: f64,
    pub created_at: DateTime<Utc>,
}

/// Stale pattern information
#[derive(Debug, Clone)]
pub struct StalePattern {
    pub id: String,
    pub problem: String,
    pub domain: String,
    pub days_since_update: i64,
    pub relevance_score: f64,
}

/// Related pattern for spreading activation
#[derive(Debug, Clone)]
pub struct RelatedPattern {
    pub id: String,
    pub similarity: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dream_config_default() {
        let config = DreamConfig::default();
        assert!(config.enabled);
        assert_eq!(config.idle_threshold_seconds, 300);
        assert_eq!(config.max_duration_seconds, 30);
        assert!(config.phases.consolidate);
        assert!(config.phases.refresh);
        assert!(config.phases.calibrate);
        assert!(config.phases.activate);
    }

    #[test]
    fn test_dream_budget_default() {
        let budget = DreamBudget::default();
        assert_eq!(budget.max_patterns_consolidated, 20);
        assert_eq!(budget.max_patterns_refreshed, 5);
        assert_eq!(budget.max_predictions_calibrated, 10);
        assert_eq!(budget.max_tokens_per_cycle, 5000);
    }

    #[test]
    fn test_dream_phase_display() {
        assert_eq!(format!("{}", DreamPhase::Consolidate), "consolidate");
        assert_eq!(format!("{}", DreamPhase::Refresh), "refresh");
        assert_eq!(format!("{}", DreamPhase::Calibrate), "calibrate");
        assert_eq!(format!("{}", DreamPhase::Activate), "activate");
    }

    #[test]
    fn test_phase_details_serialization() {
        let details = PhaseDetails::Consolidate {
            patterns_merged: 5,
            patterns_archived: 2,
            duplicates_removed: 3,
        };
        let json = serde_json::to_string(&details).unwrap();
        assert!(json.contains("\"type\":\"Consolidate\""));
    }
}
