//! Domain event types for the Nagual event bus.
//!
//! Defines all event types that can be published through the event bus.
//! Events are serializable for persistence and transmission across components.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::health::HealthStatus;

/// Unique identifier for an event.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EventId(pub String);

impl EventId {
    /// Create a new random event ID.
    pub fn new() -> Self {
        Self(uuid::Uuid::new_v4().to_string())
    }

    /// Create an event ID from an existing string.
    pub fn from_string(s: impl Into<String>) -> Self {
        Self(s.into())
    }

    /// Get the underlying string.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Default for EventId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for EventId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Sync operation type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SyncType {
    /// Full synchronization
    Full,
    /// Incremental sync (only changes)
    Incremental,
    /// Bidirectional sync
    Bidirectional,
    /// Push to remote
    Push,
    /// Pull from remote
    Pull,
}

impl std::fmt::Display for SyncType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SyncType::Full => write!(f, "full"),
            SyncType::Incremental => write!(f, "incremental"),
            SyncType::Bidirectional => write!(f, "bidirectional"),
            SyncType::Push => write!(f, "push"),
            SyncType::Pull => write!(f, "pull"),
        }
    }
}

/// Changes made to a pattern during an update.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternChanges {
    /// Fields that were modified
    pub modified_fields: Vec<String>,
    /// Previous reward value (if changed)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub previous_reward: Option<f32>,
    /// New reward value (if changed)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub new_reward: Option<f32>,
    /// Previous effectiveness (if changed)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub previous_effectiveness: Option<f32>,
    /// New effectiveness (if changed)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub new_effectiveness: Option<f32>,
    /// Whether success status changed
    pub success_changed: bool,
    /// Additional metadata about the change
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
}

impl PatternChanges {
    /// Create a new empty changes record.
    pub fn new() -> Self {
        Self {
            modified_fields: Vec::new(),
            previous_reward: None,
            new_reward: None,
            previous_effectiveness: None,
            new_effectiveness: None,
            success_changed: false,
            metadata: HashMap::new(),
        }
    }

    /// Add a modified field.
    pub fn with_field(mut self, field: impl Into<String>) -> Self {
        self.modified_fields.push(field.into());
        self
    }

    /// Set reward change.
    pub fn with_reward_change(mut self, previous: f32, new: f32) -> Self {
        self.previous_reward = Some(previous);
        self.new_reward = Some(new);
        self.modified_fields.push("reward".to_string());
        self
    }

    /// Set effectiveness change.
    pub fn with_effectiveness_change(mut self, previous: f32, new: f32) -> Self {
        self.previous_effectiveness = Some(previous);
        self.new_effectiveness = Some(new);
        self.modified_fields.push("effectiveness".to_string());
        self
    }

    /// Mark success as changed.
    pub fn with_success_changed(mut self) -> Self {
        self.success_changed = true;
        self.modified_fields.push("success".to_string());
        self
    }

    /// Add metadata.
    pub fn with_metadata(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }
}

impl Default for PatternChanges {
    fn default() -> Self {
        Self::new()
    }
}

/// All domain events that can occur in the Nagual system.
///
/// Each event carries its own timestamp and relevant data for the operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum NagualEvent {
    /// A new pattern was stored in the reasoning bank.
    PatternStored {
        /// Unique event ID
        event_id: EventId,
        /// Timestamp when the event occurred
        timestamp: DateTime<Utc>,
        /// The pattern ID that was stored
        id: String,
        /// The domain/category of the pattern
        domain: String,
        /// Optional session ID
        #[serde(skip_serializing_if = "Option::is_none")]
        session_id: Option<String>,
        /// Optional agent ID
        #[serde(skip_serializing_if = "Option::is_none")]
        agent_id: Option<String>,
    },

    /// An existing pattern was updated.
    PatternUpdated {
        /// Unique event ID
        event_id: EventId,
        /// Timestamp when the event occurred
        timestamp: DateTime<Utc>,
        /// The pattern ID that was updated
        id: String,
        /// Description of changes made
        changes: PatternChanges,
    },

    /// A pattern was deleted.
    PatternDeleted {
        /// Unique event ID
        event_id: EventId,
        /// Timestamp when the event occurred
        timestamp: DateTime<Utc>,
        /// The pattern ID that was deleted
        id: String,
    },

    /// An outcome was recorded for a pattern (SONA learning).
    OutcomeRecorded {
        /// Unique event ID
        event_id: EventId,
        /// Timestamp when the event occurred
        timestamp: DateTime<Utc>,
        /// The pattern ID the outcome applies to
        pattern_id: String,
        /// The outcome (success, partial_success, neutral, failure)
        outcome: String,
        /// The calculated reward value
        reward: f32,
        /// Optional feedback provided
        #[serde(skip_serializing_if = "Option::is_none")]
        feedback: Option<String>,
    },

    /// Pattern consolidation completed (merging similar patterns).
    ConsolidationCompleted {
        /// Unique event ID
        event_id: EventId,
        /// Timestamp when the event occurred
        timestamp: DateTime<Utc>,
        /// Number of patterns that were merged
        merged_count: usize,
        /// Number of patterns archived
        archived_count: usize,
        /// IDs of patterns that were consolidated
        #[serde(default)]
        pattern_ids: Vec<String>,
    },

    /// Database synchronization completed.
    SyncCompleted {
        /// Unique event ID
        event_id: EventId,
        /// Timestamp when the event occurred
        timestamp: DateTime<Utc>,
        /// Type of sync operation
        sync_type: SyncType,
        /// Number of records synchronized
        records: usize,
        /// Duration of sync in milliseconds
        duration_ms: u64,
        /// Whether any conflicts occurred
        conflicts_detected: bool,
        /// Number of conflicts resolved
        conflicts_resolved: usize,
    },

    /// A new prediction was created.
    PredictionCreated {
        /// Unique event ID
        event_id: EventId,
        /// Timestamp when the event occurred
        timestamp: DateTime<Utc>,
        /// The prediction ID
        id: String,
        /// Predicted probability (0.0-1.0)
        probability: f64,
        /// Confidence in the prediction (0.0-1.0)
        confidence: f64,
        /// Domain of the prediction
        #[serde(default)]
        domain: String,
        /// Number of evidence patterns
        evidence_count: usize,
    },

    /// A prediction was resolved (outcome known).
    PredictionResolved {
        /// Unique event ID
        event_id: EventId,
        /// Timestamp when the event occurred
        timestamp: DateTime<Utc>,
        /// The prediction ID
        id: String,
        /// The actual outcome (true = happened, false = did not happen)
        outcome: bool,
        /// The Brier score (0.0 = perfect, 1.0 = worst)
        brier_score: f64,
        /// Whether the prediction was correct
        was_correct: bool,
    },

    /// Health status of a component changed.
    HealthChanged {
        /// Unique event ID
        event_id: EventId,
        /// Timestamp when the event occurred
        timestamp: DateTime<Utc>,
        /// Name of the component
        component: String,
        /// Previous health status
        previous_status: HealthStatus,
        /// New health status
        new_status: HealthStatus,
        /// Description of the change
        message: String,
    },

    /// A batch operation completed.
    BatchCompleted {
        /// Unique event ID
        event_id: EventId,
        /// Timestamp when the event occurred
        timestamp: DateTime<Utc>,
        /// Type of batch operation
        operation: String,
        /// Number of items processed
        items_processed: usize,
        /// Number of items that succeeded
        items_succeeded: usize,
        /// Number of items that failed
        items_failed: usize,
        /// Duration in milliseconds
        duration_ms: u64,
    },

    /// An error occurred in the system.
    ErrorOccurred {
        /// Unique event ID
        event_id: EventId,
        /// Timestamp when the event occurred
        timestamp: DateTime<Utc>,
        /// Component where error occurred
        component: String,
        /// Error code
        error_code: String,
        /// Error message
        message: String,
        /// Whether the error was recoverable
        recoverable: bool,
    },

    /// A witness entry was appended to the chain (KOS P1).
    WitnessAppended {
        /// Unique event ID
        event_id: EventId,
        /// Timestamp when the event occurred
        timestamp: DateTime<Utc>,
        /// Sequence number in the global chain
        seq: u64,
        /// The pattern this witness relates to
        pattern_id: String,
        /// The operation that was witnessed
        operation: String,
    },

    /// A delta was recorded for a pattern (KOS P2).
    DeltaRecorded {
        /// Unique event ID
        event_id: EventId,
        /// Timestamp when the event occurred
        timestamp: DateTime<Utc>,
        /// The pattern this delta applies to
        pattern_id: String,
        /// Sequence number of the delta
        seq: u64,
        /// Number of fields that changed
        fields_changed: usize,
    },

    /// Custom event for extensibility.
    Custom {
        /// Unique event ID
        event_id: EventId,
        /// Timestamp when the event occurred
        timestamp: DateTime<Utc>,
        /// Custom event name
        name: String,
        /// Event payload
        payload: HashMap<String, serde_json::Value>,
    },
}

impl NagualEvent {
    /// Get the event ID.
    pub fn event_id(&self) -> &EventId {
        match self {
            NagualEvent::PatternStored { event_id, .. } => event_id,
            NagualEvent::PatternUpdated { event_id, .. } => event_id,
            NagualEvent::PatternDeleted { event_id, .. } => event_id,
            NagualEvent::OutcomeRecorded { event_id, .. } => event_id,
            NagualEvent::ConsolidationCompleted { event_id, .. } => event_id,
            NagualEvent::SyncCompleted { event_id, .. } => event_id,
            NagualEvent::PredictionCreated { event_id, .. } => event_id,
            NagualEvent::PredictionResolved { event_id, .. } => event_id,
            NagualEvent::HealthChanged { event_id, .. } => event_id,
            NagualEvent::BatchCompleted { event_id, .. } => event_id,
            NagualEvent::ErrorOccurred { event_id, .. } => event_id,
            NagualEvent::WitnessAppended { event_id, .. } => event_id,
            NagualEvent::DeltaRecorded { event_id, .. } => event_id,
            NagualEvent::Custom { event_id, .. } => event_id,
        }
    }

    /// Get the timestamp when the event occurred.
    pub fn timestamp(&self) -> DateTime<Utc> {
        match self {
            NagualEvent::PatternStored { timestamp, .. } => *timestamp,
            NagualEvent::PatternUpdated { timestamp, .. } => *timestamp,
            NagualEvent::PatternDeleted { timestamp, .. } => *timestamp,
            NagualEvent::OutcomeRecorded { timestamp, .. } => *timestamp,
            NagualEvent::ConsolidationCompleted { timestamp, .. } => *timestamp,
            NagualEvent::SyncCompleted { timestamp, .. } => *timestamp,
            NagualEvent::PredictionCreated { timestamp, .. } => *timestamp,
            NagualEvent::PredictionResolved { timestamp, .. } => *timestamp,
            NagualEvent::HealthChanged { timestamp, .. } => *timestamp,
            NagualEvent::BatchCompleted { timestamp, .. } => *timestamp,
            NagualEvent::ErrorOccurred { timestamp, .. } => *timestamp,
            NagualEvent::WitnessAppended { timestamp, .. } => *timestamp,
            NagualEvent::DeltaRecorded { timestamp, .. } => *timestamp,
            NagualEvent::Custom { timestamp, .. } => *timestamp,
        }
    }

    /// Get the event type name as a string.
    pub fn event_type(&self) -> &'static str {
        match self {
            NagualEvent::PatternStored { .. } => "pattern_stored",
            NagualEvent::PatternUpdated { .. } => "pattern_updated",
            NagualEvent::PatternDeleted { .. } => "pattern_deleted",
            NagualEvent::OutcomeRecorded { .. } => "outcome_recorded",
            NagualEvent::ConsolidationCompleted { .. } => "consolidation_completed",
            NagualEvent::SyncCompleted { .. } => "sync_completed",
            NagualEvent::PredictionCreated { .. } => "prediction_created",
            NagualEvent::PredictionResolved { .. } => "prediction_resolved",
            NagualEvent::HealthChanged { .. } => "health_changed",
            NagualEvent::BatchCompleted { .. } => "batch_completed",
            NagualEvent::ErrorOccurred { .. } => "error_occurred",
            NagualEvent::WitnessAppended { .. } => "witness_appended",
            NagualEvent::DeltaRecorded { .. } => "delta_recorded",
            NagualEvent::Custom { .. } => "custom",
        }
    }

    // Factory methods for creating events

    /// Create a PatternStored event.
    pub fn pattern_stored(id: impl Into<String>, domain: impl Into<String>) -> Self {
        NagualEvent::PatternStored {
            event_id: EventId::new(),
            timestamp: Utc::now(),
            id: id.into(),
            domain: domain.into(),
            session_id: None,
            agent_id: None,
        }
    }

    /// Create a PatternStored event with session and agent IDs.
    pub fn pattern_stored_with_context(
        id: impl Into<String>,
        domain: impl Into<String>,
        session_id: Option<String>,
        agent_id: Option<String>,
    ) -> Self {
        NagualEvent::PatternStored {
            event_id: EventId::new(),
            timestamp: Utc::now(),
            id: id.into(),
            domain: domain.into(),
            session_id,
            agent_id,
        }
    }

    /// Create a PatternUpdated event.
    pub fn pattern_updated(id: impl Into<String>, changes: PatternChanges) -> Self {
        NagualEvent::PatternUpdated {
            event_id: EventId::new(),
            timestamp: Utc::now(),
            id: id.into(),
            changes,
        }
    }

    /// Create a PatternDeleted event.
    pub fn pattern_deleted(id: impl Into<String>) -> Self {
        NagualEvent::PatternDeleted {
            event_id: EventId::new(),
            timestamp: Utc::now(),
            id: id.into(),
        }
    }

    /// Create an OutcomeRecorded event.
    pub fn outcome_recorded(
        pattern_id: impl Into<String>,
        outcome: impl Into<String>,
        reward: f32,
        feedback: Option<String>,
    ) -> Self {
        NagualEvent::OutcomeRecorded {
            event_id: EventId::new(),
            timestamp: Utc::now(),
            pattern_id: pattern_id.into(),
            outcome: outcome.into(),
            reward,
            feedback,
        }
    }

    /// Create a ConsolidationCompleted event.
    pub fn consolidation_completed(
        merged_count: usize,
        archived_count: usize,
        pattern_ids: Vec<String>,
    ) -> Self {
        NagualEvent::ConsolidationCompleted {
            event_id: EventId::new(),
            timestamp: Utc::now(),
            merged_count,
            archived_count,
            pattern_ids,
        }
    }

    /// Create a SyncCompleted event.
    pub fn sync_completed(
        sync_type: SyncType,
        records: usize,
        duration_ms: u64,
        conflicts_detected: bool,
        conflicts_resolved: usize,
    ) -> Self {
        NagualEvent::SyncCompleted {
            event_id: EventId::new(),
            timestamp: Utc::now(),
            sync_type,
            records,
            duration_ms,
            conflicts_detected,
            conflicts_resolved,
        }
    }

    /// Create a PredictionCreated event.
    pub fn prediction_created(
        id: impl Into<String>,
        probability: f64,
        confidence: f64,
        domain: impl Into<String>,
        evidence_count: usize,
    ) -> Self {
        NagualEvent::PredictionCreated {
            event_id: EventId::new(),
            timestamp: Utc::now(),
            id: id.into(),
            probability,
            confidence,
            domain: domain.into(),
            evidence_count,
        }
    }

    /// Create a PredictionResolved event.
    pub fn prediction_resolved(
        id: impl Into<String>,
        outcome: bool,
        brier_score: f64,
        was_correct: bool,
    ) -> Self {
        NagualEvent::PredictionResolved {
            event_id: EventId::new(),
            timestamp: Utc::now(),
            id: id.into(),
            outcome,
            brier_score,
            was_correct,
        }
    }

    /// Create a HealthChanged event.
    pub fn health_changed(
        component: impl Into<String>,
        previous_status: HealthStatus,
        new_status: HealthStatus,
        message: impl Into<String>,
    ) -> Self {
        NagualEvent::HealthChanged {
            event_id: EventId::new(),
            timestamp: Utc::now(),
            component: component.into(),
            previous_status,
            new_status,
            message: message.into(),
        }
    }

    /// Create a BatchCompleted event.
    pub fn batch_completed(
        operation: impl Into<String>,
        items_processed: usize,
        items_succeeded: usize,
        items_failed: usize,
        duration_ms: u64,
    ) -> Self {
        NagualEvent::BatchCompleted {
            event_id: EventId::new(),
            timestamp: Utc::now(),
            operation: operation.into(),
            items_processed,
            items_succeeded,
            items_failed,
            duration_ms,
        }
    }

    /// Create an ErrorOccurred event.
    pub fn error_occurred(
        component: impl Into<String>,
        error_code: impl Into<String>,
        message: impl Into<String>,
        recoverable: bool,
    ) -> Self {
        NagualEvent::ErrorOccurred {
            event_id: EventId::new(),
            timestamp: Utc::now(),
            component: component.into(),
            error_code: error_code.into(),
            message: message.into(),
            recoverable,
        }
    }

    /// Create a WitnessAppended event.
    pub fn witness_appended(
        seq: u64,
        pattern_id: impl Into<String>,
        operation: impl Into<String>,
    ) -> Self {
        NagualEvent::WitnessAppended {
            event_id: EventId::new(),
            timestamp: Utc::now(),
            seq,
            pattern_id: pattern_id.into(),
            operation: operation.into(),
        }
    }

    /// Create a DeltaRecorded event.
    pub fn delta_recorded(
        pattern_id: impl Into<String>,
        seq: u64,
        fields_changed: usize,
    ) -> Self {
        NagualEvent::DeltaRecorded {
            event_id: EventId::new(),
            timestamp: Utc::now(),
            pattern_id: pattern_id.into(),
            seq,
            fields_changed,
        }
    }

    /// Create a Custom event.
    pub fn custom(name: impl Into<String>, payload: HashMap<String, serde_json::Value>) -> Self {
        NagualEvent::Custom {
            event_id: EventId::new(),
            timestamp: Utc::now(),
            name: name.into(),
            payload,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_event_id_new() {
        let id1 = EventId::new();
        let id2 = EventId::new();
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_pattern_stored_event() {
        let event = NagualEvent::pattern_stored("pattern-123", "rust.async");

        assert_eq!(event.event_type(), "pattern_stored");

        if let NagualEvent::PatternStored { id, domain, .. } = &event {
            assert_eq!(id, "pattern-123");
            assert_eq!(domain, "rust.async");
        } else {
            panic!("Expected PatternStored event");
        }
    }

    #[test]
    fn test_outcome_recorded_event() {
        let event = NagualEvent::outcome_recorded(
            "pattern-123",
            "success",
            0.9,
            Some("Great result".to_string()),
        );

        assert_eq!(event.event_type(), "outcome_recorded");

        if let NagualEvent::OutcomeRecorded {
            pattern_id,
            outcome,
            reward,
            feedback,
            ..
        } = &event
        {
            assert_eq!(pattern_id, "pattern-123");
            assert_eq!(outcome, "success");
            assert!((reward - 0.9).abs() < 0.001);
            assert_eq!(feedback.as_deref(), Some("Great result"));
        } else {
            panic!("Expected OutcomeRecorded event");
        }
    }

    #[test]
    fn test_pattern_changes() {
        let changes = PatternChanges::new()
            .with_reward_change(0.5, 0.8)
            .with_effectiveness_change(0.6, 0.85)
            .with_success_changed()
            .with_metadata("reason", serde_json::json!("good outcome"));

        assert_eq!(changes.previous_reward, Some(0.5));
        assert_eq!(changes.new_reward, Some(0.8));
        assert!(changes.success_changed);
        assert!(changes.modified_fields.contains(&"reward".to_string()));
        assert!(changes.modified_fields.contains(&"effectiveness".to_string()));
        assert!(changes.modified_fields.contains(&"success".to_string()));
    }

    #[test]
    fn test_event_serialization() {
        let event = NagualEvent::pattern_stored("test-id", "test.domain");
        let json = serde_json::to_string(&event).unwrap();
        let deserialized: NagualEvent = serde_json::from_str(&json).unwrap();

        assert_eq!(event.event_type(), deserialized.event_type());
    }

    #[test]
    fn test_sync_type_display() {
        assert_eq!(SyncType::Full.to_string(), "full");
        assert_eq!(SyncType::Incremental.to_string(), "incremental");
        assert_eq!(SyncType::Bidirectional.to_string(), "bidirectional");
    }

    #[test]
    fn test_health_changed_event() {
        let event = NagualEvent::health_changed(
            "database",
            HealthStatus::Healthy,
            HealthStatus::Degraded,
            "Connection pool exhausted",
        );

        if let NagualEvent::HealthChanged {
            component,
            previous_status,
            new_status,
            message,
            ..
        } = &event
        {
            assert_eq!(component, "database");
            assert_eq!(*previous_status, HealthStatus::Healthy);
            assert_eq!(*new_status, HealthStatus::Degraded);
            assert!(message.contains("pool"));
        } else {
            panic!("Expected HealthChanged event");
        }
    }

    #[test]
    fn test_prediction_events() {
        let created = NagualEvent::prediction_created(
            "pred-123",
            0.75,
            0.85,
            "devops.deployment",
            5,
        );

        if let NagualEvent::PredictionCreated {
            id,
            probability,
            confidence,
            evidence_count,
            ..
        } = &created
        {
            assert_eq!(id, "pred-123");
            assert!((probability - 0.75).abs() < 0.001);
            assert!((confidence - 0.85).abs() < 0.001);
            assert_eq!(*evidence_count, 5);
        } else {
            panic!("Expected PredictionCreated event");
        }

        let resolved = NagualEvent::prediction_resolved("pred-123", true, 0.04, true);

        if let NagualEvent::PredictionResolved {
            id,
            outcome,
            brier_score,
            was_correct,
            ..
        } = &resolved
        {
            assert_eq!(id, "pred-123");
            assert!(*outcome);
            assert!((brier_score - 0.04).abs() < 0.001);
            assert!(*was_correct);
        } else {
            panic!("Expected PredictionResolved event");
        }
    }

    #[test]
    fn test_custom_event() {
        let mut payload = HashMap::new();
        payload.insert("key".to_string(), serde_json::json!("value"));

        let event = NagualEvent::custom("my_custom_event", payload);

        if let NagualEvent::Custom { name, payload, .. } = &event {
            assert_eq!(name, "my_custom_event");
            assert_eq!(payload.get("key"), Some(&serde_json::json!("value")));
        } else {
            panic!("Expected Custom event");
        }
    }
}
