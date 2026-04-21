//! PostgreSQL LISTEN/NOTIFY real-time event bridge.
//!
//! Listens on PostgreSQL notification channels fired by the triggers in
//! migration 016 (`check_pattern_graduation`, `notify_consolidation`) and
//! converts them into typed [`NagualEvent`]s published on the [`EventBus`].
//!
//! # Channels
//!
//! | Channel                         | Trigger source                          |
//! |---------------------------------|-----------------------------------------|
//! | `nagual_pattern_stored`         | INSERT on `reasoning_patterns`          |
//! | `nagual_pattern_promoted`       | UPDATE of reward/reuse_count (tier change) |
//! | `nagual_consolidation_complete` | DELETE on `reasoning_patterns`           |
//!
//! # Usage
//!
//! ```ignore
//! use nagual::db::pg_notify::PgNotifyListener;
//! use nagual::events::EventBus;
//! use std::sync::Arc;
//!
//! let event_bus = Arc::new(EventBus::new());
//! let listener = PgNotifyListener::new(
//!     "postgres://nagual:password@localhost:5432/nagual",
//!     event_bus,
//! );
//! let handle = listener.start().await?;
//! // ... later ...
//! handle.stop();
//! ```

use std::sync::Arc;

use serde::Deserialize;
use tokio::sync::watch;
use tracing::{debug, error, info, warn};

use crate::events::{EventBus, NagualEvent, PatternChanges};

// ---------------------------------------------------------------------------
// Notification channel names (must match the pg_notify calls in migration 016)
// ---------------------------------------------------------------------------

/// Channel for new pattern insertions.
pub const CHANNEL_PATTERN_STORED: &str = "nagual_pattern_stored";
/// Channel for tier promotions / demotions.
pub const CHANNEL_PATTERN_PROMOTED: &str = "nagual_pattern_promoted";
/// Channel for consolidation (pattern deletion).
pub const CHANNEL_CONSOLIDATION_COMPLETE: &str = "nagual_consolidation_complete";

// ---------------------------------------------------------------------------
// JSON payload structs (deserialized from pg_notify payloads)
// ---------------------------------------------------------------------------

/// Payload for `nagual_pattern_stored` notifications.
///
/// Fired by the INSERT branch of `check_pattern_graduation()`.
#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct PatternStoredPayload {
    /// The pattern UUID.
    pub id: String,
    /// The pattern category / domain.
    pub category: String,
    /// The pattern tier (e.g. "booster", "crystal", "reflex").
    pub tier: String,
}

/// Payload for `nagual_pattern_promoted` notifications.
///
/// Fired by the UPDATE branches of `check_pattern_graduation()` when
/// a tier transition (promotion or demotion) is detected.
#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct PatternPromotedPayload {
    /// The pattern UUID.
    pub id: String,
    /// The tier before the change.
    pub old_tier: String,
    /// The tier after the change.
    pub new_tier: String,
    /// Current reward value.
    pub reward: f64,
    /// Current reuse count.
    pub reuse_count: i64,
}

/// Payload for `nagual_consolidation_complete` notifications.
///
/// Fired by the `notify_consolidation()` trigger on DELETE.
#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct ConsolidationCompletePayload {
    /// The pattern UUID that was deleted.
    pub deleted_id: String,
    /// The category of the deleted pattern.
    pub category: String,
}

// ---------------------------------------------------------------------------
// Parsing helpers
// ---------------------------------------------------------------------------

/// Errors that can occur when parsing a notification payload.
#[derive(Debug, thiserror::Error)]
pub enum NotifyParseError {
    /// The JSON payload could not be deserialized.
    #[error("Invalid JSON in notification payload: {0}")]
    InvalidJson(#[from] serde_json::Error),

    /// The notification channel is not recognized.
    #[error("Unknown notification channel: {0}")]
    UnknownChannel(String),
}

/// A fully-parsed notification from PostgreSQL.
#[derive(Debug, Clone)]
pub enum PgNotification {
    /// A new pattern was inserted.
    PatternStored(PatternStoredPayload),
    /// A pattern changed tiers.
    PatternPromoted(PatternPromotedPayload),
    /// A pattern was deleted during consolidation.
    ConsolidationComplete(ConsolidationCompletePayload),
}

/// Parse a raw PostgreSQL notification into a typed [`PgNotification`].
///
/// # Arguments
///
/// * `channel` - The LISTEN channel name.
/// * `payload` - The JSON string sent by `pg_notify`.
pub fn parse_notification(channel: &str, payload: &str) -> Result<PgNotification, NotifyParseError> {
    match channel {
        CHANNEL_PATTERN_STORED => {
            let p: PatternStoredPayload = serde_json::from_str(payload)?;
            Ok(PgNotification::PatternStored(p))
        }
        CHANNEL_PATTERN_PROMOTED => {
            let p: PatternPromotedPayload = serde_json::from_str(payload)?;
            Ok(PgNotification::PatternPromoted(p))
        }
        CHANNEL_CONSOLIDATION_COMPLETE => {
            let p: ConsolidationCompletePayload = serde_json::from_str(payload)?;
            Ok(PgNotification::ConsolidationComplete(p))
        }
        other => Err(NotifyParseError::UnknownChannel(other.to_string())),
    }
}

/// Convert a parsed [`PgNotification`] into a [`NagualEvent`] suitable for
/// the internal event bus.
pub fn notification_to_event(notification: &PgNotification) -> NagualEvent {
    match notification {
        PgNotification::PatternStored(p) => {
            NagualEvent::pattern_stored(&p.id, &p.category)
        }
        PgNotification::PatternPromoted(p) => {
            let changes = PatternChanges::new()
                .with_field("tier")
                .with_metadata("old_tier", serde_json::json!(&p.old_tier))
                .with_metadata("new_tier", serde_json::json!(&p.new_tier))
                .with_metadata("reuse_count", serde_json::json!(p.reuse_count))
                .with_reward_change(0.0, p.reward as f32);
            NagualEvent::pattern_updated(&p.id, changes)
        }
        PgNotification::ConsolidationComplete(p) => {
            NagualEvent::consolidation_completed(
                0,
                0,
                vec![p.deleted_id.clone()],
            )
        }
    }
}

// ---------------------------------------------------------------------------
// PgNotifyListener
// ---------------------------------------------------------------------------

/// Handle returned by [`PgNotifyListener::start`] to control the background
/// listener task.
pub struct PgNotifyHandle {
    shutdown_tx: watch::Sender<bool>,
    join_handle: tokio::task::JoinHandle<()>,
}

impl PgNotifyHandle {
    /// Signal the listener to shut down gracefully.
    pub fn stop(&self) {
        let _ = self.shutdown_tx.send(true);
    }

    /// Wait for the listener task to finish (after calling [`stop`]).
    pub async fn join(self) {
        let _ = self.join_handle.await;
    }

    /// Check whether the background task is still running.
    pub fn is_running(&self) -> bool {
        !self.join_handle.is_finished()
    }
}

/// Listens for PostgreSQL `LISTEN/NOTIFY` events and bridges them to the
/// Nagual [`EventBus`].
///
/// Internally uses [`sqlx::postgres::PgListener`] and spawns a Tokio task
/// that runs until [`PgNotifyHandle::stop`] is called.
pub struct PgNotifyListener {
    /// PostgreSQL connection URL.
    postgres_url: String,
    /// Shared reference to the Nagual event bus.
    event_bus: Arc<EventBus>,
}

impl PgNotifyListener {
    /// Create a new listener. Does **not** connect yet; call [`start`] to
    /// begin listening.
    pub fn new(postgres_url: impl Into<String>, event_bus: Arc<EventBus>) -> Self {
        Self {
            postgres_url: postgres_url.into(),
            event_bus,
        }
    }

    /// Connect to PostgreSQL, subscribe to all notification channels, and
    /// spawn a background task that forwards events to the [`EventBus`].
    ///
    /// Returns a [`PgNotifyHandle`] that can be used to stop the listener.
    pub async fn start(&self) -> Result<PgNotifyHandle, crate::error::NagualError> {
        let mut pg_listener = sqlx::postgres::PgListener::connect(&self.postgres_url)
            .await
            .map_err(crate::error::DatabaseError::from)?;

        pg_listener
            .listen_all(vec![
                CHANNEL_PATTERN_STORED,
                CHANNEL_PATTERN_PROMOTED,
                CHANNEL_CONSOLIDATION_COMPLETE,
            ])
            .await
            .map_err(crate::error::DatabaseError::from)?;

        info!(
            channels = ?[CHANNEL_PATTERN_STORED, CHANNEL_PATTERN_PROMOTED, CHANNEL_CONSOLIDATION_COMPLETE],
            "PgNotifyListener subscribed to notification channels"
        );

        let (shutdown_tx, mut shutdown_rx) = watch::channel(false);
        let event_bus = Arc::clone(&self.event_bus);

        let join_handle = tokio::spawn(async move {
            loop {
                tokio::select! {
                    // Wait for the next PG notification
                    result = pg_listener.recv() => {
                        match result {
                            Ok(notification) => {
                                let channel = notification.channel();
                                let payload = notification.payload();

                                debug!(
                                    channel = %channel,
                                    payload = %payload,
                                    "Received PG notification"
                                );

                                match parse_notification(channel, payload) {
                                    Ok(parsed) => {
                                        let event = notification_to_event(&parsed);
                                        if let Err(e) = event_bus.publish(event).await {
                                            warn!(
                                                error = %e,
                                                channel = %channel,
                                                "Failed to publish PG notification to event bus"
                                            );
                                        }
                                    }
                                    Err(e) => {
                                        error!(
                                            error = %e,
                                            channel = %channel,
                                            payload = %payload,
                                            "Failed to parse PG notification payload"
                                        );
                                    }
                                }
                            }
                            Err(e) => {
                                error!(error = %e, "PgListener recv error, will retry");
                                // Brief backoff before retrying
                                tokio::time::sleep(std::time::Duration::from_secs(1)).await;
                            }
                        }
                    }
                    // Check for shutdown signal
                    _ = shutdown_rx.changed() => {
                        if *shutdown_rx.borrow() {
                            info!("PgNotifyListener shutting down");
                            break;
                        }
                    }
                }
            }
        });

        Ok(PgNotifyHandle {
            shutdown_tx,
            join_handle,
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- Payload parsing tests -----------------------------------------------

    #[test]
    fn test_parse_pattern_stored_payload() {
        let json = r#"{"id": "abc-123", "category": "rust.async", "tier": "booster"}"#;
        let result = parse_notification(CHANNEL_PATTERN_STORED, json);
        assert!(result.is_ok(), "parse should succeed");

        if let Ok(PgNotification::PatternStored(p)) = result {
            assert_eq!(p.id, "abc-123");
            assert_eq!(p.category, "rust.async");
            assert_eq!(p.tier, "booster");
        } else {
            panic!("Expected PatternStored variant");
        }
    }

    #[test]
    fn test_parse_pattern_promoted_payload() {
        let json = r#"{
            "id": "def-456",
            "old_tier": "booster",
            "new_tier": "crystal",
            "reward": 0.75,
            "reuse_count": 8
        }"#;
        let result = parse_notification(CHANNEL_PATTERN_PROMOTED, json);
        assert!(result.is_ok(), "parse should succeed");

        if let Ok(PgNotification::PatternPromoted(p)) = result {
            assert_eq!(p.id, "def-456");
            assert_eq!(p.old_tier, "booster");
            assert_eq!(p.new_tier, "crystal");
            assert!((p.reward - 0.75).abs() < f64::EPSILON);
            assert_eq!(p.reuse_count, 8);
        } else {
            panic!("Expected PatternPromoted variant");
        }
    }

    #[test]
    fn test_parse_consolidation_complete_payload() {
        let json = r#"{"deleted_id": "old-pattern-789", "category": "devops.ci"}"#;
        let result = parse_notification(CHANNEL_CONSOLIDATION_COMPLETE, json);
        assert!(result.is_ok(), "parse should succeed");

        if let Ok(PgNotification::ConsolidationComplete(p)) = result {
            assert_eq!(p.deleted_id, "old-pattern-789");
            assert_eq!(p.category, "devops.ci");
        } else {
            panic!("Expected ConsolidationComplete variant");
        }
    }

    #[test]
    fn test_parse_unknown_channel_returns_error() {
        let json = r#"{"id": "x"}"#;
        let result = parse_notification("unknown_channel", json);
        assert!(result.is_err());

        if let Err(NotifyParseError::UnknownChannel(ch)) = result {
            assert_eq!(ch, "unknown_channel");
        } else {
            panic!("Expected UnknownChannel error");
        }
    }

    #[test]
    fn test_parse_invalid_json_returns_error() {
        let bad_json = r#"not valid json"#;
        let result = parse_notification(CHANNEL_PATTERN_STORED, bad_json);
        assert!(result.is_err());
        assert!(matches!(result, Err(NotifyParseError::InvalidJson(_))));
    }

    #[test]
    fn test_parse_missing_field_returns_error() {
        // Missing 'tier' field which is required
        let json = r#"{"id": "abc", "category": "rust"}"#;
        let result = parse_notification(CHANNEL_PATTERN_STORED, json);
        assert!(result.is_err());
        assert!(matches!(result, Err(NotifyParseError::InvalidJson(_))));
    }

    // -- Event conversion tests -----------------------------------------------

    #[test]
    fn test_pattern_stored_to_event() {
        let payload = PatternStoredPayload {
            id: "p-100".to_string(),
            category: "database.optimization".to_string(),
            tier: "crystal".to_string(),
        };
        let event = notification_to_event(&PgNotification::PatternStored(payload));

        assert_eq!(event.event_type(), "pattern_stored");
        if let NagualEvent::PatternStored { id, domain, .. } = &event {
            assert_eq!(id, "p-100");
            assert_eq!(domain, "database.optimization");
        } else {
            panic!("Expected PatternStored event");
        }
    }

    #[test]
    fn test_pattern_promoted_to_event() {
        let payload = PatternPromotedPayload {
            id: "p-200".to_string(),
            old_tier: "booster".to_string(),
            new_tier: "reflex".to_string(),
            reward: 0.95,
            reuse_count: 25,
        };
        let event = notification_to_event(&PgNotification::PatternPromoted(payload));

        assert_eq!(event.event_type(), "pattern_updated");
        if let NagualEvent::PatternUpdated { id, changes, .. } = &event {
            assert_eq!(id, "p-200");
            assert!(changes.modified_fields.contains(&"tier".to_string()));
            assert_eq!(
                changes.metadata.get("old_tier"),
                Some(&serde_json::json!("booster"))
            );
            assert_eq!(
                changes.metadata.get("new_tier"),
                Some(&serde_json::json!("reflex"))
            );
            assert_eq!(
                changes.metadata.get("reuse_count"),
                Some(&serde_json::json!(25))
            );
        } else {
            panic!("Expected PatternUpdated event");
        }
    }

    #[test]
    fn test_consolidation_to_event() {
        let payload = ConsolidationCompletePayload {
            deleted_id: "dead-pattern".to_string(),
            category: "misc".to_string(),
        };
        let event = notification_to_event(&PgNotification::ConsolidationComplete(payload));

        assert_eq!(event.event_type(), "consolidation_completed");
        if let NagualEvent::ConsolidationCompleted { pattern_ids, .. } = &event {
            assert_eq!(pattern_ids, &vec!["dead-pattern".to_string()]);
        } else {
            panic!("Expected ConsolidationCompleted event");
        }
    }

    // -- Payload deserialization round-trip ------------------------------------

    #[test]
    fn test_promoted_payload_with_integer_reward() {
        // PostgreSQL may send integer values (e.g. reward = 1 instead of 1.0)
        let json = r#"{"id": "x", "old_tier": "a", "new_tier": "b", "reward": 1, "reuse_count": 5}"#;
        let result = parse_notification(CHANNEL_PATTERN_PROMOTED, json);
        assert!(result.is_ok());

        if let Ok(PgNotification::PatternPromoted(p)) = result {
            assert!((p.reward - 1.0).abs() < f64::EPSILON);
        }
    }

    #[test]
    fn test_channel_constants_match_migration() {
        // Verify channel names match what migration 016 uses
        assert_eq!(CHANNEL_PATTERN_STORED, "nagual_pattern_stored");
        assert_eq!(CHANNEL_PATTERN_PROMOTED, "nagual_pattern_promoted");
        assert_eq!(CHANNEL_CONSOLIDATION_COMPLETE, "nagual_consolidation_complete");
    }
}
