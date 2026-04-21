//! Event Bus for the Nagual system.
//!
//! Provides a publish-subscribe mechanism for domain events using
//! Tokio's broadcast channel. Components can publish events and
//! subscribe to receive notifications about system changes.
//!
//! # Architecture
//!
//! ```text
//! Publishers                         Subscribers
//! +----------------+                 +----------------+
//! | PatternStorage |--+              | Metrics        |
//! +----------------+  |              +----------------+
//!                     |   EventBus            ^
//! +----------------+  | +--------+            |
//! | SonaLearner    |---->| tx/rx |------------+
//! +----------------+  | +--------+            |
//!                     |                       v
//! +----------------+  |              +----------------+
//! | SyncService    |--+              | Audit Logger   |
//! +----------------+                 +----------------+
//! ```
//!
//! # Example
//!
//! ```ignore
//! use nagual::events::{EventBus, NagualEvent};
//!
//! // Create the event bus
//! let event_bus = EventBus::new();
//!
//! // Subscribe to events
//! let mut receiver = event_bus.subscribe();
//! tokio::spawn(async move {
//!     while let Ok(event) = receiver.recv().await {
//!         println!("Received event: {:?}", event.event_type());
//!     }
//! });
//!
//! // Publish an event
//! let event = NagualEvent::pattern_stored("pattern-123", "rust.async");
//! event_bus.publish(event).await?;
//! ```

pub mod socket;
pub mod types;

pub use types::{EventId, NagualEvent, PatternChanges, SyncType};

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use chrono::{DateTime, Utc};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::sync::broadcast;
use tracing::{debug, info, warn};

/// Default buffer size for the event channel.
pub const DEFAULT_BUFFER_SIZE: usize = 1000;

/// Errors that can occur in event bus operations.
#[derive(Error, Debug, Clone)]
pub enum EventBusError {
    /// Channel is closed
    #[error("Event bus channel is closed")]
    ChannelClosed,

    /// Channel is full (lagged)
    #[error("Event bus lagged, {missed_events} events were missed")]
    Lagged { missed_events: u64 },

    /// Event serialization error
    #[error("Event serialization error: {0}")]
    Serialization(String),
}

/// Result type for event bus operations.
pub type EventBusResult<T> = std::result::Result<T, EventBusError>;

/// Statistics about the event bus.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EventBusStats {
    /// Total events published
    pub events_published: u64,
    /// Events by type
    pub events_by_type: std::collections::HashMap<String, u64>,
    /// Current number of subscribers
    pub subscriber_count: usize,
    /// Number of lagged events (missed by slow subscribers)
    pub lagged_events: u64,
    /// Last event timestamp
    pub last_event_at: Option<DateTime<Utc>>,
    /// Bus creation time
    pub created_at: DateTime<Utc>,
}

/// A typed receiver for events from the event bus.
pub type EventReceiver = broadcast::Receiver<Arc<NagualEvent>>;

/// A typed sender for events to the event bus.
pub type EventSender = broadcast::Sender<Arc<NagualEvent>>;

/// The event bus for publishing and subscribing to domain events.
///
/// Uses Tokio's broadcast channel to allow multiple publishers and
/// multiple subscribers with a configurable buffer size.
pub struct EventBus {
    /// The broadcast sender
    sender: EventSender,
    /// Event statistics (shared between clones)
    stats: Arc<RwLock<EventBusStats>>,
    /// Counter for published events (shared between clones)
    published_count: Arc<AtomicU64>,
    /// Counter for lagged events (shared between clones)
    lagged_count: Arc<AtomicU64>,
    /// Buffer size
    buffer_size: usize,
}

impl EventBus {
    /// Create a new event bus with the default buffer size (1000).
    pub fn new() -> Self {
        Self::with_buffer_size(DEFAULT_BUFFER_SIZE)
    }

    /// Create a new event bus with a custom buffer size.
    ///
    /// The buffer size determines how many events can be held before
    /// slow subscribers start missing events.
    pub fn with_buffer_size(buffer_size: usize) -> Self {
        let (sender, _) = broadcast::channel(buffer_size);

        let stats = EventBusStats {
            created_at: Utc::now(),
            ..Default::default()
        };

        Self {
            sender,
            stats: Arc::new(RwLock::new(stats)),
            published_count: Arc::new(AtomicU64::new(0)),
            lagged_count: Arc::new(AtomicU64::new(0)),
            buffer_size,
        }
    }

    /// Publish an event to all subscribers.
    ///
    /// This is an async method that wraps the event in an Arc and sends
    /// it through the broadcast channel. It updates statistics after
    /// successful publication.
    ///
    /// # Returns
    ///
    /// The number of receivers that received the event.
    pub async fn publish(&self, event: NagualEvent) -> EventBusResult<usize> {
        let event_type = event.event_type().to_string();
        let event_timestamp = event.timestamp();
        let event_arc = Arc::new(event);

        let receiver_count = self.sender.send(event_arc).map_err(|_| {
            warn!("Event bus has no subscribers");
            EventBusError::ChannelClosed
        })?;

        // Update statistics
        self.published_count.fetch_add(1, Ordering::Relaxed);

        {
            let mut stats = self.stats.write();
            stats.events_published += 1;
            *stats.events_by_type.entry(event_type.clone()).or_insert(0) += 1;
            stats.last_event_at = Some(event_timestamp);
        }

        debug!(
            event_type = %event_type,
            receiver_count = receiver_count,
            "Published event"
        );

        Ok(receiver_count)
    }

    /// Publish an event without waiting (fire and forget).
    ///
    /// This is a synchronous method useful when you don't need
    /// to wait for the result or count receivers.
    pub fn publish_sync(&self, event: NagualEvent) {
        let event_type = event.event_type().to_string();
        let event_timestamp = event.timestamp();
        let event_arc = Arc::new(event);

        match self.sender.send(event_arc) {
            Ok(count) => {
                self.published_count.fetch_add(1, Ordering::Relaxed);

                let mut stats = self.stats.write();
                stats.events_published += 1;
                *stats.events_by_type.entry(event_type.clone()).or_insert(0) += 1;
                stats.last_event_at = Some(event_timestamp);

                debug!(
                    event_type = %event_type,
                    receiver_count = count,
                    "Published event (sync)"
                );
            }
            Err(_) => {
                warn!(event_type = %event_type, "No subscribers for event");
            }
        }
    }

    /// Subscribe to receive events from the bus.
    ///
    /// Returns a receiver that will receive all events published
    /// after the subscription was created.
    pub fn subscribe(&self) -> EventReceiver {
        let receiver = self.sender.subscribe();

        // Update subscriber count in stats
        {
            let mut stats = self.stats.write();
            stats.subscriber_count = self.sender.receiver_count();
        }

        info!(
            subscriber_count = self.sender.receiver_count(),
            "New event bus subscriber"
        );

        receiver
    }

    /// Get the current number of active subscribers.
    pub fn subscriber_count(&self) -> usize {
        self.sender.receiver_count()
    }

    /// Get the buffer size.
    pub fn buffer_size(&self) -> usize {
        self.buffer_size
    }

    /// Get event bus statistics.
    pub fn stats(&self) -> EventBusStats {
        let mut stats = self.stats.read().clone();
        stats.subscriber_count = self.sender.receiver_count();
        stats.lagged_events = self.lagged_count.load(Ordering::Relaxed);
        stats
    }

    /// Record that a subscriber lagged (missed events).
    pub fn record_lag(&self, missed_events: u64) {
        self.lagged_count.fetch_add(missed_events, Ordering::Relaxed);

        let mut stats = self.stats.write();
        stats.lagged_events += missed_events;

        warn!(
            missed_events = missed_events,
            total_lagged = stats.lagged_events,
            "Subscriber lagged, events missed"
        );
    }

    /// Get the total number of published events.
    pub fn total_published(&self) -> u64 {
        self.published_count.load(Ordering::Relaxed)
    }

    /// Check if the event bus has any active subscribers.
    pub fn has_subscribers(&self) -> bool {
        self.sender.receiver_count() > 0
    }

    /// Attach a Unix socket transport to this event bus.
    /// Returns the transport handle for management.
    pub fn attach_socket_transport(
        &self,
        path: impl Into<std::path::PathBuf>,
    ) -> (socket::UnixSocketTransport, tokio::task::JoinHandle<()>) {
        let transport = socket::UnixSocketTransport::with_path(path);
        let handle = transport.start(Arc::new(self.clone()));
        (transport, handle)
    }
}

impl Default for EventBus {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for EventBus {
    fn clone(&self) -> Self {
        Self {
            sender: self.sender.clone(),
            stats: Arc::clone(&self.stats),
            published_count: Arc::clone(&self.published_count),
            lagged_count: Arc::clone(&self.lagged_count),
            buffer_size: self.buffer_size,
        }
    }
}

/// A filtered event receiver that only receives events matching a predicate.
pub struct FilteredReceiver<F>
where
    F: Fn(&NagualEvent) -> bool + Send,
{
    receiver: EventReceiver,
    filter: F,
}

impl<F> FilteredReceiver<F>
where
    F: Fn(&NagualEvent) -> bool + Send,
{
    /// Create a new filtered receiver.
    pub fn new(receiver: EventReceiver, filter: F) -> Self {
        Self { receiver, filter }
    }

    /// Receive the next event that matches the filter.
    ///
    /// Blocks until an event matching the filter is received.
    pub async fn recv(&mut self) -> Result<Arc<NagualEvent>, broadcast::error::RecvError> {
        loop {
            let event = self.receiver.recv().await?;
            if (self.filter)(&event) {
                return Ok(event);
            }
        }
    }
}

/// Utility functions for creating common event filters.
pub mod filters {
    use super::*;

    /// Filter for pattern-related events.
    pub fn pattern_events(event: &NagualEvent) -> bool {
        matches!(
            event,
            NagualEvent::PatternStored { .. }
                | NagualEvent::PatternUpdated { .. }
                | NagualEvent::PatternDeleted { .. }
        )
    }

    /// Filter for prediction-related events.
    pub fn prediction_events(event: &NagualEvent) -> bool {
        matches!(
            event,
            NagualEvent::PredictionCreated { .. } | NagualEvent::PredictionResolved { .. }
        )
    }

    /// Filter for health-related events.
    pub fn health_events(event: &NagualEvent) -> bool {
        matches!(event, NagualEvent::HealthChanged { .. })
    }

    /// Filter for error events.
    pub fn error_events(event: &NagualEvent) -> bool {
        matches!(event, NagualEvent::ErrorOccurred { .. })
    }

    /// Filter for SONA learning events.
    pub fn learning_events(event: &NagualEvent) -> bool {
        matches!(
            event,
            NagualEvent::OutcomeRecorded { .. } | NagualEvent::ConsolidationCompleted { .. }
        )
    }

    /// Filter for sync events.
    pub fn sync_events(event: &NagualEvent) -> bool {
        matches!(event, NagualEvent::SyncCompleted { .. })
    }

    /// Create a filter for a specific event type.
    pub fn by_type(event_type: &'static str) -> impl Fn(&NagualEvent) -> bool {
        move |event| event.event_type() == event_type
    }

    /// Create a filter for events within a time range.
    pub fn within_time_range(
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> impl Fn(&NagualEvent) -> bool {
        move |event| {
            let ts = event.timestamp();
            ts >= start && ts <= end
        }
    }
}

/// Event handler trait for processing events.
#[async_trait::async_trait]
pub trait EventHandler: Send + Sync {
    /// Handle an event.
    async fn handle(&self, event: &NagualEvent);

    /// Get the handler name.
    fn name(&self) -> &str;

    /// Check if this handler should process the given event.
    fn should_handle(&self, event: &NagualEvent) -> bool {
        let _ = event;
        true
    }
}

/// An event processor that runs handlers in a background task.
pub struct EventProcessor {
    event_bus: Arc<EventBus>,
    handlers: Arc<RwLock<Vec<Arc<dyn EventHandler>>>>,
    shutdown: tokio::sync::watch::Sender<bool>,
}

impl EventProcessor {
    /// Create a new event processor.
    pub fn new(event_bus: Arc<EventBus>) -> Self {
        let (shutdown_tx, _) = tokio::sync::watch::channel(false);

        Self {
            event_bus,
            handlers: Arc::new(RwLock::new(Vec::new())),
            shutdown: shutdown_tx,
        }
    }

    /// Register an event handler.
    pub fn register_handler(&self, handler: Arc<dyn EventHandler>) {
        let mut handlers = self.handlers.write();
        info!(handler_name = handler.name(), "Registered event handler");
        handlers.push(handler);
    }

    /// Start processing events in the background.
    pub fn start(&self) -> tokio::task::JoinHandle<()> {
        let mut receiver = self.event_bus.subscribe();
        let handlers = Arc::clone(&self.handlers);
        let mut shutdown_rx = self.shutdown.subscribe();

        tokio::spawn(async move {
            loop {
                tokio::select! {
                    result = receiver.recv() => {
                        match result {
                            Ok(event) => {
                                // Clone handlers to avoid holding lock across await
                                let handler_list: Vec<Arc<dyn EventHandler>> = {
                                    handlers.read().clone()
                                };
                                for handler in handler_list.iter() {
                                    if handler.should_handle(&event) {
                                        handler.handle(&event).await;
                                    }
                                }
                            }
                            Err(broadcast::error::RecvError::Lagged(n)) => {
                                warn!(missed_events = n, "Event processor lagged");
                            }
                            Err(broadcast::error::RecvError::Closed) => {
                                info!("Event bus closed, stopping processor");
                                break;
                            }
                        }
                    }
                    _ = shutdown_rx.changed() => {
                        if *shutdown_rx.borrow() {
                            info!("Shutdown signal received, stopping event processor");
                            break;
                        }
                    }
                }
            }
        })
    }

    /// Signal the processor to shut down.
    pub fn shutdown(&self) {
        let _ = self.shutdown.send(true);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[tokio::test]
    async fn test_event_bus_publish_subscribe() {
        let bus = EventBus::new();
        let mut receiver = bus.subscribe();

        // Publish an event
        let event = NagualEvent::pattern_stored("test-123", "test.domain");
        let count = bus.publish(event).await.unwrap();
        assert_eq!(count, 1);

        // Receive the event
        let received = receiver.recv().await.unwrap();
        assert_eq!(received.event_type(), "pattern_stored");
    }

    #[tokio::test]
    async fn test_event_bus_multiple_subscribers() {
        let bus = EventBus::new();
        let mut receiver1 = bus.subscribe();
        let mut receiver2 = bus.subscribe();

        assert_eq!(bus.subscriber_count(), 2);

        // Publish an event
        let event = NagualEvent::pattern_deleted("test-456");
        let count = bus.publish(event).await.unwrap();
        assert_eq!(count, 2);

        // Both should receive
        let r1 = receiver1.recv().await.unwrap();
        let r2 = receiver2.recv().await.unwrap();
        assert_eq!(r1.event_type(), "pattern_deleted");
        assert_eq!(r2.event_type(), "pattern_deleted");
    }

    #[tokio::test]
    async fn test_event_bus_stats() {
        let bus = EventBus::new();
        let _receiver = bus.subscribe();

        // Publish several events
        bus.publish(NagualEvent::pattern_stored("1", "d1")).await.unwrap();
        bus.publish(NagualEvent::pattern_stored("2", "d2")).await.unwrap();
        bus.publish(NagualEvent::pattern_deleted("1")).await.unwrap();

        let stats = bus.stats();
        assert_eq!(stats.events_published, 3);
        assert_eq!(stats.events_by_type.get("pattern_stored"), Some(&2));
        assert_eq!(stats.events_by_type.get("pattern_deleted"), Some(&1));
        assert_eq!(stats.subscriber_count, 1);
    }

    #[tokio::test]
    async fn test_event_bus_publish_sync() {
        let bus = EventBus::new();
        let mut receiver = bus.subscribe();

        bus.publish_sync(NagualEvent::pattern_stored("sync-test", "domain"));

        // Give some time for async processing
        tokio::time::sleep(Duration::from_millis(10)).await;

        let received = tokio::time::timeout(Duration::from_millis(100), receiver.recv())
            .await
            .unwrap()
            .unwrap();

        assert_eq!(received.event_type(), "pattern_stored");
    }

    #[tokio::test]
    async fn test_event_bus_no_subscribers() {
        let bus = EventBus::new();

        // Should not panic, just log warning
        let event = NagualEvent::pattern_stored("test", "domain");
        let result = bus.publish(event).await;

        // With no subscribers, send returns error
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_event_filters() {
        let event_pattern = NagualEvent::pattern_stored("id", "domain");
        let event_prediction = NagualEvent::prediction_created("id", 0.5, 0.5, "domain", 1);
        let event_health = NagualEvent::health_changed(
            "component",
            crate::health::HealthStatus::Healthy,
            crate::health::HealthStatus::Degraded,
            "message",
        );

        assert!(filters::pattern_events(&event_pattern));
        assert!(!filters::pattern_events(&event_prediction));

        assert!(filters::prediction_events(&event_prediction));
        assert!(!filters::prediction_events(&event_pattern));

        assert!(filters::health_events(&event_health));
        assert!(!filters::health_events(&event_pattern));
    }

    #[tokio::test]
    async fn test_event_bus_clone() {
        let bus = EventBus::new();
        let mut receiver = bus.subscribe();

        let bus_clone = bus.clone();

        // Publish from clone
        bus_clone
            .publish(NagualEvent::pattern_stored("clone-test", "domain"))
            .await
            .unwrap();

        // Original receiver should receive
        let received = receiver.recv().await.unwrap();
        assert_eq!(received.event_type(), "pattern_stored");

        // Stats should be synced (through Arc)
        assert_eq!(bus.total_published(), 1);
        assert_eq!(bus_clone.total_published(), 1);
    }

    #[tokio::test]
    async fn test_event_processor() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        struct TestHandler {
            count: AtomicUsize,
        }

        #[async_trait::async_trait]
        impl EventHandler for TestHandler {
            async fn handle(&self, _event: &NagualEvent) {
                self.count.fetch_add(1, Ordering::SeqCst);
            }

            fn name(&self) -> &str {
                "test_handler"
            }
        }

        let bus = Arc::new(EventBus::new());
        let processor = EventProcessor::new(Arc::clone(&bus));

        let handler = Arc::new(TestHandler {
            count: AtomicUsize::new(0),
        });
        processor.register_handler(Arc::clone(&handler) as Arc<dyn EventHandler>);

        let _task = processor.start();

        // Publish events
        bus.publish(NagualEvent::pattern_stored("1", "d")).await.unwrap();
        bus.publish(NagualEvent::pattern_stored("2", "d")).await.unwrap();

        // Give processor time to handle
        tokio::time::sleep(Duration::from_millis(50)).await;

        assert_eq!(handler.count.load(Ordering::SeqCst), 2);

        processor.shutdown();
    }
}
