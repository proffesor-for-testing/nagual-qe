//! Unix Domain Socket transport for the Nagual event bus.
//!
//! Provides sub-millisecond inter-process event delivery via NDJSON
//! (Newline-Delimited JSON) over Unix domain sockets.
//!
//! # Usage
//!
//! ```bash
//! # Listen to events from any terminal:
//! socat - UNIX-CONNECT:/tmp/nagual-events.sock
//! ```

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use parking_lot::RwLock;
use tokio::io::AsyncWriteExt;
use tokio::net::{UnixListener, UnixStream};
use tokio::sync::broadcast;
use tracing::{debug, error, info, warn};

use super::{EventBus, NagualEvent};

/// Default socket path for the Nagual event transport.
pub const DEFAULT_SOCKET_PATH: &str = "/tmp/nagual-events.sock";

/// Statistics for the socket transport.
#[derive(Debug, Clone, Default)]
pub struct SocketTransportStats {
    /// Total events sent to socket clients
    pub events_sent: u64,
    /// Total active connections
    pub active_connections: usize,
    /// Total connections accepted since start
    pub total_connections: u64,
    /// Events that failed to serialize
    pub serialization_errors: u64,
    /// Write errors (disconnected clients)
    pub write_errors: u64,
}

/// Unix Domain Socket transport for the event bus.
///
/// Creates a socket at the configured path and bridges events from
/// the EventBus broadcast channel to connected clients as NDJSON.
pub struct UnixSocketTransport {
    /// Path to the Unix domain socket
    socket_path: PathBuf,
    /// Whether the transport is running
    running: Arc<AtomicBool>,
    /// Statistics
    stats: Arc<RwLock<SocketTransportStats>>,
    /// Connection counter
    connection_count: Arc<AtomicU64>,
    /// Events sent counter
    events_sent: Arc<AtomicU64>,
}

impl UnixSocketTransport {
    /// Create a new Unix socket transport with the default path.
    pub fn new() -> Self {
        Self::with_path(DEFAULT_SOCKET_PATH)
    }

    /// Create a new Unix socket transport with a custom path.
    pub fn with_path(path: impl Into<PathBuf>) -> Self {
        Self {
            socket_path: path.into(),
            running: Arc::new(AtomicBool::new(false)),
            stats: Arc::new(RwLock::new(SocketTransportStats::default())),
            connection_count: Arc::new(AtomicU64::new(0)),
            events_sent: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Get the socket path.
    pub fn socket_path(&self) -> &Path {
        &self.socket_path
    }

    /// Check if the transport is running.
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::Relaxed)
    }

    /// Get transport statistics.
    pub fn stats(&self) -> SocketTransportStats {
        let mut stats = self.stats.read().clone();
        stats.events_sent = self.events_sent.load(Ordering::Relaxed);
        stats.total_connections = self.connection_count.load(Ordering::Relaxed);
        stats
    }

    /// Start the socket transport, bridging events from the event bus.
    ///
    /// This spawns a background task that:
    /// 1. Creates/cleans up the Unix socket
    /// 2. Accepts incoming connections
    /// 3. For each connection, spawns a writer task that subscribes
    ///    to the event bus and writes NDJSON events
    ///
    /// Returns a JoinHandle for the listener task.
    pub fn start(&self, event_bus: Arc<EventBus>) -> tokio::task::JoinHandle<()> {
        let socket_path = self.socket_path.clone();
        let running = Arc::clone(&self.running);
        let stats = Arc::clone(&self.stats);
        let connection_count = Arc::clone(&self.connection_count);
        let events_sent = Arc::clone(&self.events_sent);

        running.store(true, Ordering::Relaxed);

        tokio::spawn(async move {
            // Clean up stale socket
            let _ = std::fs::remove_file(&socket_path);

            // Bind the listener
            let listener = match UnixListener::bind(&socket_path) {
                Ok(l) => {
                    info!(path = %socket_path.display(), "Unix socket transport started");
                    l
                }
                Err(e) => {
                    error!(path = %socket_path.display(), error = %e, "Failed to bind Unix socket");
                    running.store(false, Ordering::Relaxed);
                    return;
                }
            };

            // Accept connections
            while running.load(Ordering::Relaxed) {
                tokio::select! {
                    result = listener.accept() => {
                        match result {
                            Ok((stream, _addr)) => {
                                let conn_num = connection_count.fetch_add(1, Ordering::Relaxed) + 1;
                                info!(connection = conn_num, "New socket client connected");

                                // Update active connections
                                {
                                    let mut s = stats.write();
                                    s.active_connections += 1;
                                }

                                // Subscribe to event bus and spawn writer
                                let mut receiver = event_bus.subscribe();
                                let stats_clone = Arc::clone(&stats);
                                let events_sent_clone = Arc::clone(&events_sent);

                                tokio::spawn(async move {
                                    handle_client(
                                        stream,
                                        &mut receiver,
                                        conn_num,
                                        &stats_clone,
                                        &events_sent_clone,
                                    )
                                    .await;

                                    // Decrement active connections
                                    let mut s = stats_clone.write();
                                    s.active_connections = s.active_connections.saturating_sub(1);
                                    info!(connection = conn_num, "Socket client disconnected");
                                });
                            }
                            Err(e) => {
                                warn!(error = %e, "Failed to accept socket connection");
                            }
                        }
                    }
                }
            }

            // Cleanup
            let _ = std::fs::remove_file(&socket_path);
            info!("Unix socket transport stopped");
        })
    }

    /// Stop the transport.
    pub fn stop(&self) {
        self.running.store(false, Ordering::Relaxed);
        // Remove socket file to unblock accept()
        let _ = std::fs::remove_file(&self.socket_path);
        info!(path = %self.socket_path.display(), "Unix socket transport stopping");
    }
}

impl Default for UnixSocketTransport {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for UnixSocketTransport {
    fn drop(&mut self) {
        if self.running.load(Ordering::Relaxed) {
            self.stop();
        }
    }
}

/// Handle a connected client by streaming NDJSON events.
async fn handle_client(
    mut stream: UnixStream,
    receiver: &mut broadcast::Receiver<Arc<NagualEvent>>,
    conn_num: u64,
    stats: &Arc<RwLock<SocketTransportStats>>,
    events_sent: &Arc<AtomicU64>,
) {
    loop {
        match receiver.recv().await {
            Ok(event) => {
                // Serialize event to JSON
                let json = match serde_json::to_string(event.as_ref()) {
                    Ok(j) => j,
                    Err(e) => {
                        warn!(connection = conn_num, error = %e, "Failed to serialize event");
                        {
                            let mut s = stats.write();
                            s.serialization_errors += 1;
                        }
                        continue;
                    }
                };

                // Write NDJSON (JSON + newline)
                let line = format!("{}\n", json);
                if let Err(e) = stream.write_all(line.as_bytes()).await {
                    debug!(connection = conn_num, error = %e, "Client write failed, disconnecting");
                    {
                        let mut s = stats.write();
                        s.write_errors += 1;
                    }
                    break;
                }

                events_sent.fetch_add(1, Ordering::Relaxed);
            }
            Err(broadcast::error::RecvError::Lagged(n)) => {
                warn!(connection = conn_num, missed = n, "Client lagged, skipping events");
            }
            Err(broadcast::error::RecvError::Closed) => {
                info!(connection = conn_num, "Event bus closed, disconnecting client");
                break;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    use tokio::io::AsyncBufReadExt;
    use tokio::io::BufReader;
    use tokio::net::UnixStream as TokioUnixStream;

    #[tokio::test]
    async fn test_socket_transport_lifecycle() {
        let socket_path = format!("/tmp/nagual-test-{}.sock", std::process::id());
        let transport = UnixSocketTransport::with_path(&socket_path);
        let bus = Arc::new(EventBus::new());

        assert!(!transport.is_running());

        let _handle = transport.start(Arc::clone(&bus));

        // Wait for socket to be created
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert!(transport.is_running());

        // Connect a client
        let stream = TokioUnixStream::connect(&socket_path).await.unwrap();
        let mut reader = BufReader::new(stream);

        // Give it a moment to register
        tokio::time::sleep(Duration::from_millis(50)).await;

        // Publish an event
        bus.publish(NagualEvent::pattern_stored("test-socket", "test.domain"))
            .await
            .unwrap();

        // Read the NDJSON line
        let mut line = String::new();
        let read_result = tokio::time::timeout(
            Duration::from_millis(500),
            reader.read_line(&mut line),
        )
        .await;

        assert!(read_result.is_ok());
        let bytes_read = read_result.unwrap().unwrap();
        assert!(bytes_read > 0);

        // Verify it's valid JSON with correct event type
        let event: NagualEvent = serde_json::from_str(line.trim()).unwrap();
        assert_eq!(event.event_type(), "pattern_stored");

        // Check stats
        let stats = transport.stats();
        assert!(stats.total_connections >= 1);
        assert!(stats.events_sent >= 1);

        // Cleanup
        transport.stop();
        tokio::time::sleep(Duration::from_millis(50)).await;

        // Socket file should be cleaned up
        assert!(!std::path::Path::new(&socket_path).exists());
    }

    #[tokio::test]
    async fn test_socket_transport_multiple_clients() {
        let socket_path = format!("/tmp/nagual-test-multi-{}.sock", std::process::id());
        let transport = UnixSocketTransport::with_path(&socket_path);
        let bus = Arc::new(EventBus::new());

        let _handle = transport.start(Arc::clone(&bus));
        tokio::time::sleep(Duration::from_millis(50)).await;

        // Connect two clients
        let stream1 = TokioUnixStream::connect(&socket_path).await.unwrap();
        let stream2 = TokioUnixStream::connect(&socket_path).await.unwrap();
        let mut reader1 = BufReader::new(stream1);
        let mut reader2 = BufReader::new(stream2);

        tokio::time::sleep(Duration::from_millis(50)).await;

        // Publish an event
        bus.publish(NagualEvent::pattern_stored("multi-test", "domain"))
            .await
            .unwrap();

        // Both clients should receive
        let mut line1 = String::new();
        let mut line2 = String::new();

        let r1 =
            tokio::time::timeout(Duration::from_millis(500), reader1.read_line(&mut line1)).await;
        let r2 =
            tokio::time::timeout(Duration::from_millis(500), reader2.read_line(&mut line2)).await;

        assert!(r1.is_ok() && r1.unwrap().unwrap() > 0);
        assert!(r2.is_ok() && r2.unwrap().unwrap() > 0);

        // Both should be valid JSON
        let _: NagualEvent = serde_json::from_str(line1.trim()).unwrap();
        let _: NagualEvent = serde_json::from_str(line2.trim()).unwrap();

        transport.stop();
        tokio::time::sleep(Duration::from_millis(50)).await;
    }

    #[test]
    fn test_socket_transport_default_path() {
        let transport = UnixSocketTransport::new();
        assert_eq!(
            transport.socket_path().to_str().unwrap(),
            DEFAULT_SOCKET_PATH
        );
    }

    #[test]
    fn test_socket_transport_custom_path() {
        let transport = UnixSocketTransport::with_path("/tmp/custom-nagual.sock");
        assert_eq!(
            transport.socket_path().to_str().unwrap(),
            "/tmp/custom-nagual.sock"
        );
    }

    #[test]
    fn test_socket_transport_stats_default() {
        let transport = UnixSocketTransport::new();
        let stats = transport.stats();
        assert_eq!(stats.events_sent, 0);
        assert_eq!(stats.active_connections, 0);
        assert_eq!(stats.total_connections, 0);
    }
}
