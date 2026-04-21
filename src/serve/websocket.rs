//! WebSocket handler for real-time event streaming.
//!
//! Subscribes to the EventBus and forwards NagualEvents as JSON
//! to connected WebSocket clients. Supports multiple concurrent connections
//! and handles disconnection gracefully.

use std::sync::Arc;
use std::time::Duration;

use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::State;
use axum::response::IntoResponse;
use futures_util::{SinkExt, StreamExt};
use tokio::sync::broadcast;
use tracing::{debug, info, warn};

use super::auth::RequireAuth;
use super::AppState;

/// WebSocket upgrade handler (auth required).
///
/// Accepts a WebSocket connection and spawns a task to stream events
/// from the EventBus to the client.
pub async fn ws_handler(
    _auth: RequireAuth,
    ws: WebSocketUpgrade,
    State(state): State<AppState>,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_socket(socket, state))
}

/// Handle an established WebSocket connection.
///
/// Subscribes to the event bus and forwards events as JSON messages.
/// Also sends periodic heartbeat pings and responds to close frames.
async fn handle_socket(socket: WebSocket, state: AppState) {
    let (mut sender, mut receiver) = socket.split();

    let event_bus = Arc::clone(&state.event_bus);
    let mut event_rx = event_bus.subscribe();

    info!("WebSocket client connected");

    // Send initial connection confirmation
    let welcome = serde_json::json!({
        "type": "connected",
        "message": "Nagual WebSocket connected",
        "timestamp": chrono::Utc::now().to_rfc3339(),
    });
    if let Ok(msg) = serde_json::to_string(&welcome) {
        let _ = sender.send(Message::Text(msg.into())).await;
    }

    // Spawn a task to handle incoming messages from the client (pong, close)
    let (close_tx, mut close_rx) = tokio::sync::oneshot::channel::<()>();

    tokio::spawn(async move {
        while let Some(msg) = receiver.next().await {
            match msg {
                Ok(Message::Close(_)) => {
                    debug!("WebSocket client sent close frame");
                    break;
                }
                Ok(Message::Pong(_)) => {
                    debug!("WebSocket pong received");
                }
                Ok(_) => {}
                Err(e) => {
                    debug!("WebSocket receive error: {}", e);
                    break;
                }
            }
        }
        let _ = close_tx.send(());
    });

    // Main event forwarding loop with heartbeat
    let mut heartbeat_interval = tokio::time::interval(Duration::from_secs(30));
    heartbeat_interval.tick().await; // consume the immediate first tick

    loop {
        tokio::select! {
            // Forward events from the event bus
            event_result = event_rx.recv() => {
                match event_result {
                    Ok(event) => {
                        match serde_json::to_string(event.as_ref()) {
                            Ok(json) => {
                                if sender.send(Message::Text(json.into())).await.is_err() {
                                    debug!("WebSocket send failed, client disconnected");
                                    break;
                                }
                            }
                            Err(e) => {
                                warn!("Failed to serialize event for WebSocket: {}", e);
                            }
                        }
                    }
                    Err(broadcast::error::RecvError::Lagged(n)) => {
                        warn!("WebSocket client lagged, missed {} events", n);
                        let lag_msg = serde_json::json!({
                            "type": "lagged",
                            "missed_events": n,
                        });
                        if let Ok(json) = serde_json::to_string(&lag_msg) {
                            let _ = sender.send(Message::Text(json.into())).await;
                        }
                    }
                    Err(broadcast::error::RecvError::Closed) => {
                        info!("Event bus closed, closing WebSocket");
                        break;
                    }
                }
            }

            // Heartbeat ping
            _ = heartbeat_interval.tick() => {
                let heartbeat = serde_json::json!({
                    "type": "heartbeat",
                    "timestamp": chrono::Utc::now().to_rfc3339(),
                });
                if let Ok(json) = serde_json::to_string(&heartbeat) {
                    if sender.send(Message::Text(json.into())).await.is_err() {
                        debug!("Heartbeat send failed, client disconnected");
                        break;
                    }
                }
            }

            // Client closed the connection
            _ = &mut close_rx => {
                debug!("WebSocket client closed connection");
                break;
            }
        }
    }

    info!("WebSocket client disconnected");
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_websocket_module_compiles() {
        // Verify that the module compiles correctly.
        // WebSocket handlers are hard to unit test without a running server,
        // so integration tests should cover the actual functionality.
        assert!(true);
    }
}
