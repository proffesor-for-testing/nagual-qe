//! Serve command - browser-based dashboard for Nagual.
//!
//! Provides a web dashboard served by an axum HTTP server with
//! WebSocket real-time updates. The HTML file is embedded in the
//! Rust binary via `include_str!`.
//!
//! # Usage
//!
//! ```bash
//! nagual serve --port 3333
//! nagual serve --port 8080 --db-path ./custom.db --open
//! ```

pub mod action_handlers;
mod apikey_handlers;
mod auth;
pub mod compaction;
mod handlers;
pub mod heartbeat;
mod sync_handlers;
mod websocket;
mod write_handlers;

use std::path::PathBuf;
use std::sync::Arc;

use axum::extract::State;
use axum::http::{header, StatusCode};
use axum::response::{Html, IntoResponse, Redirect, Response};
use axum::routing::{delete, get, post, put};
use axum::Router;
use clap::Args;
use tower_http::cors::{Any, CorsLayer};
use tracing::{info, warn};

use crate::cli::common::resolve_postgres_url;
use crate::db::pg_notify::PgNotifyListener;
use crate::db::users::{session, UserStore};
use crate::error::Result;
use crate::events::socket::UnixSocketTransport;
use crate::events::EventBus;
use crate::reasoning_bank::storage::PatternStorage;
use crate::security::ApiKeyStore;
use tokio::sync::Mutex as TokioMutex;

/// Embedded dashboard HTML file.
const DASHBOARD_HTML: &str = include_str!("dashboard.html");

/// Embedded login HTML file.
const LOGIN_HTML: &str = include_str!("login.html");

/// Shared application state for all handlers.
#[derive(Clone)]
pub struct AppState {
    /// Path to the SQLite database.
    pub db_path: PathBuf,
    /// Event bus for WebSocket subscriptions.
    pub event_bus: Arc<EventBus>,
    /// Pattern storage for write API endpoints (None if not initialized).
    /// Wrapped in TokioMutex because PatternStorage contains rusqlite::Connection
    /// (via DualWriteAdapter) which is Send but not Sync.
    pub storage: Option<Arc<TokioMutex<PatternStorage>>>,
    /// Bearer token for write endpoint authentication.
    pub auth_token: Option<String>,
    /// API key store for per-agent authentication.
    pub key_store: Option<Arc<ApiKeyStore>>,
    /// User store for dashboard login (None if not initialized).
    pub user_store: Option<Arc<UserStore>>,
    /// Session secret for HMAC-signing cookies.
    pub session_secret: Vec<u8>,
    /// Whether login is required (true if users exist).
    pub login_required: bool,
}

/// Serve command - starts a browser-based dashboard.
///
/// Launches an axum HTTP server that serves a single-page dashboard
/// with REST API endpoints for data and WebSocket for real-time updates.
#[derive(Args, Debug)]
pub struct ServeCommand {
    /// Port to listen on.
    #[arg(long, default_value = "3333")]
    pub port: u16,

    /// Path to SQLite database.
    #[arg(long, default_value = "./nagual.db")]
    pub db_path: PathBuf,

    /// Auto-open the dashboard in the default browser.
    #[arg(long)]
    pub open: bool,

    /// PostgreSQL URL for pg_notify real-time event bridge.
    /// Falls back to DATABASE_URL env or ~/.nagual/config.toml.
    #[arg(long)]
    pub postgres_url: Option<String>,

    /// Unix domain socket path for inter-process event delivery.
    #[arg(long, default_value = "/tmp/nagual-events.sock")]
    pub socket_path: String,

    /// Bearer token for write API authentication.
    /// If not set, write endpoints are open (local-only mode).
    #[arg(long, env = "NAGUAL_API_TOKEN")]
    pub api_token: Option<String>,

    /// Session secret for signing login cookies.
    /// If not set, a random secret is generated (sessions won't survive restarts).
    #[arg(long, env = "NAGUAL_SESSION_SECRET")]
    pub session_secret: Option<String>,

    /// Heartbeat interval in minutes (0 to disable).
    #[arg(long, default_value = "30")]
    pub heartbeat_interval: u32,
}

impl ServeCommand {
    /// Execute the serve command.
    pub async fn run(&self) -> Result<()> {
        let event_bus = Arc::new(EventBus::new());

        // --- F17: Start Unix domain socket transport ---
        let socket_transport = UnixSocketTransport::with_path(&self.socket_path);
        let _socket_handle = socket_transport.start(Arc::clone(&event_bus));

        // --- Initialize pattern storage for write API ---
        let pg_url = resolve_postgres_url(self.postgres_url.as_deref());
        let storage = match crate::cli::common::init_storage(
            &self.db_path,
            pg_url.as_deref(),
        )
        .await
        {
            Ok(s) => Some(Arc::new(TokioMutex::new(s))),
            Err(e) => {
                warn!("Failed to initialize pattern storage: {} (write API disabled)", e);
                None
            }
        };
        let write_api_enabled = storage.is_some();

        // --- F03: Start pg_notify listener if PostgreSQL is configured ---
        let pg_handle = if let Some(ref url) = pg_url {
            let pg_listener = PgNotifyListener::new(url.as_str(), Arc::clone(&event_bus));
            match pg_listener.start().await {
                Ok(handle) => {
                    info!("pg_notify listener started");
                    Some(handle)
                }
                Err(e) => {
                    warn!("pg_notify listener failed to start: {} (continuing without)", e);
                    None
                }
            }
        } else {
            None
        };

        // Initialize API key store from the same SQLite DB
        let key_store = match crate::db::SqliteDb::open(&self.db_path) {
            Ok(db) => match ApiKeyStore::new(Arc::new(db)).await {
                Ok(ks) => {
                    info!("API key store initialized");
                    Some(Arc::new(ks))
                }
                Err(e) => {
                    warn!("Failed to initialize API key store: {} (key auth disabled)", e);
                    None
                }
            },
            Err(e) => {
                warn!("Failed to open DB for key store: {} (key auth disabled)", e);
                None
            }
        };

        // Initialize user store for dashboard login
        let (user_store, login_required) = match crate::db::SqliteDb::open(&self.db_path) {
            Ok(db) => {
                let db = Arc::new(db);
                match UserStore::new(db).await {
                    Ok(store) => {
                        let has_users = store.has_users().await.unwrap_or(false);
                        if has_users {
                            info!("Dashboard login enabled ({} user(s) configured)", "1+");
                        } else {
                            info!("Dashboard login disabled (no users — create with: nagual user create <name> --role admin)");
                        }
                        (Some(Arc::new(store)), has_users)
                    }
                    Err(e) => {
                        warn!("Failed to initialize user store: {} (login disabled)", e);
                        (None, false)
                    }
                }
            }
            Err(e) => {
                warn!("Failed to open DB for user store: {} (login disabled)", e);
                (None, false)
            }
        };

        // Session secret: from CLI/env or random (ephemeral)
        let session_secret = match &self.session_secret {
            Some(s) => s.as_bytes().to_vec(),
            None => {
                let secret = session::generate_secret();
                if login_required {
                    info!("Using ephemeral session secret (sessions won't survive restarts). Set NAGUAL_SESSION_SECRET for persistence.");
                }
                secret
            }
        };

        // --- Start heartbeat ---
        let heartbeat_handle = if self.heartbeat_interval > 0 {
            let hb_config = heartbeat::HeartbeatConfig {
                interval: std::time::Duration::from_secs(self.heartbeat_interval as u64 * 60),
                ..heartbeat::HeartbeatConfig::default()
            };
            let handle = heartbeat::start_heartbeat(
                hb_config,
                self.db_path.clone(),
                Arc::clone(&event_bus),
                std::time::Instant::now(),
            );
            info!(interval_min = self.heartbeat_interval, "Heartbeat started");
            Some(handle)
        } else {
            None
        };

        let state = AppState {
            db_path: self.db_path.clone(),
            event_bus: Arc::clone(&event_bus),
            storage,
            auth_token: self.api_token.clone(),
            key_store,
            user_store,
            session_secret,
            login_required,
        };

        let cors = CorsLayer::new()
            .allow_origin(Any)
            .allow_methods(Any)
            .allow_headers(Any);

        let app = Router::new()
            // Dashboard HTML (session-gated)
            .route("/", get(serve_dashboard))
            // Login / Logout
            .route("/login", get(serve_login).post(handle_login))
            .route("/logout", get(handle_logout))
            // REST API endpoints (read-only dashboard)
            .route("/api/status", get(handlers::api_status))
            .route("/api/patterns", get(handlers::api_patterns)
                .post(write_handlers::api_store_pattern))
            .route("/api/patterns/search", post(write_handlers::api_search_patterns))
            .route("/api/patterns/bulk", post(action_handlers::api_patterns_bulk))
            .route("/api/patterns/:id", get(write_handlers::api_get_pattern)
                .put(write_handlers::api_update_pattern)
                .delete(write_handlers::api_delete_pattern))
            .route("/api/patterns/:id/outcome", post(write_handlers::api_record_outcome))
            .route("/api/patterns/:id/history", get(action_handlers::api_pattern_history))
            .route("/api/patterns/:id/archive", post(action_handlers::api_pattern_archive))
            .route("/api/patterns/:id/promote", post(action_handlers::api_pattern_promote))
            .route("/api/domains", get(handlers::api_domains))
            .route("/api/tiers", get(handlers::api_tiers))
            .route("/api/pulse", get(handlers::api_pulse))
            .route("/api/graph", get(handlers::api_graph))
            .route("/api/graph/3d", get(handlers::api_graph_3d))
            // Sync endpoints (cloud push/pull)
            .route("/api/sync/push", post(sync_handlers::api_sync_push))
            .route("/api/sync/pull", get(sync_handlers::api_sync_pull))
            // Auth endpoints
            .route("/api/auth/login", post(apikey_handlers::api_auth_login))
            .route("/api/auth/whoami", get(apikey_handlers::api_whoami))
            // Compaction endpoint
            .route("/api/compaction/flush", post(compaction::compaction_flush_handler))
            // Action endpoints (learning jobs)
            .route("/api/actions/embed", post(action_handlers::api_action_embed))
            .route("/api/actions/consolidate", post(action_handlers::api_action_consolidate))
            .route("/api/actions/dedup", post(action_handlers::api_action_dedup))
            .route("/api/actions/pyramid", post(action_handlers::api_action_pyramid))
            .route("/api/actions/status/:job_id", get(action_handlers::api_action_status))
            .route("/api/actions/jobs", get(action_handlers::api_action_jobs))
            // Insights & recommendations
            .route("/api/insights", get(action_handlers::api_insights))
            .route("/api/recommendations", get(action_handlers::api_recommendations))
            // Phase 2: Pattern management
            .route("/api/tags", get(action_handlers::api_tags))
            // Phase 3: Intelligence & Visualization
            .route("/api/search/semantic", post(action_handlers::api_semantic_search))
            .route("/api/domains/stats", get(action_handlers::api_domain_stats))
            .route("/api/graph/nodes", get(action_handlers::api_graph_nodes))
            .route("/api/graph/edges", get(action_handlers::api_graph_edges))
            .route("/api/predictions", get(action_handlers::api_predictions_list)
                .post(action_handlers::api_predictions_create))
            .route("/api/predictions/calibration", get(action_handlers::api_predictions_calibration))
            .route("/api/predictions/:id/resolve", put(action_handlers::api_predictions_resolve))
            .route("/api/sessions/stats", get(action_handlers::api_session_stats))
            .route("/api/surprise", get(action_handlers::api_surprise_patterns))
            // Phase 4: Automation & Integration
            .route("/api/schedule", get(action_handlers::api_schedule_list)
                .post(action_handlers::api_schedule_create))
            .route("/api/schedule/:id", delete(action_handlers::api_schedule_delete))
            .route("/api/webhook/learn", post(action_handlers::api_webhook_learn))
            .route("/api/events/recent", get(action_handlers::api_events_recent))
            .route("/api/health/detailed", get(action_handlers::api_health_detailed))
            .route("/api/export", get(action_handlers::api_export))
            .route("/api/import", post(action_handlers::api_import))
            // WebSocket endpoint
            .route("/ws", get(websocket::ws_handler))
            .layer(cors)
            .with_state(state);

        let addr = format!("0.0.0.0:{}", self.port);
        let listener = tokio::net::TcpListener::bind(&addr).await.map_err(|e| {
            crate::error::NagualError::Internal { message: format!("Failed to bind to {}: {}", addr, e) }
        })?;

        info!(port = self.port, "Nagual dashboard running");
        println!();
        println!("  Nagual Dashboard");
        println!("  ================");
        println!("  URL:       http://localhost:{}", self.port);
        println!("  Database:  {}", self.db_path.display());
        println!("  WebSocket: ws://localhost:{}/ws", self.port);
        println!("  Socket:    {}", self.socket_path);
        println!("  Write API: {}", if write_api_enabled { "enabled" } else { "disabled (storage init failed)" });
        println!("  Auth:      {}", if self.api_token.is_some() { "bearer token required" } else { "open (local-only mode)" });
        println!("  Login:     {}", if login_required { "enabled (users configured)" } else { "disabled (no users — nagual user create <name> --role admin)" });
        println!("  Heartbeat: {}", if self.heartbeat_interval > 0 { format!("every {} min", self.heartbeat_interval) } else { "disabled".to_string() });
        if let Some(ref url) = pg_url {
            let masked = mask_pg_url(url);
            println!("  pg_notify: {}", if pg_handle.is_some() {
                format!("listening ({})", masked)
            } else {
                format!("failed ({})", masked)
            });
        } else {
            println!("  pg_notify: not configured");
        }
        println!();
        println!("  Press Ctrl+C to stop.");
        println!();

        if self.open {
            let url = format!("http://localhost:{}", self.port);
            #[cfg(target_os = "macos")]
            {
                let _ = std::process::Command::new("open").arg(&url).spawn();
            }
            #[cfg(target_os = "linux")]
            {
                let _ = std::process::Command::new("xdg-open").arg(&url).spawn();
            }
        }

        // Run server with graceful shutdown
        axum::serve(listener, app)
            .with_graceful_shutdown(shutdown_signal())
            .await
            .map_err(|e| crate::error::NagualError::Internal { message: format!("Server error: {}", e) })?;

        // Cleanup
        info!("Shutting down...");
        if let Some(handle) = heartbeat_handle {
            handle.abort();
        }
        socket_transport.stop();
        if let Some(handle) = pg_handle {
            handle.stop();
            handle.join().await;
        }

        Ok(())
    }
}

/// Handler that serves the embedded dashboard HTML.
/// If login is required and no valid session cookie exists, redirects to /login.
async fn serve_dashboard(
    State(state): State<AppState>,
    headers: axum::http::HeaderMap,
) -> Response {
    if state.login_required {
        // Check session cookie
        let has_session = headers
            .get(header::COOKIE)
            .and_then(|v| v.to_str().ok())
            .and_then(|cookie_header| session::extract_from_cookie_header(cookie_header))
            .and_then(|cookie_val| session::verify_cookie(cookie_val, &state.session_secret))
            .is_some();

        if !has_session {
            return Redirect::to("/login").into_response();
        }
    }
    Html(DASHBOARD_HTML).into_response()
}

/// Serve the login page HTML.
async fn serve_login() -> Html<&'static str> {
    Html(LOGIN_HTML)
}

/// Login request body (JSON).
#[derive(serde::Deserialize)]
struct LoginRequest {
    username: String,
    password: String,
}

/// Handle login submission (JSON body from fetch).
async fn handle_login(
    State(state): State<AppState>,
    axum::Json(req): axum::Json<LoginRequest>,
) -> Response {
    let store = match &state.user_store {
        Some(s) => s,
        None => {
            return (StatusCode::SERVICE_UNAVAILABLE, "Login not configured")
                .into_response();
        }
    };

    match store.verify_user(&req.username, &req.password).await {
        Ok(Some(user)) => {
            let cookie_value =
                session::create_cookie(&user.username, &user.role, &state.session_secret);
            let set_cookie = session::set_cookie_header(&cookie_value);
            // Return 200 with Set-Cookie — the JS will redirect to /
            Response::builder()
                .status(StatusCode::OK)
                .header(header::SET_COOKIE, set_cookie)
                .header(header::CONTENT_TYPE, "text/plain")
                .body(axum::body::Body::from("ok"))
                .unwrap()
        }
        Ok(None) => {
            (StatusCode::UNAUTHORIZED, "Invalid username or password").into_response()
        }
        Err(e) => {
            warn!("Login error: {}", e);
            (StatusCode::INTERNAL_SERVER_ERROR, "Login error").into_response()
        }
    }
}

/// Handle logout — clear session cookie and redirect to login.
async fn handle_logout() -> Response {
    let clear_cookie = session::clear_cookie_header();
    Response::builder()
        .status(StatusCode::SEE_OTHER)
        .header(header::LOCATION, "/login")
        .header(header::SET_COOKIE, clear_cookie)
        .body(axum::body::Body::empty())
        .unwrap()
}

/// Wait for Ctrl+C signal for graceful shutdown.
async fn shutdown_signal() {
    tokio::signal::ctrl_c()
        .await
        .expect("failed to listen for ctrl_c");
    info!("Ctrl+C received, initiating graceful shutdown");
}

/// Mask password in a PostgreSQL URL for display.
fn mask_pg_url(url: &str) -> String {
    // postgres://user:password@host:port/db -> postgres://user:***@host:port/db
    if let Some(at_pos) = url.find('@') {
        if let Some(colon_pos) = url[..at_pos].rfind(':') {
            let prefix = &url[..colon_pos + 1];
            let suffix = &url[at_pos..];
            return format!("{}***{}", prefix, suffix);
        }
    }
    url.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dashboard_html_embedded() {
        assert!(!DASHBOARD_HTML.is_empty(), "Dashboard HTML should be embedded");
        assert!(
            DASHBOARD_HTML.contains("<!DOCTYPE html>"),
            "Dashboard should be valid HTML"
        );
    }

    #[test]
    fn test_app_state_clone() {
        let state = AppState {
            db_path: PathBuf::from("./test.db"),
            event_bus: Arc::new(EventBus::new()),
            storage: None,
            auth_token: None,
            key_store: None,
            user_store: None,
            session_secret: vec![0u8; 32],
            login_required: false,
        };
        let cloned = state.clone();
        assert_eq!(cloned.db_path, state.db_path);
    }

    #[test]
    fn test_serve_command_defaults() {
        use clap::Parser;

        #[derive(Parser, Debug)]
        struct TestCli {
            #[command(subcommand)]
            cmd: TestCmd,
        }

        #[derive(clap::Subcommand, Debug)]
        enum TestCmd {
            Serve(ServeCommand),
        }

        let args = vec!["test", "serve"];
        let cli = TestCli::try_parse_from(args).unwrap();
        match cli.cmd {
            TestCmd::Serve(cmd) => {
                assert_eq!(cmd.port, 3333);
                assert_eq!(cmd.db_path, PathBuf::from("./nagual.db"));
                assert!(!cmd.open);
                assert!(cmd.postgres_url.is_none());
                assert_eq!(cmd.socket_path, "/tmp/nagual-events.sock");
            }
        }
    }

    #[test]
    fn test_serve_command_custom_args() {
        use clap::Parser;

        #[derive(Parser, Debug)]
        struct TestCli {
            #[command(subcommand)]
            cmd: TestCmd,
        }

        #[derive(clap::Subcommand, Debug)]
        enum TestCmd {
            Serve(ServeCommand),
        }

        let args = vec![
            "test", "serve",
            "--port", "8080",
            "--db-path", "/tmp/test.db",
            "--open",
            "--postgres-url", "postgres://nagual:pass@localhost/nagual",
            "--socket-path", "/tmp/custom.sock",
        ];
        let cli = TestCli::try_parse_from(args).unwrap();
        match cli.cmd {
            TestCmd::Serve(cmd) => {
                assert_eq!(cmd.port, 8080);
                assert_eq!(cmd.db_path, PathBuf::from("/tmp/test.db"));
                assert!(cmd.open);
                assert_eq!(
                    cmd.postgres_url.as_deref(),
                    Some("postgres://nagual:pass@localhost/nagual")
                );
                assert_eq!(cmd.socket_path, "/tmp/custom.sock");
            }
        }
    }

    #[test]
    fn test_resolve_postgres_url_explicit() {
        let result = resolve_postgres_url(Some("postgres://test@localhost/db"));
        assert_eq!(result, Some("postgres://test@localhost/db".to_string()));
    }

    #[test]
    fn test_resolve_postgres_url_none() {
        // When no explicit URL and no env var, result depends on config.toml
        // Just verify it doesn't panic
        let _ = resolve_postgres_url(None);
    }

    #[test]
    fn test_mask_pg_url() {
        assert_eq!(
            mask_pg_url("postgres://nagual:secret@localhost:5432/nagual"),
            "postgres://nagual:***@localhost:5432/nagual"
        );
        // No password
        assert_eq!(
            mask_pg_url("postgres://localhost/nagual"),
            "postgres://localhost/nagual"
        );
    }
}
