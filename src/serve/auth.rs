//! Authentication extractors for API endpoints.
//!
//! Supports three identity types:
//! - **Master token**: static `--api-token` / `NAGUAL_API_TOKEN` (backward-compat)
//! - **API key**: per-agent `ngk_*` keys stored in SQLite, scoped to read/write/admin
//! - **Local-only**: no token configured, all requests allowed
//!
//! `RequireAuth` checks identity (any valid token).
//! `RequireWrite` additionally verifies the `write` scope.

use axum::extract::FromRequestParts;
use axum::http::request::Parts;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use ring::constant_time;

use crate::db::users::session;

use super::AppState;

/// The authenticated identity of a request.
#[derive(Debug, Clone)]
pub enum AuthIdentity {
    /// Static master token from `--api-token`.
    Master,
    /// Per-agent API key with scopes.
    Key {
        id: String,
        name: String,
        scopes: Vec<String>,
    },
    /// Browser session via signed cookie.
    Session {
        username: String,
        role: String,
    },
    /// No authentication configured (local-only mode).
    LocalOnly,
}

impl AuthIdentity {
    /// Check if this identity has a given scope.
    pub fn has_scope(&self, scope: &str) -> bool {
        match self {
            AuthIdentity::Master | AuthIdentity::LocalOnly => true,
            AuthIdentity::Key { scopes, .. } => {
                scopes.iter().any(|s| s == scope || s == "admin")
            }
            AuthIdentity::Session { role, .. } => {
                // admin sessions have all scopes; viewer sessions have read only
                role == "admin" || scope == "read"
            }
        }
    }

    /// Human-readable identity type.
    pub fn identity_type(&self) -> &'static str {
        match self {
            AuthIdentity::Master => "master",
            AuthIdentity::Key { .. } => "api_key",
            AuthIdentity::Session { .. } => "session",
            AuthIdentity::LocalOnly => "local",
        }
    }

    /// Human-readable name.
    pub fn identity_name(&self) -> &str {
        match self {
            AuthIdentity::Master => "master",
            AuthIdentity::Key { name, .. } => name.as_str(),
            AuthIdentity::Session { username, .. } => username.as_str(),
            AuthIdentity::LocalOnly => "local",
        }
    }
}

/// Authentication error.
#[derive(Debug)]
pub enum AuthError {
    /// No Authorization header present.
    Missing,
    /// Token does not match any valid credential.
    Invalid,
    /// Token is valid but lacks the required scope.
    Forbidden,
}

impl IntoResponse for AuthError {
    fn into_response(self) -> Response {
        match self {
            AuthError::Missing => (
                StatusCode::UNAUTHORIZED,
                "Missing Authorization header. Use: Authorization: Bearer <token>",
            )
                .into_response(),
            AuthError::Invalid => {
                (StatusCode::FORBIDDEN, "Invalid bearer token").into_response()
            }
            AuthError::Forbidden => (
                StatusCode::FORBIDDEN,
                "Insufficient scope for this operation",
            )
                .into_response(),
        }
    }
}

/// Axum extractor that requires a valid identity (any scope).
///
/// If `AppState.auth_token` is `None` **and** no key store, all requests
/// pass through as `LocalOnly`. Otherwise the bearer token is checked
/// against the master token first (constant-time), then the key store.
pub struct RequireAuth(pub AuthIdentity);

impl FromRequestParts<AppState> for RequireAuth {
    type Rejection = AuthError;

    async fn from_request_parts(
        parts: &mut Parts,
        state: &AppState,
    ) -> std::result::Result<Self, Self::Rejection> {
        let has_master = state.auth_token.is_some();
        let has_key_store = state.key_store.is_some();

        // Local-only mode: no master token and no key store
        if !has_master && !has_key_store {
            return Ok(RequireAuth(AuthIdentity::LocalOnly));
        }

        // 1. Check session cookie first (browser auth — no Authorization header needed)
        if let Some(cookie_header) = parts
            .headers
            .get("cookie")
            .and_then(|v| v.to_str().ok())
        {
            if let Some(cookie_val) = session::extract_from_cookie_header(cookie_header) {
                if let Some((username, role)) =
                    session::verify_cookie(cookie_val, &state.session_secret)
                {
                    return Ok(RequireAuth(AuthIdentity::Session { username, role }));
                }
            }
        }

        // 2. Check for Authorization header (programmatic access)
        let auth_header = parts
            .headers
            .get("authorization")
            .and_then(|v| v.to_str().ok());

        let token = match auth_header {
            None => return Err(AuthError::Missing),
            Some(header) => {
                match header
                    .strip_prefix("Bearer ")
                    .or_else(|| header.strip_prefix("bearer "))
                {
                    Some(t) => t.to_string(),
                    None => return Err(AuthError::Invalid), // e.g. "Basic ..."
                }
            }
        };

        // 3. Check master token (constant-time comparison via ring)
        if let Some(ref expected) = state.auth_token {
            if constant_time::verify_slices_are_equal(
                token.as_bytes(),
                expected.as_bytes(),
            )
            .is_ok()
            {
                return Ok(RequireAuth(AuthIdentity::Master));
            }
        }

        // 4. Check API key store
        if let Some(ref key_store) = state.key_store {
            if let Ok(Some(record)) = key_store.validate_key(&token).await {
                // Fire-and-forget last_used update
                let store = key_store.clone();
                let key_id = record.id.clone();
                tokio::spawn(async move {
                    let _ = store.touch_last_used(&key_id).await;
                });

                return Ok(RequireAuth(AuthIdentity::Key {
                    id: record.id,
                    name: record.name,
                    scopes: record.scopes,
                }));
            }
        }

        // 5. Check if Bearer token is a session token (from POST /api/auth/login)
        if let Some((username, role)) =
            session::verify_cookie(&token, &state.session_secret)
        {
            return Ok(RequireAuth(AuthIdentity::Session { username, role }));
        }

        Err(AuthError::Invalid)
    }
}

/// Axum extractor that requires `write` scope (or master/local).
///
/// Delegates to `RequireAuth` then verifies the `write` scope.
pub struct RequireWrite(pub AuthIdentity);

impl FromRequestParts<AppState> for RequireWrite {
    type Rejection = AuthError;

    async fn from_request_parts(
        parts: &mut Parts,
        state: &AppState,
    ) -> std::result::Result<Self, Self::Rejection> {
        let RequireAuth(identity) = RequireAuth::from_request_parts(parts, state).await?;

        if identity.has_scope("write") {
            Ok(RequireWrite(identity))
        } else {
            Err(AuthError::Forbidden)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::security::ApiKeyStore;
    use axum::http::Request;
    use std::path::PathBuf;
    use std::sync::Arc;
    use crate::events::EventBus;

    fn test_state(token: Option<String>) -> AppState {
        AppState {
            db_path: PathBuf::from("./test.db"),
            event_bus: Arc::new(EventBus::new()),
            storage: None,
            auth_token: token,
            key_store: None,
            user_store: None,
            session_secret: vec![0u8; 32],
            login_required: false,
        }
    }

    #[tokio::test]
    async fn test_no_token_configured_allows_all() {
        let state = test_state(None);
        let req = Request::builder().body(()).unwrap();
        let (mut parts, _) = req.into_parts();

        let result = RequireAuth::from_request_parts(&mut parts, &state).await;
        assert!(result.is_ok());
        let RequireAuth(identity) = result.unwrap();
        assert!(matches!(identity, AuthIdentity::LocalOnly));
    }

    #[tokio::test]
    async fn test_valid_bearer_token() {
        let state = test_state(Some("secret123".to_string()));
        let req = Request::builder()
            .header("authorization", "Bearer secret123")
            .body(())
            .unwrap();
        let (mut parts, _) = req.into_parts();

        let result = RequireAuth::from_request_parts(&mut parts, &state).await;
        assert!(result.is_ok());
        let RequireAuth(identity) = result.unwrap();
        assert!(matches!(identity, AuthIdentity::Master));
    }

    #[tokio::test]
    async fn test_missing_header_returns_401() {
        let state = test_state(Some("secret123".to_string()));
        let req = Request::builder().body(()).unwrap();
        let (mut parts, _) = req.into_parts();

        let result = RequireAuth::from_request_parts(&mut parts, &state).await;
        assert!(matches!(result, Err(AuthError::Missing)));
    }

    #[tokio::test]
    async fn test_wrong_token_returns_403() {
        let state = test_state(Some("secret123".to_string()));
        let req = Request::builder()
            .header("authorization", "Bearer wrong-token")
            .body(())
            .unwrap();
        let (mut parts, _) = req.into_parts();

        let result = RequireAuth::from_request_parts(&mut parts, &state).await;
        assert!(matches!(result, Err(AuthError::Invalid)));
    }

    #[tokio::test]
    async fn test_malformed_auth_header() {
        let state = test_state(Some("secret123".to_string()));
        let req = Request::builder()
            .header("authorization", "Basic abc123")
            .body(())
            .unwrap();
        let (mut parts, _) = req.into_parts();

        let result = RequireAuth::from_request_parts(&mut parts, &state).await;
        assert!(matches!(result, Err(AuthError::Invalid)));
    }

    #[tokio::test]
    async fn test_api_key_auth() {
        let db = Arc::new(crate::db::SqliteDb::open_in_memory().unwrap());
        let store = Arc::new(ApiKeyStore::new(db).await.unwrap());

        let (plaintext, _) = store
            .create_key("test-agent", &["read".into(), "write".into()], None)
            .await
            .unwrap();

        let mut state = test_state(Some("master-secret".to_string()));
        state.key_store = Some(store);

        let req = Request::builder()
            .header("authorization", format!("Bearer {}", plaintext))
            .body(())
            .unwrap();
        let (mut parts, _) = req.into_parts();

        let result = RequireAuth::from_request_parts(&mut parts, &state).await;
        assert!(result.is_ok());
        let RequireAuth(identity) = result.unwrap();
        match identity {
            AuthIdentity::Key { name, scopes, .. } => {
                assert_eq!(name, "test-agent");
                assert!(scopes.contains(&"write".to_string()));
            }
            other => panic!("Expected Key identity, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_require_write_rejects_read_only() {
        let db = Arc::new(crate::db::SqliteDb::open_in_memory().unwrap());
        let store = Arc::new(ApiKeyStore::new(db).await.unwrap());

        let (plaintext, _) = store
            .create_key("read-only-agent", &["read".into()], None)
            .await
            .unwrap();

        let mut state = test_state(None);
        state.key_store = Some(store);

        let req = Request::builder()
            .header("authorization", format!("Bearer {}", plaintext))
            .body(())
            .unwrap();
        let (mut parts, _) = req.into_parts();

        let result = RequireWrite::from_request_parts(&mut parts, &state).await;
        assert!(matches!(result, Err(AuthError::Forbidden)));
    }

    #[tokio::test]
    async fn test_identity_has_scope() {
        let master = AuthIdentity::Master;
        assert!(master.has_scope("read"));
        assert!(master.has_scope("write"));
        assert!(master.has_scope("admin"));

        let key = AuthIdentity::Key {
            id: "x".into(),
            name: "x".into(),
            scopes: vec!["read".into()],
        };
        assert!(key.has_scope("read"));
        assert!(!key.has_scope("write"));

        let admin_session = AuthIdentity::Session {
            username: "admin".into(),
            role: "admin".into(),
        };
        assert!(admin_session.has_scope("read"));
        assert!(admin_session.has_scope("write"));

        let viewer_session = AuthIdentity::Session {
            username: "viewer".into(),
            role: "viewer".into(),
        };
        assert!(viewer_session.has_scope("read"));
        assert!(!viewer_session.has_scope("write"));

        let local = AuthIdentity::LocalOnly;
        assert!(local.has_scope("anything"));
    }
}
