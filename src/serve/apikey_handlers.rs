//! Auth-related HTTP handlers (whoami, API login).

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::Json;
use tracing::warn;

use crate::db::users::session;

use super::auth::{AuthIdentity, RequireAuth};
use super::AppState;

/// GET /api/auth/whoami — returns the identity of the current caller.
pub async fn api_whoami(
    RequireAuth(identity): RequireAuth,
) -> Json<serde_json::Value> {
    match &identity {
        AuthIdentity::Master => Json(serde_json::json!({
            "identity": "master",
            "type": "master",
            "scopes": ["read", "write", "admin"],
        })),
        AuthIdentity::Key { name, scopes, .. } => Json(serde_json::json!({
            "identity": name,
            "type": "api_key",
            "scopes": scopes,
        })),
        AuthIdentity::Session { username, role } => {
            let scopes = if role == "admin" {
                vec!["read", "write", "admin"]
            } else {
                vec!["read"]
            };
            Json(serde_json::json!({
                "identity": username,
                "type": "session",
                "role": role,
                "scopes": scopes,
            }))
        }
        AuthIdentity::LocalOnly => Json(serde_json::json!({
            "identity": "local",
            "type": "local",
            "scopes": ["read", "write", "admin"],
        })),
    }
}

/// Login request body.
#[derive(serde::Deserialize)]
pub struct ApiLoginRequest {
    pub username: String,
    pub password: String,
}

/// POST /api/auth/login — authenticate with username/password, get a session token.
///
/// Returns a signed session token that can be used as `Authorization: Bearer <token>`
/// or as the `nagual_session` cookie. This lets agents log in programmatically
/// with the same credentials used for the dashboard.
pub async fn api_auth_login(
    State(state): State<AppState>,
    Json(req): Json<ApiLoginRequest>,
) -> impl IntoResponse {
    let store = match &state.user_store {
        Some(s) => s,
        None => {
            return (StatusCode::SERVICE_UNAVAILABLE, Json(serde_json::json!({
                "error": "Login not configured (no users exist)"
            }))).into_response();
        }
    };

    match store.verify_user(&req.username, &req.password).await {
        Ok(Some(user)) => {
            let token = session::create_cookie(&user.username, &user.role, &state.session_secret);
            let scopes = if user.role == "admin" {
                vec!["read", "write", "admin"]
            } else {
                vec!["read"]
            };
            Json(serde_json::json!({
                "token": token,
                "identity": user.username,
                "role": user.role,
                "scopes": scopes,
                "usage": "Use as: Authorization: Bearer <token>"
            })).into_response()
        }
        Ok(None) => {
            (StatusCode::UNAUTHORIZED, Json(serde_json::json!({
                "error": "Invalid username or password"
            }))).into_response()
        }
        Err(e) => {
            warn!("API login error: {}", e);
            (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({
                "error": "Login error"
            }))).into_response()
        }
    }
}
