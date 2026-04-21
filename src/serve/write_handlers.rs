//! Write API endpoints for pattern CRUD operations.
//!
//! Mutating endpoints are gated by `RequireWrite` (requires `write` scope).
//! Read-only endpoints (search, get by ID) require `RequireAuth` (any valid identity).

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::{Deserialize, Serialize};
use tracing::{info, warn};

use super::auth::{RequireAuth, RequireWrite};
use super::AppState;
use crate::error::NagualError;
#[allow(unused_imports)]
use rusqlite;
use crate::events::NagualEvent;
use crate::reasoning_bank::pattern::{FailureMode, Pattern, PatternCategory, PatternId};

/// API error wrapper mapping NagualError to HTTP status codes.
#[derive(Debug)]
pub(crate) struct ApiError(NagualError);

impl From<NagualError> for ApiError {
    fn from(err: NagualError) -> Self {
        ApiError(err)
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let (status, message) = match &self.0 {
            NagualError::Database(_) => (StatusCode::INTERNAL_SERVER_ERROR, self.0.to_string()),
            NagualError::Config { .. } => (StatusCode::BAD_REQUEST, self.0.to_string()),
            NagualError::Serde(_) => (StatusCode::BAD_REQUEST, self.0.to_string()),
            NagualError::Internal { .. } => (StatusCode::INTERNAL_SERVER_ERROR, self.0.to_string()),
            _ => (StatusCode::INTERNAL_SERVER_ERROR, self.0.to_string()),
        };

        warn!(status = %status, error = %message, "API error");

        let body = serde_json::json!({
            "error": message,
            "code": self.0.error_code(),
        });

        (status, Json(body)).into_response()
    }
}

// --- Request/Response types ---

/// Request body for storing a new pattern.
#[derive(Debug, Deserialize)]
pub struct StorePatternRequest {
    pub problem: String,
    pub solution: String,
    #[serde(default)]
    pub context: Option<String>,
    #[serde(default)]
    pub domain: Option<String>,
    #[serde(default)]
    pub tags: Option<Vec<String>>,
    #[serde(default)]
    pub confidence: Option<f32>,
    #[serde(default)]
    pub agent_id: Option<String>,
    #[serde(default)]
    pub session_id: Option<String>,
    /// KOS P0: Parent pattern ID for lineage tracking.
    #[serde(default)]
    pub parent_id: Option<String>,
    /// KOS P0: Derivation type (original, merge, consolidation, improvement, fork, transfer).
    #[serde(default)]
    pub derivation_type: Option<String>,
}

/// Request body for searching patterns.
#[derive(Debug, Deserialize)]
pub struct SearchPatternsRequest {
    pub query: String,
    #[serde(default = "default_search_limit")]
    pub limit: usize,
    #[serde(default)]
    pub domain: Option<String>,
}

fn default_search_limit() -> usize {
    10
}

/// Request body for recording an outcome.
#[derive(Debug, Deserialize)]
pub struct RecordOutcomeRequest {
    /// "success" or "failure"
    pub outcome: String,
    #[serde(default)]
    pub feedback: Option<String>,
    #[serde(default)]
    pub failure_mode: Option<String>,
}

/// Request body for updating a pattern.
#[derive(Debug, Deserialize)]
pub struct UpdatePatternRequest {
    #[serde(default)]
    pub problem: Option<String>,
    #[serde(default)]
    pub solution: Option<String>,
    #[serde(default)]
    pub context: Option<String>,
    #[serde(default)]
    pub domain: Option<String>,
    #[serde(default)]
    pub tags: Option<Vec<String>>,
}

/// Response for a stored/retrieved pattern.
#[derive(Debug, Serialize)]
pub struct PatternResponse {
    pub id: String,
    pub problem: String,
    pub solution: String,
    pub context: String,
    pub domain: String,
    pub tags: Vec<String>,
    pub reward: f32,
    pub confidence: f32,
    pub success: bool,
    pub reuse_count: u32,
    pub effectiveness: f32,
    pub created_at: String,
    pub updated_at: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub agent_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content_hash: Option<String>,
}

impl From<&Pattern> for PatternResponse {
    fn from(p: &Pattern) -> Self {
        PatternResponse {
            id: p.id().to_string(),
            problem: p.problem().to_string(),
            solution: p.solution().to_string(),
            context: p.context().to_string(),
            domain: p.category().to_string(),
            tags: p.tags().to_vec(),
            reward: p.reward(),
            confidence: p.confidence(),
            success: p.success(),
            reuse_count: p.reuse_count(),
            effectiveness: p.effectiveness(),
            created_at: p.timestamp().to_rfc3339(),
            updated_at: p.updated_at().to_rfc3339(),
            agent_id: p.agent_id().map(|s| s.to_string()),
            session_id: p.session_id().map(|s| s.to_string()),
            content_hash: p.content_hash().map(|s| s.to_string()),
        }
    }
}

// --- Handlers ---

/// POST /api/patterns — Store a new pattern (auth required).
pub async fn api_store_pattern(
    State(state): State<AppState>,
    _auth: RequireWrite,
    Json(req): Json<StorePatternRequest>,
) -> Result<impl IntoResponse, ApiError> {
    let storage_mutex = state.storage.as_ref().ok_or_else(|| {
        NagualError::internal("Storage not initialized")
    })?;
    let storage = storage_mutex.lock().await;

    let mut builder = Pattern::builder()
        .problem(&req.problem)
        .solution(&req.solution);

    if let Some(ref ctx) = req.context {
        builder = builder.context(ctx);
    }
    if let Some(ref domain) = req.domain {
        builder = builder.category(PatternCategory::from(domain.as_str()));
    }
    if let Some(ref tags) = req.tags {
        builder = builder.tags(tags.clone());
    }
    if let Some(confidence) = req.confidence {
        builder = builder.confidence(confidence);
    }
    if let Some(ref agent_id) = req.agent_id {
        builder = builder.agent_id(agent_id);
    }
    if let Some(ref session_id) = req.session_id {
        builder = builder.session_id(session_id);
    }

    let pattern = builder.build();
    let id = storage.store_pattern(&pattern).await?;

    let domain = req.domain.clone().unwrap_or_else(|| "general".to_string());
    state.event_bus.publish_sync(NagualEvent::pattern_stored(
        id.to_string(),
        domain,
    ));

    info!(pattern_id = %id, "Pattern stored via API");

    let resp = serde_json::json!({
        "id": id.to_string(),
        "status": "stored",
    });

    Ok((StatusCode::CREATED, Json(resp)))
}

/// POST /api/patterns/search — Full-text search (auth required).
pub async fn api_search_patterns(
    State(state): State<AppState>,
    _auth: RequireAuth,
    Json(req): Json<SearchPatternsRequest>,
) -> Result<impl IntoResponse, ApiError> {
    let storage_mutex = state.storage.as_ref().ok_or_else(|| {
        NagualError::internal("Storage not initialized")
    })?;
    let storage = storage_mutex.lock().await;

    let results = storage.fts_search(&req.query, req.limit).await?;

    // Filter by domain if specified
    let filtered: Vec<PatternResponse> = results
        .iter()
        .filter(|p| {
            req.domain.as_ref().map_or(true, |d| {
                p.category().to_string().eq_ignore_ascii_case(d)
            })
        })
        .map(PatternResponse::from)
        .collect();

    // Collect pattern IDs for usage recording (done after releasing the lock)
    let pattern_ids: Vec<String> = results.iter().map(|p| p.id().to_string()).collect();
    drop(storage); // Release the mutex before spawning

    // Record usage for returned patterns (feeds auto-promotion engine)
    // Fire-and-forget via spawn_blocking with a direct SQLite connection
    if !pattern_ids.is_empty() {
        let db_path = state.db_path.clone();
        tokio::task::spawn_blocking(move || {
            if let Ok(conn) = rusqlite::Connection::open(&db_path) {
                for pid in &pattern_ids {
                    let _ = conn.execute(
                        "INSERT INTO pattern_usage_log (pattern_id, session_id, task_id, outcome) VALUES (?, '', '', 'retrieval')",
                        rusqlite::params![pid],
                    );
                }
            }
        });
    }

    Ok(Json(serde_json::json!({
        "query": req.query,
        "count": filtered.len(),
        "patterns": filtered,
    })))
}

/// GET /api/patterns/:id — Get a single pattern (auth required).
#[axum::debug_handler]
pub async fn api_get_pattern(
    State(state): State<AppState>,
    _auth: RequireAuth,
    Path(id): Path<String>,
) -> Result<impl IntoResponse, ApiError> {
    let storage_mutex = state.storage.as_ref().ok_or_else(|| {
        NagualError::internal("Storage not initialized")
    })?;
    let storage = storage_mutex.lock().await;

    let pattern_id = PatternId::from(id.as_str());
    match storage.get_pattern(&pattern_id).await? {
        Some(pattern) => {
            let resp = PatternResponse::from(&pattern);
            Ok(Json(serde_json::json!(resp)).into_response())
        }
        None => {
            let body = serde_json::json!({ "error": "Pattern not found", "id": id });
            Ok((StatusCode::NOT_FOUND, Json(body)).into_response())
        }
    }
}

/// DELETE /api/patterns/:id — Delete a pattern (auth required).
pub async fn api_delete_pattern(
    State(state): State<AppState>,
    _auth: RequireWrite,
    Path(id): Path<String>,
) -> Result<impl IntoResponse, ApiError> {
    let storage_mutex = state.storage.as_ref().ok_or_else(|| {
        NagualError::internal("Storage not initialized")
    })?;
    let storage = storage_mutex.lock().await;

    let pattern_id = PatternId::from(id.as_str());

    // Verify pattern exists before deleting
    let exists = storage.get_pattern(&pattern_id).await?.is_some();
    if !exists {
        let body = serde_json::json!({ "error": "Pattern not found", "id": id });
        return Ok((StatusCode::NOT_FOUND, Json(body)).into_response());
    }

    storage.delete_pattern(&pattern_id).await?;

    state.event_bus.publish_sync(NagualEvent::pattern_deleted(id.clone()));

    info!(pattern_id = %id, "Pattern deleted via API");

    Ok(Json(serde_json::json!({
        "id": id,
        "status": "deleted",
    }))
    .into_response())
}

/// POST /api/patterns/:id/outcome — Record success/failure (auth required).
pub async fn api_record_outcome(
    State(state): State<AppState>,
    _auth: RequireWrite,
    Path(id): Path<String>,
    Json(req): Json<RecordOutcomeRequest>,
) -> Result<impl IntoResponse, ApiError> {
    let storage_mutex = state.storage.as_ref().ok_or_else(|| {
        NagualError::internal("Storage not initialized")
    })?;
    let storage = storage_mutex.lock().await;

    let pattern_id = PatternId::from(id.as_str());
    let mut pattern = storage
        .get_pattern(&pattern_id)
        .await?
        .ok_or_else(|| NagualError::internal(format!("Pattern not found: {}", id)))?;

    let is_success = req.outcome.eq_ignore_ascii_case("success");

    // Update reward: boost on success, decay on failure
    let current_reward = pattern.reward();
    let new_reward = if is_success {
        (current_reward + 0.1).min(1.0)
    } else {
        (current_reward - 0.15).max(0.0)
    };

    pattern.set_reward(new_reward);
    pattern.set_success(is_success);

    if let Some(ref feedback) = req.feedback {
        pattern.set_critique(feedback);
    }

    if !is_success {
        if let Some(ref fm) = req.failure_mode {
            let mode = match fm.to_lowercase().as_str() {
                "specification" => FailureMode::SpecificationIssue,
                "misalignment" => FailureMode::InterAgentMisalignment,
                "verification" => FailureMode::TaskVerification,
                "resource" => FailureMode::ResourceIssue,
                _ => FailureMode::Unknown,
            };
            pattern.set_failure_mode(mode);
        }
    }

    storage.update_pattern(&pattern).await?;

    state.event_bus.publish_sync(NagualEvent::outcome_recorded(
        id.clone(),
        req.outcome.clone(),
        new_reward,
        req.feedback.clone(),
    ));

    info!(
        pattern_id = %id,
        outcome = %req.outcome,
        new_reward = new_reward,
        "Outcome recorded via API"
    );

    Ok(Json(serde_json::json!({
        "id": id,
        "outcome": req.outcome,
        "reward": new_reward,
        "status": "recorded",
    })))
}

/// PUT /api/patterns/:id — Update pattern fields (auth required).
pub async fn api_update_pattern(
    State(state): State<AppState>,
    _auth: RequireWrite,
    Path(id): Path<String>,
    Json(req): Json<UpdatePatternRequest>,
) -> Result<impl IntoResponse, ApiError> {
    let storage_mutex = state.storage.as_ref().ok_or_else(|| {
        NagualError::internal("Storage not initialized")
    })?;
    let storage = storage_mutex.lock().await;

    let pattern_id = PatternId::from(id.as_str());
    let existing = storage
        .get_pattern(&pattern_id)
        .await?
        .ok_or_else(|| NagualError::internal(format!("Pattern not found: {}", id)))?;

    // Rebuild pattern with updated fields
    let mut builder = Pattern::builder()
        .id(existing.id().clone())
        .timestamp(existing.timestamp())
        .problem(req.problem.as_deref().unwrap_or(existing.problem()))
        .solution(req.solution.as_deref().unwrap_or(existing.solution()))
        .context(req.context.as_deref().unwrap_or(existing.context()))
        .category(
            req.domain
                .as_ref()
                .map(|d| PatternCategory::from(d.as_str()))
                .unwrap_or_else(|| existing.category().clone()),
        )
        .reward(existing.reward())
        .confidence(existing.confidence())
        .success(existing.success())
        .effectiveness(existing.effectiveness())
        .reuse_count(existing.reuse_count());

    if let Some(ref tags) = req.tags {
        builder = builder.tags(tags.clone());
    } else {
        builder = builder.tags(existing.tags().to_vec());
    }

    if let Some(agent_id) = existing.agent_id() {
        builder = builder.agent_id(agent_id);
    }
    if let Some(session_id) = existing.session_id() {
        builder = builder.session_id(session_id);
    }
    if let Some(critique) = Some(existing.critique()).filter(|c| !c.is_empty()) {
        builder = builder.critique(critique);
    }
    if let Some(ref fm) = existing.failure_mode() {
        builder = builder.failure_mode((*fm).clone());
    }
    if let Some(embedding) = existing.embedding() {
        builder = builder.embedding(embedding.to_vec());
    }

    let updated = builder.build();
    storage.update_pattern(&updated).await?;

    // Build change descriptor
    let mut changes = crate::events::types::PatternChanges::new();
    if req.problem.is_some() {
        changes = changes.with_field("problem");
    }
    if req.solution.is_some() {
        changes = changes.with_field("solution");
    }
    if req.context.is_some() {
        changes = changes.with_field("context");
    }
    if req.domain.is_some() {
        changes = changes.with_field("domain");
    }
    if req.tags.is_some() {
        changes = changes.with_field("tags");
    }

    state.event_bus.publish_sync(NagualEvent::pattern_updated(id.clone(), changes));

    info!(pattern_id = %id, "Pattern updated via API");

    let resp = PatternResponse::from(&updated);
    Ok(Json(serde_json::json!({
        "status": "updated",
        "pattern": resp,
    })))
}
