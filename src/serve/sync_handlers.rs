//! Sync API endpoints for cloud push/pull operations.
//!
//! Two endpoints gated by `RequireWrite`:
//! - `POST /api/sync/push` — receive patterns from a remote client
//! - `GET /api/sync/pull` — return patterns modified since a timestamp

use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::Json;
use chrono::{DateTime, Utc};
use serde::Deserialize;
use tracing::{debug, info, warn};

use super::auth::RequireWrite;
use super::write_handlers::ApiError;
use super::AppState;
use crate::cloud::types::{PullResponse, PushRequest, PushResponse, SyncPatternData};
use crate::error::NagualError;
use crate::reasoning_bank::pattern::PatternId;

/// POST /api/sync/push — Receive patterns from a remote client.
///
/// For each incoming pattern:
/// - If ID doesn't exist → store (create)
/// - If ID exists and incoming updated_at > server's → update
/// - If ID exists and incoming updated_at <= server's → skip (server wins)
pub async fn api_sync_push(
    State(state): State<AppState>,
    _auth: RequireWrite,
    Json(req): Json<PushRequest>,
) -> Result<impl IntoResponse, ApiError> {
    let storage_mutex = state.storage.as_ref().ok_or_else(|| {
        NagualError::internal("Storage not initialized")
    })?;
    let storage = storage_mutex.lock().await;

    let received = req.patterns.len();
    let mut created = 0usize;
    let mut updated = 0usize;
    let mut skipped = 0usize;

    for sync_data in &req.patterns {
        let pattern_id = PatternId::from_string(&sync_data.id);

        // Check if pattern exists on server
        match storage.get_pattern(&pattern_id).await {
            Ok(Some(existing)) => {
                // Compare updated_at timestamps
                let incoming_ts = DateTime::parse_from_rfc3339(&sync_data.updated_at)
                    .map(|dt| dt.with_timezone(&Utc))
                    .unwrap_or_else(|_| Utc::now());

                // Check if incoming data differs in content (reward, effectiveness, confidence)
                let content_differs = (sync_data.reward - existing.reward()).abs() > 0.001
                    || (sync_data.effectiveness - existing.effectiveness()).abs() > 0.001
                    || (sync_data.confidence - existing.confidence()).abs() > 0.001
                    || sync_data.reuse_count != existing.reuse_count();

                if incoming_ts > existing.updated_at() || content_differs {
                    // Incoming is newer or has different content — update
                    let pattern = sync_data.to_pattern();
                    match storage.update_pattern(&pattern).await {
                        Ok(_) => {
                            updated += 1;
                            debug!(id = %sync_data.id, "Pattern updated via sync push");
                        }
                        Err(e) => {
                            warn!(id = %sync_data.id, error = %e, "Failed to update pattern during sync push");
                            skipped += 1;
                        }
                    }
                } else {
                    // Server is same or newer with same content — skip
                    skipped += 1;
                }
            }
            Ok(None) => {
                // Pattern doesn't exist — create
                let pattern = sync_data.to_pattern();
                match storage.store_pattern(&pattern).await {
                    Ok(_) => {
                        created += 1;
                        debug!(id = %sync_data.id, "Pattern created via sync push");
                    }
                    Err(e) => {
                        warn!(id = %sync_data.id, error = %e, "Failed to store pattern during sync push");
                        skipped += 1;
                    }
                }
            }
            Err(e) => {
                warn!(id = %sync_data.id, error = %e, "Failed to check pattern during sync push");
                skipped += 1;
            }
        }
    }

    info!(
        received = received,
        created = created,
        updated = updated,
        skipped = skipped,
        "Sync push completed"
    );

    let resp = PushResponse {
        received,
        created,
        updated,
        skipped,
    };

    Ok((StatusCode::OK, Json(resp)))
}

/// Query parameters for the pull endpoint.
#[derive(Debug, Deserialize)]
pub struct PullParams {
    /// Only return patterns modified after this RFC3339 timestamp.
    pub since: Option<String>,
    /// Maximum patterns per page (default: 100).
    #[serde(default = "default_pull_limit")]
    pub limit: usize,
    /// Offset for pagination (default: 0).
    #[serde(default)]
    pub offset: usize,
}

fn default_pull_limit() -> usize {
    100
}

/// GET /api/sync/pull — Return patterns modified since a timestamp.
pub async fn api_sync_pull(
    State(state): State<AppState>,
    _auth: RequireWrite,
    Query(params): Query<PullParams>,
) -> Result<impl IntoResponse, ApiError> {
    let since = params
        .since
        .as_ref()
        .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
        .map(|dt| dt.with_timezone(&Utc));

    let limit = params.limit.min(500); // Cap at 500 per page
    let offset = params.offset;

    // Query patterns from SQLite directly for efficiency
    let db_path = state.db_path.clone();
    let since_clone = since;

    let (patterns, total) = tokio::task::spawn_blocking(move || {
        query_patterns_since_blocking(&db_path, since_clone, limit, offset)
    })
    .await
    .map_err(|e| NagualError::internal(format!("Task join error: {}", e)))??;

    let has_more = offset + patterns.len() < total;

    debug!(
        total = total,
        returned = patterns.len(),
        has_more = has_more,
        "Sync pull served"
    );

    let resp = PullResponse {
        patterns,
        total,
        has_more,
    };

    Ok(Json(resp))
}

/// Query patterns since a timestamp (blocking, for use with spawn_blocking).
fn query_patterns_since_blocking(
    db_path: &std::path::Path,
    since: Option<DateTime<Utc>>,
    limit: usize,
    offset: usize,
) -> crate::error::Result<(Vec<SyncPatternData>, usize)> {
    let conn = rusqlite::Connection::open(db_path)
        .map_err(|e| NagualError::internal(format!("Failed to open DB: {}", e)))?;

    // Get total count
    let total: usize = if let Some(ref since) = since {
        let since_str = since.to_rfc3339();
        conn.query_row(
            "SELECT COUNT(*) FROM reasoning_patterns WHERE updated_at > ?",
            rusqlite::params![since_str],
            |row| row.get::<_, i64>(0),
        )
        .map_err(|e| NagualError::internal(format!("Count query failed: {}", e)))?
            as usize
    } else {
        conn.query_row(
            "SELECT COUNT(*) FROM reasoning_patterns",
            [],
            |row| row.get::<_, i64>(0),
        )
        .map_err(|e| NagualError::internal(format!("Count query failed: {}", e)))?
            as usize
    };

    // Query patterns
    let (sql, params): (String, Vec<Box<dyn rusqlite::types::ToSql>>) = if let Some(ref since) = since {
        let since_str = since.to_rfc3339();
        (
            format!(
                "SELECT id, problem, solution, context, category, tags, reward, confidence, \
                 success, reuse_count, effectiveness, timestamp, updated_at, agent_id, \
                 session_id, content_hash, critique \
                 FROM reasoning_patterns WHERE updated_at > ? \
                 ORDER BY updated_at ASC LIMIT {} OFFSET {}",
                limit, offset
            ),
            vec![Box::new(since_str) as Box<dyn rusqlite::types::ToSql>],
        )
    } else {
        (
            format!(
                "SELECT id, problem, solution, context, category, tags, reward, confidence, \
                 success, reuse_count, effectiveness, timestamp, updated_at, agent_id, \
                 session_id, content_hash, critique \
                 FROM reasoning_patterns \
                 ORDER BY updated_at ASC LIMIT {} OFFSET {}",
                limit, offset
            ),
            vec![],
        )
    };

    let mut stmt = conn
        .prepare(&sql)
        .map_err(|e| NagualError::internal(format!("Prepare failed: {}", e)))?;

    let param_refs: Vec<&dyn rusqlite::types::ToSql> = params.iter().map(|p| p.as_ref()).collect();

    let rows = stmt
        .query_map(param_refs.as_slice(), |row| {
            let id: String = row.get(0)?;
            let problem: String = row.get(1)?;
            let solution: String = row.get(2)?;
            let context: String = row.get::<_, Option<String>>(3)?.unwrap_or_default();
            let domain: String = row.get(4)?;
            let tags_json: String = row.get::<_, Option<String>>(5)?.unwrap_or_else(|| "[]".to_string());
            let reward: f64 = row.get(6)?;
            let confidence: f64 = row.get(7)?;
            let success: bool = row.get::<_, i32>(8)? != 0;
            let reuse_count: i32 = row.get(9)?;
            let effectiveness: f64 = row.get(10)?;
            let created_at: String = row.get(11)?;
            let updated_at: String = row.get(12)?;
            let agent_id: Option<String> = row.get(13)?;
            let session_id: Option<String> = row.get(14)?;
            let content_hash: Option<String> = row.get(15)?;
            let critique: Option<String> = row.get(16)?;

            let tags: Vec<String> = serde_json::from_str(&tags_json).unwrap_or_default();

            Ok(SyncPatternData {
                id,
                problem,
                solution,
                context,
                domain,
                tags,
                reward: reward as f32,
                confidence: confidence as f32,
                success,
                reuse_count: reuse_count as u32,
                effectiveness: effectiveness as f32,
                created_at,
                updated_at,
                agent_id,
                session_id,
                content_hash,
                critique: critique.filter(|c| !c.is_empty()),
            })
        })
        .map_err(|e| NagualError::internal(format!("Query failed: {}", e)))?;

    let mut patterns = Vec::new();
    for row in rows {
        patterns.push(
            row.map_err(|e| NagualError::internal(format!("Row parse failed: {}", e)))?,
        );
    }

    Ok((patterns, total))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pull_params_defaults() {
        let params: PullParams = serde_json::from_str("{}").unwrap();
        assert!(params.since.is_none());
        assert_eq!(params.limit, 100);
        assert_eq!(params.offset, 0);
    }

    #[test]
    fn test_pull_params_with_since() {
        let json = r#"{"since":"2026-01-01T00:00:00+00:00","limit":50,"offset":10}"#;
        let params: PullParams = serde_json::from_str(json).unwrap();
        assert!(params.since.is_some());
        assert_eq!(params.limit, 50);
        assert_eq!(params.offset, 10);
    }
}
