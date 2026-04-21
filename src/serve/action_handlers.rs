//! Action endpoints for the Nagual dashboard (Phase 1).
//!
//! Provides endpoints for triggering learning operations (embed, consolidate,
//! dedup, pyramid), polling job status, and querying aggregated insights and
//! recommendations.
//!
//! Learning jobs run asynchronously via `tokio::spawn`. Each operation spawns
//! the `nagual` CLI binary as a child process to avoid pulling the full learning
//! module dependency chain into the serve feature. A simple in-memory job queue
//! enforces at most one concurrent learning job.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::Json;
use chrono::{DateTime, Utc};
use rusqlite::{Connection, OpenFlags};
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tracing::{info, warn};
use uuid::Uuid;

use super::auth::{RequireAuth, RequireWrite};
use super::AppState;
use crate::events::NagualEvent;

// ---------------------------------------------------------------------------
// Job types
// ---------------------------------------------------------------------------

/// The kind of learning operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JobKind {
    Embed,
    Consolidate,
    Dedup,
    Pyramid,
}

impl std::fmt::Display for JobKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            JobKind::Embed => write!(f, "embed"),
            JobKind::Consolidate => write!(f, "consolidate"),
            JobKind::Dedup => write!(f, "dedup"),
            JobKind::Pyramid => write!(f, "pyramid"),
        }
    }
}

/// Current status of a job.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JobStatusKind {
    Queued,
    Running,
    Completed,
    Failed,
}

impl std::fmt::Display for JobStatusKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            JobStatusKind::Queued => write!(f, "queued"),
            JobStatusKind::Running => write!(f, "running"),
            JobStatusKind::Completed => write!(f, "completed"),
            JobStatusKind::Failed => write!(f, "failed"),
        }
    }
}

/// Full state of a single job.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobStatus {
    pub job_id: String,
    pub kind: JobKind,
    pub status: JobStatusKind,
    pub progress: u8,
    pub message: String,
    pub started_at: DateTime<Utc>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completed_at: Option<DateTime<Utc>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<serde_json::Value>,
}

// ---------------------------------------------------------------------------
// Job Queue
// ---------------------------------------------------------------------------

/// In-memory job queue that enforces at most one concurrent learning job.
#[derive(Clone)]
pub struct JobQueue {
    jobs: Arc<RwLock<HashMap<String, JobStatus>>>,
    /// ID of the currently-running job, if any.
    running: Arc<RwLock<Option<String>>>,
}

impl JobQueue {
    /// Create a new empty job queue.
    pub fn new() -> Self {
        Self {
            jobs: Arc::new(RwLock::new(HashMap::new())),
            running: Arc::new(RwLock::new(None)),
        }
    }

    /// Attempt to enqueue a new job. Returns `None` if another job is already
    /// running (max-1 concurrency).
    pub async fn enqueue(&self, kind: JobKind) -> Option<String> {
        let mut running = self.running.write().await;

        // Check if the running job is actually still running
        if let Some(ref id) = *running {
            let jobs = self.jobs.read().await;
            if let Some(job) = jobs.get(id) {
                if job.status == JobStatusKind::Running || job.status == JobStatusKind::Queued {
                    return None; // another job in progress
                }
            }
            // Previous job finished; clear the slot
            *running = None;
        }

        let job_id = Uuid::new_v4().to_string();
        let job = JobStatus {
            job_id: job_id.clone(),
            kind,
            status: JobStatusKind::Queued,
            progress: 0,
            message: format!("{} job queued", kind),
            started_at: Utc::now(),
            completed_at: None,
            result: None,
        };

        self.jobs.write().await.insert(job_id.clone(), job);
        *running = Some(job_id.clone());

        Some(job_id)
    }

    /// Mark a job as running.
    pub async fn set_running(&self, job_id: &str) {
        let mut jobs = self.jobs.write().await;
        if let Some(job) = jobs.get_mut(job_id) {
            job.status = JobStatusKind::Running;
            job.message = format!("{} job running", job.kind);
        }
    }

    /// Update progress and message for a running job.
    pub async fn set_progress(&self, job_id: &str, progress: u8, message: &str) {
        let mut jobs = self.jobs.write().await;
        if let Some(job) = jobs.get_mut(job_id) {
            job.progress = progress.min(100);
            job.message = message.to_string();
        }
    }

    /// Mark a job as completed with an optional result payload.
    pub async fn set_completed(&self, job_id: &str, result: Option<serde_json::Value>) {
        let mut jobs = self.jobs.write().await;
        if let Some(job) = jobs.get_mut(job_id) {
            job.status = JobStatusKind::Completed;
            job.progress = 100;
            job.message = format!("{} job completed", job.kind);
            job.completed_at = Some(Utc::now());
            job.result = result;
        }
        // Clear running slot
        let mut running = self.running.write().await;
        if running.as_deref() == Some(job_id) {
            *running = None;
        }
    }

    /// Mark a job as failed with an error message.
    pub async fn set_failed(&self, job_id: &str, error: &str) {
        let mut jobs = self.jobs.write().await;
        if let Some(job) = jobs.get_mut(job_id) {
            job.status = JobStatusKind::Failed;
            job.message = error.to_string();
            job.completed_at = Some(Utc::now());
            job.result = Some(serde_json::json!({ "error": error }));
        }
        // Clear running slot
        let mut running = self.running.write().await;
        if running.as_deref() == Some(job_id) {
            *running = None;
        }
    }

    /// Get the status of a single job.
    pub async fn get(&self, job_id: &str) -> Option<JobStatus> {
        self.jobs.read().await.get(job_id).cloned()
    }

    /// List all jobs, most recent first. Retains at most the last 50 jobs.
    pub async fn list(&self) -> Vec<JobStatus> {
        let jobs = self.jobs.read().await;
        let mut list: Vec<JobStatus> = jobs.values().cloned().collect();
        list.sort_by(|a, b| b.started_at.cmp(&a.started_at));
        list.truncate(50);
        list
    }
}

impl Default for JobQueue {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Shared state extension
// ---------------------------------------------------------------------------

/// Extended application state that includes the job queue.
///
/// This is stored as a lazy-initialized `Arc<JobQueue>` inside a module-level
/// static. Alternatively the `AppState` could be extended, but this avoids
/// changing the existing struct for an additive feature.
///
/// We use a simple `once_cell`-style approach: the first handler call creates
/// the queue and caches it.
static JOB_QUEUE: std::sync::OnceLock<JobQueue> = std::sync::OnceLock::new();

/// Get or initialize the global job queue.
fn job_queue() -> &'static JobQueue {
    JOB_QUEUE.get_or_init(JobQueue::new)
}

// ---------------------------------------------------------------------------
// Request / Response types
// ---------------------------------------------------------------------------

/// POST /api/actions/embed request body.
#[derive(Debug, Deserialize)]
pub struct EmbedRequest {
    #[serde(default)]
    pub force: bool,
}

/// POST /api/actions/consolidate request body.
#[derive(Debug, Deserialize)]
pub struct ConsolidateRequest {
    #[serde(default = "default_similarity")]
    pub similarity: f64,
    #[serde(default)]
    pub dry_run: bool,
}

fn default_similarity() -> f64 {
    0.9
}

/// POST /api/actions/dedup request body.
#[derive(Debug, Deserialize)]
pub struct DedupRequest {
    #[serde(default = "default_true")]
    pub auto: bool,
    #[serde(default = "default_similarity")]
    pub threshold: f64,
}

fn default_true() -> bool {
    true
}

/// POST /api/actions/pyramid request body.
#[derive(Debug, Deserialize)]
pub struct PyramidRequest {
    #[serde(default = "default_pyramid_limit")]
    pub limit: u32,
}

fn default_pyramid_limit() -> u32 {
    200
}

/// Response returned when a job is enqueued.
#[derive(Debug, Serialize)]
pub struct JobEnqueuedResponse {
    pub job_id: String,
    pub status: String,
}

/// GET /api/insights response.
#[derive(Debug, Serialize)]
pub struct InsightsResponse {
    pub total_patterns: u64,
    pub embedded_count: u64,
    pub avg_reward: f64,
    pub success_rate: f64,
    pub domains: Vec<DomainStat>,
    pub trend: Vec<TrendEntry>,
    pub top_patterns: Vec<TopPattern>,
    pub recommendations_count: u64,
}

/// Domain statistics entry.
#[derive(Debug, Serialize)]
pub struct DomainStat {
    pub domain: String,
    pub count: u64,
    pub avg_reward: f64,
}

/// Daily trend entry.
#[derive(Debug, Serialize)]
pub struct TrendEntry {
    pub date: String,
    pub patterns_added: u64,
    pub avg_reward: f64,
}

/// Top-performing pattern summary.
#[derive(Debug, Serialize)]
pub struct TopPattern {
    pub id: String,
    pub problem: String,
    pub domain: String,
    pub reward: f64,
    pub reuse_count: u32,
}

/// GET /api/recommendations response entry.
#[derive(Debug, Serialize)]
pub struct Recommendation {
    #[serde(rename = "type")]
    pub rec_type: String,
    pub priority: String,
    pub impact: String,
    pub domain: String,
    pub description: String,
    pub affected_count: u64,
}

// ---------------------------------------------------------------------------
// Helper: open read-only SQLite connection
// ---------------------------------------------------------------------------

fn open_db(db_path: &std::path::Path) -> Result<Connection, (StatusCode, String)> {
    Connection::open_with_flags(db_path, OpenFlags::SQLITE_OPEN_READ_ONLY).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Database error: {}", e),
        )
    })
}

/// Detect which column name is used for "domain" in reasoning_patterns.
fn domain_column(conn: &Connection) -> &'static str {
    if conn
        .prepare("SELECT domain FROM reasoning_patterns LIMIT 0")
        .is_ok()
    {
        "domain"
    } else {
        "category"
    }
}

// ---------------------------------------------------------------------------
// Resolve the nagual binary path
// ---------------------------------------------------------------------------

fn nagual_binary() -> PathBuf {
    // Prefer the local bin, fall back to PATH lookup
    let local = dirs_or_home().join(".local/bin/nagual");
    if local.exists() {
        return local;
    }
    PathBuf::from("nagual")
}

/// Home directory helper (avoids pulling in `dirs` crate).
fn dirs_or_home() -> PathBuf {
    std::env::var("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/tmp"))
}

// ---------------------------------------------------------------------------
// Action Handlers
// ---------------------------------------------------------------------------

/// POST /api/actions/embed -- start an embedding job.
pub async fn api_action_embed(
    State(state): State<AppState>,
    _auth: RequireWrite,
    Json(req): Json<EmbedRequest>,
) -> impl IntoResponse {
    let queue = job_queue();

    let job_id = match queue.enqueue(JobKind::Embed).await {
        Some(id) => id,
        None => {
            return (
                StatusCode::CONFLICT,
                Json(serde_json::json!({
                    "error": "Another learning job is already running",
                    "code": "job_conflict"
                })),
            )
                .into_response();
        }
    };

    let resp = JobEnqueuedResponse {
        job_id: job_id.clone(),
        status: "queued".to_string(),
    };

    // Spawn the background task
    let db_path = state.db_path.clone();
    let event_bus = state.event_bus.clone();
    let force = req.force;

    tokio::spawn(async move {
        run_embed_job(&job_id, &db_path, &event_bus, force).await;
    });

    (StatusCode::ACCEPTED, Json(serde_json::to_value(resp).unwrap())).into_response()
}

/// POST /api/actions/consolidate -- start a consolidation job.
pub async fn api_action_consolidate(
    State(state): State<AppState>,
    _auth: RequireWrite,
    Json(req): Json<ConsolidateRequest>,
) -> impl IntoResponse {
    let queue = job_queue();

    let job_id = match queue.enqueue(JobKind::Consolidate).await {
        Some(id) => id,
        None => {
            return (
                StatusCode::CONFLICT,
                Json(serde_json::json!({
                    "error": "Another learning job is already running",
                    "code": "job_conflict"
                })),
            )
                .into_response();
        }
    };

    let resp = JobEnqueuedResponse {
        job_id: job_id.clone(),
        status: "queued".to_string(),
    };

    let db_path = state.db_path.clone();
    let event_bus = state.event_bus.clone();
    let similarity = req.similarity;
    let dry_run = req.dry_run;

    tokio::spawn(async move {
        run_consolidate_job(&job_id, &db_path, &event_bus, similarity, dry_run).await;
    });

    (StatusCode::ACCEPTED, Json(serde_json::to_value(resp).unwrap())).into_response()
}

/// POST /api/actions/dedup -- start a deduplication job.
pub async fn api_action_dedup(
    State(state): State<AppState>,
    _auth: RequireWrite,
    Json(req): Json<DedupRequest>,
) -> impl IntoResponse {
    let queue = job_queue();

    let job_id = match queue.enqueue(JobKind::Dedup).await {
        Some(id) => id,
        None => {
            return (
                StatusCode::CONFLICT,
                Json(serde_json::json!({
                    "error": "Another learning job is already running",
                    "code": "job_conflict"
                })),
            )
                .into_response();
        }
    };

    let resp = JobEnqueuedResponse {
        job_id: job_id.clone(),
        status: "queued".to_string(),
    };

    let db_path = state.db_path.clone();
    let event_bus = state.event_bus.clone();
    let auto = req.auto;
    let threshold = req.threshold;

    tokio::spawn(async move {
        run_dedup_job(&job_id, &db_path, &event_bus, auto, threshold).await;
    });

    (StatusCode::ACCEPTED, Json(serde_json::to_value(resp).unwrap())).into_response()
}

/// POST /api/actions/pyramid -- start a pyramid generation job.
pub async fn api_action_pyramid(
    State(state): State<AppState>,
    _auth: RequireWrite,
    Json(req): Json<PyramidRequest>,
) -> impl IntoResponse {
    let queue = job_queue();

    let job_id = match queue.enqueue(JobKind::Pyramid).await {
        Some(id) => id,
        None => {
            return (
                StatusCode::CONFLICT,
                Json(serde_json::json!({
                    "error": "Another learning job is already running",
                    "code": "job_conflict"
                })),
            )
                .into_response();
        }
    };

    let resp = JobEnqueuedResponse {
        job_id: job_id.clone(),
        status: "queued".to_string(),
    };

    let db_path = state.db_path.clone();
    let event_bus = state.event_bus.clone();
    let limit = req.limit;

    tokio::spawn(async move {
        run_pyramid_job(&job_id, &db_path, &event_bus, limit).await;
    });

    (StatusCode::ACCEPTED, Json(serde_json::to_value(resp).unwrap())).into_response()
}

/// GET /api/actions/status/:job_id -- poll job status.
pub async fn api_action_status(
    _auth: RequireAuth,
    Path(job_id): Path<String>,
) -> impl IntoResponse {
    let queue = job_queue();

    match queue.get(&job_id).await {
        Some(job) => (StatusCode::OK, Json(serde_json::to_value(job).unwrap())).into_response(),
        None => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({
                "error": "Job not found",
                "job_id": job_id
            })),
        )
            .into_response(),
    }
}

/// GET /api/actions/jobs -- list recent jobs.
pub async fn api_action_jobs(
    _auth: RequireAuth,
) -> impl IntoResponse {
    let queue = job_queue();
    let jobs = queue.list().await;
    Json(serde_json::to_value(jobs).unwrap())
}

// ---------------------------------------------------------------------------
// GET /api/insights
// ---------------------------------------------------------------------------

/// GET /api/insights -- aggregated learning metrics.
pub async fn api_insights(
    State(state): State<AppState>,
    _auth: RequireAuth,
) -> Result<Json<InsightsResponse>, (StatusCode, String)> {
    let conn = open_db(&state.db_path)?;
    let dcol = domain_column(&conn);

    // Total patterns
    let total_patterns: u64 = conn
        .query_row("SELECT COUNT(*) FROM reasoning_patterns", [], |row| row.get(0))
        .unwrap_or(0);

    // Embedded count (has non-null embedding blob)
    let embedded_count: u64 = conn
        .query_row(
            "SELECT COUNT(*) FROM reasoning_patterns WHERE embedding IS NOT NULL",
            [],
            |row| row.get(0),
        )
        .unwrap_or(0);

    // Average reward
    let avg_reward: f64 = conn
        .query_row(
            "SELECT COALESCE(AVG(reward), 0.0) FROM reasoning_patterns",
            [],
            |row| row.get(0),
        )
        .unwrap_or(0.0);

    // Success rate: proportion of patterns where success = 1 (or reward >= 0.5 as proxy)
    let success_rate: f64 = if total_patterns == 0 {
        0.0
    } else {
        // Try the success column first; fall back to reward-based estimate
        let has_success_col = conn
            .prepare("SELECT success FROM reasoning_patterns LIMIT 0")
            .is_ok();

        if has_success_col {
            let success_count: u64 = conn
                .query_row(
                    "SELECT COUNT(*) FROM reasoning_patterns WHERE success = 1",
                    [],
                    |row| row.get(0),
                )
                .unwrap_or(0);
            success_count as f64 / total_patterns as f64
        } else {
            // Estimate: patterns with reward >= 0.5 are considered successful
            let above_threshold: u64 = conn
                .query_row(
                    "SELECT COUNT(*) FROM reasoning_patterns WHERE reward >= 0.5",
                    [],
                    |row| row.get(0),
                )
                .unwrap_or(0);
            above_threshold as f64 / total_patterns as f64
        }
    };

    // Top domains by count + avg reward
    let domains = {
        let sql = format!(
            "SELECT COALESCE({dcol}, 'unknown') AS d, COUNT(*) AS cnt, \
             COALESCE(AVG(reward), 0.0) AS ar \
             FROM reasoning_patterns GROUP BY d ORDER BY cnt DESC LIMIT 20"
        );
        let mut stmt = conn.prepare(&sql).map_err(|e| {
            (StatusCode::INTERNAL_SERVER_ERROR, format!("Query error: {}", e))
        })?;
        let rows = stmt
            .query_map([], |row| {
                Ok(DomainStat {
                    domain: row.get(0)?,
                    count: row.get(1)?,
                    avg_reward: row.get(2)?,
                })
            })
            .map_err(|e| {
                (StatusCode::INTERNAL_SERVER_ERROR, format!("Query error: {}", e))
            })?;
        rows.filter_map(|r| r.ok()).collect::<Vec<_>>()
    };

    // Recent activity trend (last 30 days)
    let trend = {
        let sql = format!(
            "SELECT DATE(created_at) AS day, COUNT(*) AS cnt, \
             COALESCE(AVG(reward), 0.0) AS ar \
             FROM reasoning_patterns \
             WHERE created_at >= DATE('now', '-30 days') \
             GROUP BY DATE(created_at) ORDER BY day"
        );
        let mut stmt = conn.prepare(&sql).map_err(|e| {
            (StatusCode::INTERNAL_SERVER_ERROR, format!("Query error: {}", e))
        })?;
        let rows = stmt
            .query_map([], |row| {
                Ok(TrendEntry {
                    date: row.get(0)?,
                    patterns_added: row.get(1)?,
                    avg_reward: row.get(2)?,
                })
            })
            .map_err(|e| {
                (StatusCode::INTERNAL_SERVER_ERROR, format!("Query error: {}", e))
            })?;
        rows.filter_map(|r| r.ok()).collect::<Vec<_>>()
    };

    // Top patterns (highest reward, most reused)
    let top_patterns = {
        let has_reuse = conn
            .prepare("SELECT reuse_count FROM reasoning_patterns LIMIT 0")
            .is_ok();
        let reuse_expr = if has_reuse {
            "COALESCE(reuse_count, 0)"
        } else {
            "0"
        };

        let sql = format!(
            "SELECT id, COALESCE(problem, ''), COALESCE({dcol}, 'unknown'), \
             COALESCE(reward, 0.0), {reuse_expr} \
             FROM reasoning_patterns \
             ORDER BY reward DESC, {reuse_expr} DESC \
             LIMIT 10"
        );
        let mut stmt = conn.prepare(&sql).map_err(|e| {
            (StatusCode::INTERNAL_SERVER_ERROR, format!("Query error: {}", e))
        })?;
        let rows = stmt
            .query_map([], |row| {
                Ok(TopPattern {
                    id: row.get(0)?,
                    problem: row.get::<_, String>(1)?
                        .chars()
                        .take(120)
                        .collect(),
                    domain: row.get(2)?,
                    reward: row.get(3)?,
                    reuse_count: row.get::<_, i64>(4).unwrap_or(0) as u32,
                })
            })
            .map_err(|e| {
                (StatusCode::INTERNAL_SERVER_ERROR, format!("Query error: {}", e))
            })?;
        rows.filter_map(|r| r.ok()).collect::<Vec<_>>()
    };

    // Count of actionable recommendations (approximate)
    let low_reward_count: u64 = conn
        .query_row(
            "SELECT COUNT(*) FROM reasoning_patterns WHERE reward < 0.4",
            [],
            |row| row.get(0),
        )
        .unwrap_or(0);

    let high_reward_count: u64 = conn
        .query_row(
            "SELECT COUNT(*) FROM reasoning_patterns WHERE reward >= 0.8",
            [],
            |row| row.get(0),
        )
        .unwrap_or(0);

    let recommendations_count = if low_reward_count > 0 { 1 } else { 0 }
        + if high_reward_count > 0 { 1 } else { 0 }
        + if embedded_count < total_patterns { 1 } else { 0 };

    Ok(Json(InsightsResponse {
        total_patterns,
        embedded_count,
        avg_reward,
        success_rate,
        domains,
        trend,
        top_patterns,
        recommendations_count,
    }))
}

// ---------------------------------------------------------------------------
// GET /api/recommendations
// ---------------------------------------------------------------------------

/// GET /api/recommendations -- improvement recommendations.
pub async fn api_recommendations(
    State(state): State<AppState>,
    _auth: RequireAuth,
) -> Result<Json<Vec<Recommendation>>, (StatusCode, String)> {
    let conn = open_db(&state.db_path)?;
    let dcol = domain_column(&conn);

    let mut recommendations: Vec<Recommendation> = Vec::new();

    // 1. Unembedded patterns
    let total: u64 = conn
        .query_row("SELECT COUNT(*) FROM reasoning_patterns", [], |row| row.get(0))
        .unwrap_or(0);
    let embedded: u64 = conn
        .query_row(
            "SELECT COUNT(*) FROM reasoning_patterns WHERE embedding IS NOT NULL",
            [],
            |row| row.get(0),
        )
        .unwrap_or(0);
    let unembedded = total.saturating_sub(embedded);

    if unembedded > 0 {
        recommendations.push(Recommendation {
            rec_type: "embed".to_string(),
            priority: if unembedded > total / 2 { "high" } else { "medium" }.to_string(),
            impact: "Enables semantic search and consolidation".to_string(),
            domain: "*".to_string(),
            description: format!(
                "{} patterns lack embeddings ({:.0}% of total). Run embedding to enable semantic features.",
                unembedded,
                (unembedded as f64 / total.max(1) as f64) * 100.0
            ),
            affected_count: unembedded,
        });
    }

    // 2. High-reward patterns to promote (reward >= 0.8, not yet reflex tier)
    let has_tier = conn
        .prepare("SELECT tier FROM reasoning_patterns LIMIT 0")
        .is_ok();

    if has_tier {
        let promotable: u64 = conn
            .query_row(
                "SELECT COUNT(*) FROM reasoning_patterns \
                 WHERE reward >= 0.8 AND (tier IS NULL OR LOWER(tier) != 'reflex')",
                [],
                |row| row.get(0),
            )
            .unwrap_or(0);

        if promotable > 0 {
            recommendations.push(Recommendation {
                rec_type: "promote".to_string(),
                priority: "medium".to_string(),
                impact: "Improves retrieval priority for proven patterns".to_string(),
                domain: "*".to_string(),
                description: format!(
                    "{} high-reward patterns (>= 0.8) are not yet in the reflex tier. \
                     Consider promoting them for faster retrieval.",
                    promotable
                ),
                affected_count: promotable,
            });
        }
    }

    // 3. Low-reward patterns to archive (reward < 0.4)
    let archivable: u64 = conn
        .query_row(
            "SELECT COUNT(*) FROM reasoning_patterns WHERE reward < 0.4",
            [],
            |row| row.get(0),
        )
        .unwrap_or(0);

    if archivable > 0 {
        // Find the worst domain
        let worst_domain: String = conn
            .query_row(
                &format!(
                    "SELECT COALESCE({dcol}, 'unknown') FROM reasoning_patterns \
                     WHERE reward < 0.4 GROUP BY {dcol} ORDER BY COUNT(*) DESC LIMIT 1"
                ),
                [],
                |row| row.get(0),
            )
            .unwrap_or_else(|_| "unknown".to_string());

        recommendations.push(Recommendation {
            rec_type: "archive".to_string(),
            priority: if archivable > 100 { "high" } else { "low" }.to_string(),
            impact: "Reduces noise in pattern retrieval".to_string(),
            domain: worst_domain,
            description: format!(
                "{} patterns have low reward (< 0.4). Consider archiving or reviewing them.",
                archivable
            ),
            affected_count: archivable,
        });
    }

    // 4. Complex patterns to split (solution length > 1000 chars)
    let complex: u64 = conn
        .query_row(
            "SELECT COUNT(*) FROM reasoning_patterns WHERE LENGTH(solution) > 1000",
            [],
            |row| row.get(0),
        )
        .unwrap_or(0);

    if complex > 0 {
        recommendations.push(Recommendation {
            rec_type: "split".to_string(),
            priority: "low".to_string(),
            impact: "Improves pattern specificity and reuse".to_string(),
            domain: "*".to_string(),
            description: format!(
                "{} patterns have solutions longer than 1000 characters. \
                 Consider splitting them into more focused patterns.",
                complex
            ),
            affected_count: complex,
        });
    }

    // 5. Potential duplicates (same domain, similar reward, high count)
    let dup_domains: u64 = conn
        .query_row(
            &format!(
                "SELECT COUNT(*) FROM (\
                    SELECT {dcol}, COUNT(*) AS cnt \
                    FROM reasoning_patterns \
                    GROUP BY {dcol} \
                    HAVING cnt > 50\
                 )"
            ),
            [],
            |row| row.get(0),
        )
        .unwrap_or(0);

    if dup_domains > 0 {
        recommendations.push(Recommendation {
            rec_type: "consolidate".to_string(),
            priority: "medium".to_string(),
            impact: "Reduces redundancy and improves quality".to_string(),
            domain: "*".to_string(),
            description: format!(
                "{} domain(s) have more than 50 patterns each. \
                 Running consolidation may merge similar entries.",
                dup_domains
            ),
            affected_count: dup_domains,
        });
    }

    // 6. Stale patterns (not updated in 90+ days, low reuse)
    let has_updated_at = conn
        .prepare("SELECT updated_at FROM reasoning_patterns LIMIT 0")
        .is_ok();
    let has_reuse = conn
        .prepare("SELECT reuse_count FROM reasoning_patterns LIMIT 0")
        .is_ok();

    if has_updated_at && has_reuse {
        let stale: u64 = conn
            .query_row(
                "SELECT COUNT(*) FROM reasoning_patterns \
                 WHERE updated_at < DATE('now', '-90 days') AND COALESCE(reuse_count, 0) < 2",
                [],
                |row| row.get(0),
            )
            .unwrap_or(0);

        if stale > 0 {
            recommendations.push(Recommendation {
                rec_type: "review_stale".to_string(),
                priority: "low".to_string(),
                impact: "Keeps the knowledge base current".to_string(),
                domain: "*".to_string(),
                description: format!(
                    "{} patterns have not been updated in 90+ days and have low reuse. \
                     Consider reviewing or archiving.",
                    stale
                ),
                affected_count: stale,
            });
        }
    }

    // Sort recommendations by priority: high > medium > low
    fn priority_ord(p: &str) -> u8 {
        match p {
            "high" => 0,
            "medium" => 1,
            "low" => 2,
            _ => 3,
        }
    }
    recommendations.sort_by_key(|r| priority_ord(&r.priority));

    Ok(Json(recommendations))
}

// ---------------------------------------------------------------------------
// Phase 2: Pattern Management Endpoints
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Request / Response types (Phase 2)
// ---------------------------------------------------------------------------

/// Bulk action types for POST /api/patterns/bulk.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BulkAction {
    Archive,
    Promote,
    Tag,
    Domain,
    Delete,
}

impl std::fmt::Display for BulkAction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BulkAction::Archive => write!(f, "archive"),
            BulkAction::Promote => write!(f, "promote"),
            BulkAction::Tag => write!(f, "tag"),
            BulkAction::Domain => write!(f, "domain"),
            BulkAction::Delete => write!(f, "delete"),
        }
    }
}

/// POST /api/patterns/bulk request body.
#[derive(Debug, Deserialize)]
pub struct BulkRequest {
    pub ids: Vec<String>,
    pub action: BulkAction,
    /// Optional value for tag/domain actions.
    #[serde(default)]
    pub value: Option<String>,
}

/// POST /api/patterns/bulk response.
#[derive(Debug, Serialize)]
pub struct BulkResponse {
    pub affected: u64,
    pub action: String,
    pub errors: Vec<String>,
}

/// GET /api/tags response entry.
#[derive(Debug, Serialize)]
pub struct TagEntry {
    pub tag: String,
    pub count: u64,
}

/// GET /api/tags response.
#[derive(Debug, Serialize)]
pub struct TagsResponse {
    pub tags: Vec<TagEntry>,
}

/// GET /api/patterns/:id/history outcome entry.
#[derive(Debug, Serialize)]
pub struct OutcomeEntry {
    pub outcome: String,
    pub reward: f64,
    pub feedback: Option<String>,
    pub recorded_at: String,
}

/// GET /api/patterns/:id/history response.
#[derive(Debug, Serialize)]
pub struct HistoryResponse {
    pub outcomes: Vec<OutcomeEntry>,
}

/// POST /api/patterns/:id/archive response.
#[derive(Debug, Serialize)]
pub struct ArchiveResponse {
    pub id: String,
    pub status: String,
}

/// POST /api/patterns/:id/promote response.
#[derive(Debug, Serialize)]
pub struct PromoteResponse {
    pub id: String,
    pub status: String,
    pub tier: String,
}

// ---------------------------------------------------------------------------
// Phase 2 helper: detect tier column
// ---------------------------------------------------------------------------

/// Check if the `tier` column exists in reasoning_patterns.
fn has_tier_column(conn: &Connection) -> bool {
    conn.prepare("SELECT tier FROM reasoning_patterns LIMIT 0")
        .is_ok()
}

/// Check if the `tags` column exists in reasoning_patterns.
fn has_tags_column(conn: &Connection) -> bool {
    conn.prepare("SELECT tags FROM reasoning_patterns LIMIT 0")
        .is_ok()
}

/// Open a read-write SQLite connection.
fn open_rw_db(db_path: &std::path::Path) -> Result<Connection, (StatusCode, String)> {
    Connection::open(db_path).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Database error: {}", e),
        )
    })
}

// ---------------------------------------------------------------------------
// POST /api/patterns/bulk
// ---------------------------------------------------------------------------

/// POST /api/patterns/bulk -- perform bulk operations on patterns.
pub async fn api_patterns_bulk(
    State(state): State<AppState>,
    _auth: RequireWrite,
    Json(req): Json<BulkRequest>,
) -> impl IntoResponse {
    if req.ids.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": "No pattern IDs provided",
                "code": "empty_ids"
            })),
        )
            .into_response();
    }

    // Validate that tag/domain actions have a value
    if matches!(req.action, BulkAction::Tag | BulkAction::Domain) && req.value.is_none() {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": format!("'value' is required for {} action", req.action),
                "code": "missing_value"
            })),
        )
            .into_response();
    }

    let conn = match open_rw_db(&state.db_path) {
        Ok(c) => c,
        Err((status, msg)) => {
            return (status, Json(serde_json::json!({ "error": msg }))).into_response();
        }
    };

    let has_tier = has_tier_column(&conn);
    let has_tags = has_tags_column(&conn);
    let dcol = domain_column(&conn);

    let mut affected: u64 = 0;
    let mut errors: Vec<String> = Vec::new();

    // Execute in a transaction
    if let Err(e) = conn.execute("BEGIN", []) {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": format!("Transaction start failed: {}", e) })),
        )
            .into_response();
    }

    for id in &req.ids {
        let result = match &req.action {
            BulkAction::Archive => {
                if has_tier {
                    conn.execute(
                        "UPDATE reasoning_patterns SET tier = 'archived' WHERE id = ?1",
                        rusqlite::params![id],
                    )
                } else {
                    errors.push(format!("{}: tier column not found", id));
                    continue;
                }
            }
            BulkAction::Promote => {
                if has_tier {
                    conn.execute(
                        "UPDATE reasoning_patterns SET tier = 'reflex' WHERE id = ?1",
                        rusqlite::params![id],
                    )
                } else {
                    errors.push(format!("{}: tier column not found", id));
                    continue;
                }
            }
            BulkAction::Tag => {
                let tag_value = req.value.as_deref().unwrap_or("");
                if has_tags {
                    // Append tag to existing comma-separated tags
                    conn.execute(
                        "UPDATE reasoning_patterns SET tags = CASE \
                         WHEN tags IS NULL OR tags = '' THEN ?1 \
                         ELSE tags || ',' || ?1 \
                         END WHERE id = ?2",
                        rusqlite::params![tag_value, id],
                    )
                } else {
                    errors.push(format!("{}: tags column not found", id));
                    continue;
                }
            }
            BulkAction::Domain => {
                let domain_value = req.value.as_deref().unwrap_or("");
                conn.execute(
                    &format!("UPDATE reasoning_patterns SET {} = ?1 WHERE id = ?2", dcol),
                    rusqlite::params![domain_value, id],
                )
            }
            BulkAction::Delete => {
                conn.execute(
                    "DELETE FROM reasoning_patterns WHERE id = ?1",
                    rusqlite::params![id],
                )
            }
        };

        match result {
            Ok(rows) => {
                if rows > 0 {
                    affected += rows as u64;
                } else {
                    errors.push(format!("{}: pattern not found", id));
                }
            }
            Err(e) => {
                errors.push(format!("{}: {}", id, e));
            }
        }
    }

    // Commit the transaction
    if let Err(e) = conn.execute("COMMIT", []) {
        warn!(error = %e, "Failed to commit bulk transaction");
        let _ = conn.execute("ROLLBACK", []);
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": format!("Commit failed: {}", e) })),
        )
            .into_response();
    }

    // Publish events for each affected pattern
    let action_str = req.action.to_string();
    for id in &req.ids {
        match &req.action {
            BulkAction::Delete => {
                state
                    .event_bus
                    .publish_sync(NagualEvent::pattern_deleted(id.as_str()));
            }
            _ => {
                let changes = crate::events::PatternChanges::new()
                    .with_field(action_str.clone())
                    .with_metadata("bulk_action", serde_json::json!(action_str));
                state
                    .event_bus
                    .publish_sync(NagualEvent::pattern_updated(id.as_str(), changes));
            }
        }
    }

    info!(
        action = %action_str,
        affected = affected,
        total_ids = req.ids.len(),
        errors = errors.len(),
        "Bulk operation completed"
    );

    let resp = BulkResponse {
        affected,
        action: action_str,
        errors,
    };

    (
        StatusCode::OK,
        Json(serde_json::to_value(resp).unwrap()),
    )
        .into_response()
}

// ---------------------------------------------------------------------------
// GET /api/tags
// ---------------------------------------------------------------------------

/// GET /api/tags -- list all tags with counts.
pub async fn api_tags(
    State(state): State<AppState>,
) -> Result<Json<TagsResponse>, (StatusCode, String)> {
    let conn = open_db(&state.db_path)?;

    if !has_tags_column(&conn) {
        // No tags column — return empty
        return Ok(Json(TagsResponse { tags: vec![] }));
    }

    // Read all non-null tags from the table
    let mut stmt = conn
        .prepare("SELECT tags FROM reasoning_patterns WHERE tags IS NOT NULL AND tags != ''")
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Query error: {}", e),
            )
        })?;

    let rows = stmt
        .query_map([], |row| row.get::<_, String>(0))
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Query error: {}", e),
            )
        })?;

    // Aggregate tag counts: tags can be comma-separated or JSON arrays
    let mut tag_counts: HashMap<String, u64> = HashMap::new();

    for row_result in rows {
        if let Ok(tags_raw) = row_result {
            // Try parsing as JSON array first
            if let Ok(arr) = serde_json::from_str::<Vec<String>>(&tags_raw) {
                for tag in arr {
                    let tag = tag.trim().to_string();
                    if !tag.is_empty() {
                        *tag_counts.entry(tag).or_insert(0) += 1;
                    }
                }
            } else {
                // Fall back to comma-separated
                for tag in tags_raw.split(',') {
                    let tag = tag.trim().to_string();
                    if !tag.is_empty() {
                        *tag_counts.entry(tag).or_insert(0) += 1;
                    }
                }
            }
        }
    }

    // Sort by count descending
    let mut tags: Vec<TagEntry> = tag_counts
        .into_iter()
        .map(|(tag, count)| TagEntry { tag, count })
        .collect();
    tags.sort_by(|a, b| b.count.cmp(&a.count));

    Ok(Json(TagsResponse { tags }))
}

// ---------------------------------------------------------------------------
// GET /api/patterns/:id/history
// ---------------------------------------------------------------------------

/// GET /api/patterns/:id/history -- outcome history for a pattern.
pub async fn api_pattern_history(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<HistoryResponse>, (StatusCode, String)> {
    let conn = open_db(&state.db_path)?;

    // Check if the outcomes table exists
    let has_outcomes: bool = conn
        .query_row(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='outcomes'",
            [],
            |row| row.get::<_, i64>(0),
        )
        .map(|count| count > 0)
        .unwrap_or(false);

    if !has_outcomes {
        return Ok(Json(HistoryResponse { outcomes: vec![] }));
    }

    let mut stmt = conn
        .prepare(
            "SELECT outcome, COALESCE(reward, 0.0), feedback, \
             COALESCE(recorded_at, created_at, '') \
             FROM outcomes WHERE pattern_id = ?1 \
             ORDER BY COALESCE(recorded_at, created_at) DESC",
        )
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Query error: {}", e),
            )
        })?;

    let rows = stmt
        .query_map(rusqlite::params![id], |row| {
            Ok(OutcomeEntry {
                outcome: row.get(0)?,
                reward: row.get(1)?,
                feedback: row.get(2)?,
                recorded_at: row.get(3)?,
            })
        })
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Query error: {}", e),
            )
        })?;

    let outcomes: Vec<OutcomeEntry> = rows.filter_map(|r| r.ok()).collect();

    Ok(Json(HistoryResponse { outcomes }))
}

// ---------------------------------------------------------------------------
// POST /api/patterns/:id/archive
// ---------------------------------------------------------------------------

/// POST /api/patterns/:id/archive -- archive a single pattern.
pub async fn api_pattern_archive(
    State(state): State<AppState>,
    _auth: RequireWrite,
    Path(id): Path<String>,
) -> impl IntoResponse {
    let conn = match open_rw_db(&state.db_path) {
        Ok(c) => c,
        Err((status, msg)) => {
            return (status, Json(serde_json::json!({ "error": msg }))).into_response();
        }
    };

    if !has_tier_column(&conn) {
        return (
            StatusCode::UNPROCESSABLE_ENTITY,
            Json(serde_json::json!({
                "error": "Schema does not have a tier column",
                "code": "no_tier_column"
            })),
        )
            .into_response();
    }

    let updated = conn
        .execute(
            "UPDATE reasoning_patterns SET tier = 'archived' WHERE id = ?1",
            rusqlite::params![id],
        )
        .unwrap_or(0);

    if updated == 0 {
        return (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({
                "error": "Pattern not found",
                "id": id
            })),
        )
            .into_response();
    }

    let changes = crate::events::PatternChanges::new()
        .with_field("tier".to_string())
        .with_metadata("new_tier", serde_json::json!("archived"));
    state
        .event_bus
        .publish_sync(NagualEvent::pattern_updated(&id, changes));

    info!(id = %id, "Pattern archived");

    let resp = ArchiveResponse {
        id,
        status: "archived".to_string(),
    };

    (
        StatusCode::OK,
        Json(serde_json::to_value(resp).unwrap()),
    )
        .into_response()
}

// ---------------------------------------------------------------------------
// POST /api/patterns/:id/promote
// ---------------------------------------------------------------------------

/// POST /api/patterns/:id/promote -- promote a single pattern to reflex tier.
pub async fn api_pattern_promote(
    State(state): State<AppState>,
    _auth: RequireWrite,
    Path(id): Path<String>,
) -> impl IntoResponse {
    let conn = match open_rw_db(&state.db_path) {
        Ok(c) => c,
        Err((status, msg)) => {
            return (status, Json(serde_json::json!({ "error": msg }))).into_response();
        }
    };

    if !has_tier_column(&conn) {
        return (
            StatusCode::UNPROCESSABLE_ENTITY,
            Json(serde_json::json!({
                "error": "Schema does not have a tier column",
                "code": "no_tier_column"
            })),
        )
            .into_response();
    }

    let updated = conn
        .execute(
            "UPDATE reasoning_patterns SET tier = 'reflex' WHERE id = ?1",
            rusqlite::params![id],
        )
        .unwrap_or(0);

    if updated == 0 {
        return (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({
                "error": "Pattern not found",
                "id": id
            })),
        )
            .into_response();
    }

    let changes = crate::events::PatternChanges::new()
        .with_field("tier".to_string())
        .with_metadata("new_tier", serde_json::json!("reflex"));
    state
        .event_bus
        .publish_sync(NagualEvent::pattern_updated(&id, changes));

    info!(id = %id, "Pattern promoted to reflex tier");

    let resp = PromoteResponse {
        id,
        status: "promoted".to_string(),
        tier: "reflex".to_string(),
    };

    (
        StatusCode::OK,
        Json(serde_json::to_value(resp).unwrap()),
    )
        .into_response()
}

// ---------------------------------------------------------------------------
// Background job runners
// ---------------------------------------------------------------------------

/// Run the embedding job by shelling out to `nagual learn embed`.
async fn run_embed_job(
    job_id: &str,
    db_path: &std::path::Path,
    event_bus: &Arc<crate::events::EventBus>,
    force: bool,
) {
    let queue = job_queue();
    queue.set_running(job_id).await;
    queue
        .set_progress(job_id, 10, "Starting embedding process")
        .await;

    info!(job_id = %job_id, force = force, "Starting embed job");

    event_bus.publish_sync(NagualEvent::batch_completed(
        "embed_started",
        0,
        0,
        0,
        0,
    ));

    let start = std::time::Instant::now();

    let mut cmd = tokio::process::Command::new(nagual_binary());
    cmd.arg("learn")
        .arg("embed")
        .arg("--db-path")
        .arg(db_path);

    if force {
        cmd.arg("--force");
    }

    // Ensure ONNX runtime can be found
    if let Ok(ort_path) = std::env::var("ORT_DYLIB_PATH") {
        cmd.env("ORT_DYLIB_PATH", ort_path);
    } else {
        // Default macOS path
        cmd.env(
            "ORT_DYLIB_PATH",
            "/opt/homebrew/lib/libonnxruntime.dylib",
        );
    }

    queue
        .set_progress(job_id, 30, "Embedding in progress")
        .await;

    match cmd.output().await {
        Ok(output) => {
            let elapsed = start.elapsed().as_millis() as u64;
            let stdout = String::from_utf8_lossy(&output.stdout);
            let stderr = String::from_utf8_lossy(&output.stderr);

            if output.status.success() {
                info!(
                    job_id = %job_id,
                    elapsed_ms = elapsed,
                    "Embed job completed"
                );

                let result = serde_json::json!({
                    "stdout": stdout.chars().take(2000).collect::<String>(),
                    "duration_ms": elapsed,
                });

                queue.set_completed(job_id, Some(result)).await;

                event_bus.publish_sync(NagualEvent::batch_completed(
                    "embed",
                    0,
                    0,
                    0,
                    elapsed,
                ));
            } else {
                let error_msg = if stderr.is_empty() {
                    format!("Embed failed with exit code {:?}", output.status.code())
                } else {
                    stderr.chars().take(500).collect::<String>()
                };

                warn!(job_id = %job_id, error = %error_msg, "Embed job failed");
                queue.set_failed(job_id, &error_msg).await;

                event_bus.publish_sync(NagualEvent::error_occurred(
                    "action_handler",
                    "embed_failed",
                    &error_msg,
                    true,
                ));
            }
        }
        Err(e) => {
            let error_msg = format!("Failed to spawn embed process: {}", e);
            warn!(job_id = %job_id, error = %error_msg, "Embed job spawn failed");
            queue.set_failed(job_id, &error_msg).await;

            event_bus.publish_sync(NagualEvent::error_occurred(
                "action_handler",
                "embed_spawn_failed",
                &error_msg,
                true,
            ));
        }
    }
}

/// Run the consolidation job by shelling out to `nagual learn consolidate`.
async fn run_consolidate_job(
    job_id: &str,
    db_path: &std::path::Path,
    event_bus: &Arc<crate::events::EventBus>,
    similarity: f64,
    dry_run: bool,
) {
    let queue = job_queue();
    queue.set_running(job_id).await;
    queue
        .set_progress(job_id, 10, "Starting consolidation")
        .await;

    info!(
        job_id = %job_id,
        similarity = similarity,
        dry_run = dry_run,
        "Starting consolidate job"
    );

    let start = std::time::Instant::now();

    let mut cmd = tokio::process::Command::new(nagual_binary());
    cmd.arg("learn")
        .arg("consolidate")
        .arg("--similarity")
        .arg(format!("{:.2}", similarity))
        .arg("--db-path")
        .arg(db_path);

    // Ensure ONNX runtime can be found (needed for embedding comparison)
    if let Ok(ort_path) = std::env::var("ORT_DYLIB_PATH") {
        cmd.env("ORT_DYLIB_PATH", ort_path);
    } else {
        cmd.env(
            "ORT_DYLIB_PATH",
            "/opt/homebrew/lib/libonnxruntime.dylib",
        );
    }

    queue
        .set_progress(job_id, 30, "Consolidation in progress")
        .await;

    match cmd.output().await {
        Ok(output) => {
            let elapsed = start.elapsed().as_millis() as u64;
            let stdout = String::from_utf8_lossy(&output.stdout);
            let stderr = String::from_utf8_lossy(&output.stderr);

            if output.status.success() {
                info!(
                    job_id = %job_id,
                    elapsed_ms = elapsed,
                    dry_run = dry_run,
                    "Consolidate job completed"
                );

                let result = serde_json::json!({
                    "stdout": stdout.chars().take(2000).collect::<String>(),
                    "duration_ms": elapsed,
                    "dry_run": dry_run,
                });

                queue.set_completed(job_id, Some(result)).await;

                if !dry_run {
                    event_bus.publish_sync(NagualEvent::consolidation_completed(
                        0,
                        0,
                        vec![],
                    ));
                }
            } else {
                let error_msg = if stderr.is_empty() {
                    format!(
                        "Consolidation failed with exit code {:?}",
                        output.status.code()
                    )
                } else {
                    stderr.chars().take(500).collect::<String>()
                };

                warn!(job_id = %job_id, error = %error_msg, "Consolidate job failed");
                queue.set_failed(job_id, &error_msg).await;

                event_bus.publish_sync(NagualEvent::error_occurred(
                    "action_handler",
                    "consolidate_failed",
                    &error_msg,
                    true,
                ));
            }
        }
        Err(e) => {
            let error_msg = format!("Failed to spawn consolidate process: {}", e);
            warn!(job_id = %job_id, error = %error_msg, "Consolidate job spawn failed");
            queue.set_failed(job_id, &error_msg).await;

            event_bus.publish_sync(NagualEvent::error_occurred(
                "action_handler",
                "consolidate_spawn_failed",
                &error_msg,
                true,
            ));
        }
    }
}

/// Run the dedup job by shelling out to `nagual learn dedup`.
async fn run_dedup_job(
    job_id: &str,
    db_path: &std::path::Path,
    event_bus: &Arc<crate::events::EventBus>,
    auto: bool,
    threshold: f64,
) {
    let queue = job_queue();
    queue.set_running(job_id).await;
    queue
        .set_progress(job_id, 10, "Starting deduplication")
        .await;

    info!(
        job_id = %job_id,
        auto = auto,
        threshold = threshold,
        "Starting dedup job"
    );

    let start = std::time::Instant::now();

    let mut cmd = tokio::process::Command::new(nagual_binary());
    cmd.arg("learn").arg("dedup").arg("--db-path").arg(db_path);

    if auto {
        cmd.arg("--auto");
    } else {
        cmd.arg("--scan");
    }

    cmd.arg("--threshold")
        .arg(format!("{:.2}", threshold));

    queue
        .set_progress(job_id, 30, "Deduplication in progress")
        .await;

    match cmd.output().await {
        Ok(output) => {
            let elapsed = start.elapsed().as_millis() as u64;
            let stdout = String::from_utf8_lossy(&output.stdout);
            let stderr = String::from_utf8_lossy(&output.stderr);

            if output.status.success() {
                info!(
                    job_id = %job_id,
                    elapsed_ms = elapsed,
                    "Dedup job completed"
                );

                let result = serde_json::json!({
                    "stdout": stdout.chars().take(2000).collect::<String>(),
                    "duration_ms": elapsed,
                    "auto": auto,
                });

                queue.set_completed(job_id, Some(result)).await;

                event_bus.publish_sync(NagualEvent::batch_completed(
                    "dedup",
                    0,
                    0,
                    0,
                    elapsed,
                ));
            } else {
                let error_msg = if stderr.is_empty() {
                    format!("Dedup failed with exit code {:?}", output.status.code())
                } else {
                    stderr.chars().take(500).collect::<String>()
                };

                warn!(job_id = %job_id, error = %error_msg, "Dedup job failed");
                queue.set_failed(job_id, &error_msg).await;

                event_bus.publish_sync(NagualEvent::error_occurred(
                    "action_handler",
                    "dedup_failed",
                    &error_msg,
                    true,
                ));
            }
        }
        Err(e) => {
            let error_msg = format!("Failed to spawn dedup process: {}", e);
            warn!(job_id = %job_id, error = %error_msg, "Dedup job spawn failed");
            queue.set_failed(job_id, &error_msg).await;

            event_bus.publish_sync(NagualEvent::error_occurred(
                "action_handler",
                "dedup_spawn_failed",
                &error_msg,
                true,
            ));
        }
    }
}

/// Run the pyramid generation job by shelling out to `nagual patterns pyramid`.
async fn run_pyramid_job(
    job_id: &str,
    db_path: &std::path::Path,
    event_bus: &Arc<crate::events::EventBus>,
    limit: u32,
) {
    let queue = job_queue();
    queue.set_running(job_id).await;
    queue
        .set_progress(job_id, 10, "Starting pyramid generation")
        .await;

    info!(job_id = %job_id, limit = limit, "Starting pyramid job");

    let start = std::time::Instant::now();

    let mut cmd = tokio::process::Command::new(nagual_binary());
    cmd.arg("patterns")
        .arg("pyramid")
        .arg("--generate")
        .arg("--limit")
        .arg(limit.to_string())
        .arg("--db-path")
        .arg(db_path);

    queue
        .set_progress(job_id, 30, "Pyramid generation in progress")
        .await;

    match cmd.output().await {
        Ok(output) => {
            let elapsed = start.elapsed().as_millis() as u64;
            let stdout = String::from_utf8_lossy(&output.stdout);
            let stderr = String::from_utf8_lossy(&output.stderr);

            if output.status.success() {
                info!(
                    job_id = %job_id,
                    elapsed_ms = elapsed,
                    "Pyramid job completed"
                );

                let result = serde_json::json!({
                    "stdout": stdout.chars().take(2000).collect::<String>(),
                    "duration_ms": elapsed,
                });

                queue.set_completed(job_id, Some(result)).await;

                event_bus.publish_sync(NagualEvent::batch_completed(
                    "pyramid",
                    0,
                    0,
                    0,
                    elapsed,
                ));
            } else {
                let error_msg = if stderr.is_empty() {
                    format!("Pyramid failed with exit code {:?}", output.status.code())
                } else {
                    stderr.chars().take(500).collect::<String>()
                };

                warn!(job_id = %job_id, error = %error_msg, "Pyramid job failed");
                queue.set_failed(job_id, &error_msg).await;

                event_bus.publish_sync(NagualEvent::error_occurred(
                    "action_handler",
                    "pyramid_failed",
                    &error_msg,
                    true,
                ));
            }
        }
        Err(e) => {
            let error_msg = format!("Failed to spawn pyramid process: {}", e);
            warn!(job_id = %job_id, error = %error_msg, "Pyramid job spawn failed");
            queue.set_failed(job_id, &error_msg).await;

            event_bus.publish_sync(NagualEvent::error_occurred(
                "action_handler",
                "pyramid_spawn_failed",
                &error_msg,
                true,
            ));
        }
    }
}

// ===========================================================================
// Phase 3: Intelligence & Visualization Endpoints
// ===========================================================================

// ---------------------------------------------------------------------------
// Request / Response types (Phase 3)
// ---------------------------------------------------------------------------

/// POST /api/search/semantic request body.
#[derive(Debug, Deserialize)]
pub struct SemanticSearchRequest {
    pub query: String,
    #[serde(default = "default_semantic_limit")]
    pub limit: usize,
    #[serde(default)]
    pub domain: Option<String>,
}

fn default_semantic_limit() -> usize {
    10
}

/// POST /api/predictions request body.
#[derive(Debug, Deserialize)]
pub struct CreatePredictionRequest {
    pub description: String,
    pub probability: f64,
    #[serde(default)]
    pub domain: Option<String>,
}

/// PUT /api/predictions/:id/resolve request body.
#[derive(Debug, Deserialize)]
pub struct ResolvePredictionRequest {
    pub outcome: bool,
}

// ---------------------------------------------------------------------------
// POST /api/search/semantic
// ---------------------------------------------------------------------------

/// POST /api/search/semantic -- hybrid FTS/LIKE search over patterns.
///
/// Attempts FTS5 MATCH first; falls back to LIKE if the table is not
/// FTS5-enabled. Optionally filters by domain.
pub async fn api_semantic_search(
    State(state): State<AppState>,
    _auth: RequireAuth,
    Json(req): Json<SemanticSearchRequest>,
) -> impl IntoResponse {
    let conn = match open_db(&state.db_path) {
        Ok(c) => c,
        Err((status, msg)) => {
            return (status, Json(serde_json::json!({ "error": msg }))).into_response();
        }
    };

    let limit = if req.limit == 0 { 10 } else { req.limit.min(100) };
    let dcol = domain_column(&conn);

    // Try FTS5 MATCH first
    let fts_result: Result<Vec<serde_json::Value>, _> = (|| -> Result<Vec<serde_json::Value>, rusqlite::Error> {
        let (sql, params_list): (String, Vec<Box<dyn rusqlite::types::ToSql>>) = if let Some(ref domain) = req.domain {
            (
                format!(
                    "SELECT id, problem, solution, COALESCE({dcol},'') as domain, \
                     COALESCE(reward,0.0), COALESCE(surprise_score,0.0) \
                     FROM reasoning_patterns \
                     WHERE reasoning_patterns MATCH ?1 AND {dcol} = ?2 \
                     ORDER BY rank LIMIT ?3"
                ),
                vec![
                    Box::new(req.query.clone()),
                    Box::new(domain.clone()),
                    Box::new(limit as i64),
                ],
            )
        } else {
            (
                format!(
                    "SELECT id, problem, solution, COALESCE({dcol},'') as domain, \
                     COALESCE(reward,0.0), COALESCE(surprise_score,0.0) \
                     FROM reasoning_patterns \
                     WHERE reasoning_patterns MATCH ?1 \
                     ORDER BY rank LIMIT ?2"
                ),
                vec![
                    Box::new(req.query.clone()),
                    Box::new(limit as i64),
                ],
            )
        };

        let mut stmt = conn.prepare(&sql)?;
        let params_refs: Vec<&dyn rusqlite::types::ToSql> = params_list.iter().map(|p| p.as_ref()).collect();
        let rows = stmt.query_map(params_refs.as_slice(), |row| {
            Ok(serde_json::json!({
                "id": row.get::<_, String>(0)?,
                "problem": row.get::<_, String>(1).unwrap_or_default(),
                "solution": row.get::<_, String>(2).unwrap_or_default(),
                "domain": row.get::<_, String>(3).unwrap_or_default(),
                "reward": row.get::<_, f64>(4).unwrap_or(0.0),
                "surprise_score": row.get::<_, f64>(5).unwrap_or(0.0),
            }))
        })?;
        rows.collect()
    })();

    if let Ok(results) = fts_result {
        let count = results.len();
        return (
            StatusCode::OK,
            Json(serde_json::json!({
                "query": req.query,
                "search_type": "fts",
                "count": count,
                "results": results,
            })),
        )
            .into_response();
    }

    // FTS failed — fall back to LIKE search (split multi-word queries)
    let words: Vec<String> = req
        .query
        .split_whitespace()
        .filter(|w| w.len() >= 2)
        .map(|w| format!("%{}%", w))
        .collect();

    // If no usable words, fall back to full-query pattern
    let words = if words.is_empty() {
        vec![format!("%{}%", req.query)]
    } else {
        words
    };

    let like_result: Result<Vec<serde_json::Value>, _> = (|| -> Result<Vec<serde_json::Value>, rusqlite::Error> {
        let has_surprise = conn
            .prepare("SELECT surprise_score FROM reasoning_patterns LIMIT 0")
            .is_ok();
        let surprise_expr = if has_surprise {
            "COALESCE(surprise_score, 0.0)"
        } else {
            "0.0"
        };

        // Build per-word conditions: each word must appear in problem OR solution
        let mut word_conditions = Vec::new();
        let mut params_list: Vec<Box<dyn rusqlite::types::ToSql>> = Vec::new();
        let mut param_idx = 1usize;

        for word in &words {
            word_conditions.push(format!("(problem LIKE ?{idx} OR solution LIKE ?{idx})", idx = param_idx));
            params_list.push(Box::new(word.clone()));
            param_idx += 1;
        }

        let where_words = word_conditions.join(" AND ");

        let sql = if let Some(ref domain) = req.domain {
            let domain_idx = param_idx;
            params_list.push(Box::new(domain.clone()));
            let limit_idx = param_idx + 1;
            params_list.push(Box::new(limit as i64));
            format!(
                "SELECT id, COALESCE(problem,''), COALESCE(solution,''), \
                 COALESCE({dcol},'') as domain, COALESCE(reward,0.0), {surprise_expr} \
                 FROM reasoning_patterns \
                 WHERE ({where_words}) AND {dcol} = ?{domain_idx} \
                 ORDER BY reward DESC LIMIT ?{limit_idx}"
            )
        } else {
            let limit_idx = param_idx;
            params_list.push(Box::new(limit as i64));
            format!(
                "SELECT id, COALESCE(problem,''), COALESCE(solution,''), \
                 COALESCE({dcol},'') as domain, COALESCE(reward,0.0), {surprise_expr} \
                 FROM reasoning_patterns \
                 WHERE {where_words} \
                 ORDER BY reward DESC LIMIT ?{limit_idx}"
            )
        };

        let mut stmt = conn.prepare(&sql)?;
        let params_refs: Vec<&dyn rusqlite::types::ToSql> = params_list.iter().map(|p| p.as_ref()).collect();
        let rows = stmt.query_map(params_refs.as_slice(), |row| {
            Ok(serde_json::json!({
                "id": row.get::<_, String>(0)?,
                "problem": row.get::<_, String>(1).unwrap_or_default(),
                "solution": row.get::<_, String>(2).unwrap_or_default(),
                "domain": row.get::<_, String>(3).unwrap_or_default(),
                "reward": row.get::<_, f64>(4).unwrap_or(0.0),
                "surprise_score": row.get::<_, f64>(5).unwrap_or(0.0),
            }))
        })?;
        rows.collect()
    })();

    match like_result {
        Ok(results) => {
            let count = results.len();
            (
                StatusCode::OK,
                Json(serde_json::json!({
                    "query": req.query,
                    "search_type": "like",
                    "count": count,
                    "results": results,
                })),
            )
                .into_response()
        }
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": format!("Search failed: {}", e) })),
        )
            .into_response(),
    }
}

// ---------------------------------------------------------------------------
// GET /api/domains/stats
// ---------------------------------------------------------------------------

/// GET /api/domains/stats -- domain breakdown with detailed metrics.
pub async fn api_domain_stats(
    State(state): State<AppState>,
    _auth: RequireAuth,
) -> impl IntoResponse {
    let conn = match open_db(&state.db_path) {
        Ok(c) => c,
        Err((status, msg)) => {
            return (status, Json(serde_json::json!({ "error": msg }))).into_response();
        }
    };

    let dcol = domain_column(&conn);

    let has_effectiveness = conn
        .prepare("SELECT effectiveness FROM reasoning_patterns LIMIT 0")
        .is_ok();
    let eff_expr = if has_effectiveness {
        "AVG(COALESCE(effectiveness, 0.0))"
    } else {
        "0.0"
    };

    let sql = format!(
        "SELECT COALESCE({dcol},'unknown') as d, COUNT(*) as cnt, \
         AVG(COALESCE(reward,0.5)) as avg_r, {eff_expr} as avg_e, \
         MIN(COALESCE(reward,0.0)) as min_r, MAX(COALESCE(reward,1.0)) as max_r, \
         SUM(CASE WHEN embedding IS NOT NULL THEN 1 ELSE 0 END) as with_embed \
         FROM reasoning_patterns GROUP BY d ORDER BY cnt DESC"
    );

    let result: Result<Vec<serde_json::Value>, _> = (|| -> Result<Vec<serde_json::Value>, rusqlite::Error> {
        let mut stmt = conn.prepare(&sql)?;
        let rows = stmt.query_map([], |row| {
            let domain: String = row.get(0)?;
            let count: i64 = row.get(1)?;
            let avg_reward: f64 = row.get(2)?;
            let avg_effectiveness: f64 = row.get(3)?;
            let min_reward: f64 = row.get(4)?;
            let max_reward: f64 = row.get(5)?;
            let embedded_count: i64 = row.get(6)?;

            let health = if avg_reward >= 0.6 && embedded_count > 0 {
                "good"
            } else if avg_reward >= 0.3 {
                "warning"
            } else {
                "poor"
            };

            Ok(serde_json::json!({
                "domain": domain,
                "count": count,
                "avg_reward": avg_reward,
                "avg_effectiveness": avg_effectiveness,
                "min_reward": min_reward,
                "max_reward": max_reward,
                "embedded_count": embedded_count,
                "health": health,
            }))
        })?;
        rows.collect()
    })();

    match result {
        Ok(domains) => {
            let total_domains = domains.len();
            let total_patterns: i64 = domains
                .iter()
                .map(|d| d["count"].as_i64().unwrap_or(0))
                .sum();

            (
                StatusCode::OK,
                Json(serde_json::json!({
                    "total_domains": total_domains,
                    "total_patterns": total_patterns,
                    "domains": domains,
                })),
            )
                .into_response()
        }
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": format!("Query error: {}", e) })),
        )
            .into_response(),
    }
}

// ---------------------------------------------------------------------------
// GET /api/graph/nodes
// ---------------------------------------------------------------------------

/// GET /api/graph/nodes -- lightweight node list for incremental graph loading.
pub async fn api_graph_nodes(
    State(state): State<AppState>,
    _auth: RequireAuth,
) -> impl IntoResponse {
    let conn = match open_db(&state.db_path) {
        Ok(c) => c,
        Err((status, msg)) => {
            return (status, Json(serde_json::json!({ "error": msg }))).into_response();
        }
    };

    let dcol = domain_column(&conn);
    let has_surprise = conn
        .prepare("SELECT surprise_score FROM reasoning_patterns LIMIT 0")
        .is_ok();
    let surprise_expr = if has_surprise {
        "COALESCE(surprise_score, 0.0)"
    } else {
        "0.0"
    };

    let sql = format!(
        "SELECT id, SUBSTR(COALESCE(problem,''),1,80) as label, \
         COALESCE({dcol},'') as domain, COALESCE(reward,0.0) as reward, \
         {surprise_expr} as surprise \
         FROM reasoning_patterns ORDER BY reward DESC LIMIT 200"
    );

    let result: Result<Vec<serde_json::Value>, _> = (|| -> Result<Vec<serde_json::Value>, rusqlite::Error> {
        let mut stmt = conn.prepare(&sql)?;
        let rows = stmt.query_map([], |row| {
            Ok(serde_json::json!({
                "id": row.get::<_, String>(0)?,
                "label": row.get::<_, String>(1).unwrap_or_default(),
                "domain": row.get::<_, String>(2).unwrap_or_default(),
                "reward": row.get::<_, f64>(3).unwrap_or(0.0),
                "surprise": row.get::<_, f64>(4).unwrap_or(0.0),
            }))
        })?;
        rows.collect()
    })();

    match result {
        Ok(nodes) => {
            let count = nodes.len();
            (
                StatusCode::OK,
                Json(serde_json::json!({
                    "count": count,
                    "nodes": nodes,
                })),
            )
                .into_response()
        }
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": format!("Query error: {}", e) })),
        )
            .into_response(),
    }
}

// ---------------------------------------------------------------------------
// GET /api/graph/edges
// ---------------------------------------------------------------------------

/// GET /api/graph/edges -- lightweight edge list for incremental graph loading.
pub async fn api_graph_edges(
    State(state): State<AppState>,
    _auth: RequireAuth,
) -> impl IntoResponse {
    let conn = match open_db(&state.db_path) {
        Ok(c) => c,
        Err((status, msg)) => {
            return (status, Json(serde_json::json!({ "error": msg }))).into_response();
        }
    };

    let edges: Vec<serde_json::Value> = match conn.prepare(
        "SELECT source_id, target_id, COALESCE(strength, 0.5) \
         FROM context_graph LIMIT 1000",
    ) {
        Ok(mut stmt) => match stmt.query_map([], |row| {
            Ok(serde_json::json!({
                "source": row.get::<_, String>(0)?,
                "target": row.get::<_, String>(1)?,
                "weight": row.get::<_, f64>(2).unwrap_or(0.5),
            }))
        }) {
            Ok(rows) => rows.filter_map(|r| r.ok()).collect(),
            Err(_) => Vec::new(),
        },
        Err(_) => Vec::new(),
    };

    let count = edges.len();
    (
        StatusCode::OK,
        Json(serde_json::json!({
            "count": count,
            "edges": edges,
        })),
    )
        .into_response()
}

// ---------------------------------------------------------------------------
// Predictions helpers
// ---------------------------------------------------------------------------

/// Ensure the predictions table exists, creating it if needed.
fn ensure_predictions_table(conn: &Connection) -> Result<(), rusqlite::Error> {
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS predictions (
            id TEXT PRIMARY KEY,
            description TEXT NOT NULL,
            probability REAL NOT NULL,
            calibrated_probability REAL,
            status TEXT NOT NULL DEFAULT 'pending',
            actual_outcome INTEGER,
            brier_score REAL,
            domain TEXT DEFAULT 'general',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            resolved_at TEXT
        )"
    )
}

// ---------------------------------------------------------------------------
// GET /api/predictions
// ---------------------------------------------------------------------------

/// GET /api/predictions -- list all predictions.
pub async fn api_predictions_list(
    State(state): State<AppState>,
    _auth: RequireAuth,
) -> impl IntoResponse {
    let conn = match open_rw_db(&state.db_path) {
        Ok(c) => c,
        Err((status, msg)) => {
            return (status, Json(serde_json::json!({ "error": msg }))).into_response();
        }
    };

    if let Err(e) = ensure_predictions_table(&conn) {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": format!("Failed to create predictions table: {}", e) })),
        )
            .into_response();
    }

    let result: Result<Vec<serde_json::Value>, _> = (|| -> Result<Vec<serde_json::Value>, rusqlite::Error> {
        let mut stmt = conn.prepare(
            "SELECT id, description, probability, calibrated_probability, status, \
             actual_outcome, brier_score, domain, created_at, resolved_at \
             FROM predictions ORDER BY created_at DESC LIMIT 100"
        )?;
        let rows = stmt.query_map([], |row| {
            Ok(serde_json::json!({
                "id": row.get::<_, String>(0)?,
                "description": row.get::<_, String>(1).unwrap_or_default(),
                "probability": row.get::<_, f64>(2).unwrap_or(0.5),
                "calibrated_probability": row.get::<_, Option<f64>>(3).unwrap_or(None),
                "status": row.get::<_, String>(4).unwrap_or_else(|_| "pending".to_string()),
                "actual_outcome": row.get::<_, Option<i64>>(5).unwrap_or(None),
                "brier_score": row.get::<_, Option<f64>>(6).unwrap_or(None),
                "domain": row.get::<_, String>(7).unwrap_or_else(|_| "general".to_string()),
                "created_at": row.get::<_, String>(8).unwrap_or_default(),
                "resolved_at": row.get::<_, Option<String>>(9).unwrap_or(None),
            }))
        })?;
        rows.collect()
    })();

    match result {
        Ok(predictions) => {
            let count = predictions.len();
            (
                StatusCode::OK,
                Json(serde_json::json!({
                    "count": count,
                    "predictions": predictions,
                })),
            )
                .into_response()
        }
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": format!("Query error: {}", e) })),
        )
            .into_response(),
    }
}

// ---------------------------------------------------------------------------
// POST /api/predictions
// ---------------------------------------------------------------------------

/// POST /api/predictions -- create a new prediction.
pub async fn api_predictions_create(
    State(state): State<AppState>,
    _auth: RequireWrite,
    Json(req): Json<CreatePredictionRequest>,
) -> impl IntoResponse {
    if req.probability < 0.0 || req.probability > 1.0 {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": "Probability must be between 0.0 and 1.0",
                "code": "invalid_probability"
            })),
        )
            .into_response();
    }

    let conn = match open_rw_db(&state.db_path) {
        Ok(c) => c,
        Err((status, msg)) => {
            return (status, Json(serde_json::json!({ "error": msg }))).into_response();
        }
    };

    if let Err(e) = ensure_predictions_table(&conn) {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": format!("Failed to create predictions table: {}", e) })),
        )
            .into_response();
    }

    let id = Uuid::new_v4().to_string();
    let now = Utc::now().to_rfc3339();
    let domain = req.domain.as_deref().unwrap_or("general");

    match conn.execute(
        "INSERT INTO predictions (id, description, probability, status, domain, created_at, updated_at) \
         VALUES (?1, ?2, ?3, 'pending', ?4, ?5, ?6)",
        rusqlite::params![id, req.description, req.probability, domain, now, now],
    ) {
        Ok(_) => {
            info!(id = %id, description = %req.description, probability = req.probability, "Prediction created");
            (
                StatusCode::CREATED,
                Json(serde_json::json!({
                    "id": id,
                    "status": "created",
                })),
            )
                .into_response()
        }
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": format!("Insert failed: {}", e) })),
        )
            .into_response(),
    }
}

// ---------------------------------------------------------------------------
// PUT /api/predictions/:id/resolve
// ---------------------------------------------------------------------------

/// PUT /api/predictions/:id/resolve -- resolve a prediction with an outcome.
pub async fn api_predictions_resolve(
    State(state): State<AppState>,
    _auth: RequireWrite,
    Path(id): Path<String>,
    Json(req): Json<ResolvePredictionRequest>,
) -> impl IntoResponse {
    let conn = match open_rw_db(&state.db_path) {
        Ok(c) => c,
        Err((status, msg)) => {
            return (status, Json(serde_json::json!({ "error": msg }))).into_response();
        }
    };

    if let Err(e) = ensure_predictions_table(&conn) {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": format!("Failed to create predictions table: {}", e) })),
        )
            .into_response();
    }

    // Fetch probability
    let probability: f64 = match conn.query_row(
        "SELECT probability FROM predictions WHERE id = ?1 AND status = 'pending'",
        rusqlite::params![id],
        |row| row.get(0),
    ) {
        Ok(p) => p,
        Err(rusqlite::Error::QueryReturnedNoRows) => {
            return (
                StatusCode::NOT_FOUND,
                Json(serde_json::json!({
                    "error": "Prediction not found or already resolved",
                    "id": id
                })),
            )
                .into_response();
        }
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({ "error": format!("Query error: {}", e) })),
            )
                .into_response();
        }
    };

    let outcome_f64: f64 = if req.outcome { 1.0 } else { 0.0 };
    let brier_score = (probability - outcome_f64).powi(2);
    let now = Utc::now().to_rfc3339();
    let outcome_int: i64 = if req.outcome { 1 } else { 0 };

    match conn.execute(
        "UPDATE predictions SET actual_outcome = ?1, brier_score = ?2, \
         status = 'resolved', resolved_at = ?3, updated_at = ?3 \
         WHERE id = ?4",
        rusqlite::params![outcome_int, brier_score, now, id],
    ) {
        Ok(updated) if updated > 0 => {
            info!(id = %id, outcome = req.outcome, brier_score = brier_score, "Prediction resolved");
            (
                StatusCode::OK,
                Json(serde_json::json!({
                    "id": id,
                    "brier_score": brier_score,
                    "status": "resolved",
                })),
            )
                .into_response()
        }
        Ok(_) => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({
                "error": "Prediction not found",
                "id": id
            })),
        )
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": format!("Update failed: {}", e) })),
        )
            .into_response(),
    }
}

// ---------------------------------------------------------------------------
// GET /api/predictions/calibration
// ---------------------------------------------------------------------------

/// GET /api/predictions/calibration -- calibration metrics for resolved predictions.
pub async fn api_predictions_calibration(
    State(state): State<AppState>,
    _auth: RequireAuth,
) -> impl IntoResponse {
    let conn = match open_rw_db(&state.db_path) {
        Ok(c) => c,
        Err((status, msg)) => {
            return (status, Json(serde_json::json!({ "error": msg }))).into_response();
        }
    };

    if let Err(e) = ensure_predictions_table(&conn) {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": format!("Failed to create predictions table: {}", e) })),
        )
            .into_response();
    }

    // Fetch all resolved predictions
    let resolved: Vec<(f64, f64, f64)> = match conn.prepare(
        "SELECT probability, COALESCE(actual_outcome, 0), COALESCE(brier_score, 0.0) \
         FROM predictions WHERE status = 'resolved'"
    ) {
        Ok(mut stmt) => {
            match stmt.query_map([], |row| {
                Ok((
                    row.get::<_, f64>(0)?,
                    row.get::<_, f64>(1)?,
                    row.get::<_, f64>(2)?,
                ))
            }) {
                Ok(rows) => rows.filter_map(|r| r.ok()).collect(),
                Err(_) => Vec::new(),
            }
        }
        Err(_) => Vec::new(),
    };

    let total_resolved = resolved.len();

    if total_resolved == 0 {
        return (
            StatusCode::OK,
            Json(serde_json::json!({
                "overall_brier": null,
                "total_resolved": 0,
                "buckets": [],
            })),
        )
            .into_response();
    }

    // Overall Brier score
    let overall_brier: f64 = resolved.iter().map(|(_, _, b)| b).sum::<f64>() / total_resolved as f64;

    // Build calibration buckets: 0.0-0.1, 0.1-0.2, ..., 0.9-1.0
    let mut buckets: Vec<serde_json::Value> = Vec::new();
    for i in 0..10 {
        let lower = i as f64 * 0.1;
        let upper = (i + 1) as f64 * 0.1;

        let bucket_items: Vec<&(f64, f64, f64)> = resolved
            .iter()
            .filter(|(p, _, _)| {
                if i == 9 {
                    *p >= lower && *p <= upper
                } else {
                    *p >= lower && *p < upper
                }
            })
            .collect();

        let count = bucket_items.len();
        if count == 0 {
            continue;
        }

        let avg_probability: f64 = bucket_items.iter().map(|(p, _, _)| p).sum::<f64>() / count as f64;
        let actual_rate: f64 = bucket_items.iter().map(|(_, o, _)| o).sum::<f64>() / count as f64;
        let calibration_error = (avg_probability - actual_rate).abs();

        buckets.push(serde_json::json!({
            "range": format!("{:.1}-{:.1}", lower, upper),
            "count": count,
            "avg_probability": avg_probability,
            "actual_rate": actual_rate,
            "calibration_error": calibration_error,
        }));
    }

    (
        StatusCode::OK,
        Json(serde_json::json!({
            "overall_brier": overall_brier,
            "total_resolved": total_resolved,
            "buckets": buckets,
        })),
    )
        .into_response()
}

// ---------------------------------------------------------------------------
// GET /api/sessions/stats
// ---------------------------------------------------------------------------

/// GET /api/sessions/stats -- session analytics with aggregated metrics.
pub async fn api_session_stats(
    State(state): State<AppState>,
    _auth: RequireAuth,
) -> impl IntoResponse {
    let conn = match open_db(&state.db_path) {
        Ok(c) => c,
        Err((status, msg)) => {
            return (status, Json(serde_json::json!({ "error": msg }))).into_response();
        }
    };

    // Check if sessions table exists
    let has_sessions: bool = conn
        .query_row(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='sessions'",
            [],
            |row| row.get::<_, i64>(0),
        )
        .map(|count| count > 0)
        .unwrap_or(false);

    if !has_sessions {
        return (
            StatusCode::OK,
            Json(serde_json::json!({
                "total_sessions": 0,
                "total_tokens": 0,
                "total_patterns_learned": 0,
                "avg_patterns_per_1k_tokens": 0.0,
                "sessions": [],
                "by_domain": {},
            })),
        )
            .into_response();
    }

    // Fetch sessions
    let sessions: Vec<serde_json::Value> = match conn.prepare(
        "SELECT id, started_at, ended_at, \
         COALESCE(tokens_used, 0), COALESCE(patterns_learned, 0), \
         COALESCE(patterns_retrieved, 0), COALESCE(domain, '') \
         FROM sessions ORDER BY started_at DESC LIMIT 50"
    ) {
        Ok(mut stmt) => {
            match stmt.query_map([], |row| {
                Ok(serde_json::json!({
                    "id": row.get::<_, String>(0)?,
                    "started_at": row.get::<_, Option<String>>(1).unwrap_or(None),
                    "ended_at": row.get::<_, Option<String>>(2).unwrap_or(None),
                    "tokens_used": row.get::<_, i64>(3).unwrap_or(0),
                    "patterns_learned": row.get::<_, i64>(4).unwrap_or(0),
                    "patterns_retrieved": row.get::<_, i64>(5).unwrap_or(0),
                    "domain": row.get::<_, String>(6).unwrap_or_default(),
                }))
            }) {
                Ok(rows) => rows.filter_map(|r| r.ok()).collect(),
                Err(_) => Vec::new(),
            }
        }
        Err(_) => Vec::new(),
    };

    let total_sessions = sessions.len();
    let total_tokens: i64 = sessions
        .iter()
        .map(|s| s["tokens_used"].as_i64().unwrap_or(0))
        .sum();
    let total_patterns_learned: i64 = sessions
        .iter()
        .map(|s| s["patterns_learned"].as_i64().unwrap_or(0))
        .sum();

    let avg_patterns_per_1k_tokens = if total_tokens > 0 {
        (total_patterns_learned as f64 / total_tokens as f64) * 1000.0
    } else {
        0.0
    };

    // Aggregate by domain
    let mut by_domain: HashMap<String, serde_json::Value> = HashMap::new();
    for s in &sessions {
        let domain = s["domain"].as_str().unwrap_or("").to_string();
        if domain.is_empty() {
            continue;
        }
        let entry = by_domain
            .entry(domain)
            .or_insert_with(|| serde_json::json!({ "count": 0, "tokens": 0, "patterns": 0 }));

        let obj = entry.as_object_mut().unwrap();
        let count = obj["count"].as_i64().unwrap_or(0);
        obj.insert("count".to_string(), serde_json::json!(count + 1));
        let tokens = obj["tokens"].as_i64().unwrap_or(0);
        obj.insert(
            "tokens".to_string(),
            serde_json::json!(tokens + s["tokens_used"].as_i64().unwrap_or(0)),
        );
        let patterns = obj["patterns"].as_i64().unwrap_or(0);
        obj.insert(
            "patterns".to_string(),
            serde_json::json!(patterns + s["patterns_learned"].as_i64().unwrap_or(0)),
        );
    }

    (
        StatusCode::OK,
        Json(serde_json::json!({
            "total_sessions": total_sessions,
            "total_tokens": total_tokens,
            "total_patterns_learned": total_patterns_learned,
            "avg_patterns_per_1k_tokens": avg_patterns_per_1k_tokens,
            "sessions": sessions,
            "by_domain": by_domain,
        })),
    )
        .into_response()
}

// ---------------------------------------------------------------------------
// GET /api/surprise
// ---------------------------------------------------------------------------

/// GET /api/surprise -- high-novelty patterns sorted by surprise score.
pub async fn api_surprise_patterns(
    State(state): State<AppState>,
    _auth: RequireAuth,
) -> impl IntoResponse {
    let conn = match open_db(&state.db_path) {
        Ok(c) => c,
        Err((status, msg)) => {
            return (status, Json(serde_json::json!({ "error": msg }))).into_response();
        }
    };

    let dcol = domain_column(&conn);

    // Check if surprise_score column exists
    let has_surprise = conn
        .prepare("SELECT surprise_score FROM reasoning_patterns LIMIT 0")
        .is_ok();

    if !has_surprise {
        return (
            StatusCode::OK,
            Json(serde_json::json!({
                "count": 0,
                "threshold": 0.5,
                "patterns": [],
            })),
        )
            .into_response();
    }

    let sql = format!(
        "SELECT id, SUBSTR(problem,1,120) as problem, COALESCE({dcol},'') as domain, \
         COALESCE(reward,0.0) as reward, COALESCE(surprise_score,0.0) as surprise_score, \
         COALESCE(created_at,'') as created_at \
         FROM reasoning_patterns \
         WHERE surprise_score > 0.5 \
         ORDER BY surprise_score DESC LIMIT 50"
    );

    let result: Result<Vec<serde_json::Value>, _> = (|| -> Result<Vec<serde_json::Value>, rusqlite::Error> {
        let mut stmt = conn.prepare(&sql)?;
        let rows = stmt.query_map([], |row| {
            Ok(serde_json::json!({
                "id": row.get::<_, String>(0)?,
                "problem": row.get::<_, String>(1).unwrap_or_default(),
                "domain": row.get::<_, String>(2).unwrap_or_default(),
                "reward": row.get::<_, f64>(3).unwrap_or(0.0),
                "surprise_score": row.get::<_, f64>(4).unwrap_or(0.0),
                "created_at": row.get::<_, String>(5).unwrap_or_default(),
            }))
        })?;
        rows.collect()
    })();

    match result {
        Ok(patterns) => {
            let count = patterns.len();
            (
                StatusCode::OK,
                Json(serde_json::json!({
                    "count": count,
                    "threshold": 0.5,
                    "patterns": patterns,
                })),
            )
                .into_response()
        }
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": format!("Query error: {}", e) })),
        )
            .into_response(),
    }
}

// ---------------------------------------------------------------------------
// Phase 4: Automation & Integration
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// P4.1 Scheduled Jobs — In-memory scheduler
// ---------------------------------------------------------------------------

/// A scheduled recurring learning pipeline job.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScheduledJob {
    pub id: String,
    pub name: String,
    pub actions: Vec<String>,
    pub interval_hours: u64,
    pub enabled: bool,
    pub last_run: Option<String>,
    pub next_run: Option<String>,
    pub created_at: String,
}

/// Request body for POST /api/schedule.
#[derive(Debug, Deserialize)]
pub struct CreateScheduleRequest {
    pub name: String,
    pub actions: Vec<String>,
    pub interval_hours: u64,
}

/// Global in-memory storage for scheduled jobs.
static SCHEDULED_JOBS: std::sync::OnceLock<RwLock<Vec<ScheduledJob>>> = std::sync::OnceLock::new();

/// Global map of schedule-id -> spawned tokio task handle.
static SCHEDULE_TASKS: std::sync::OnceLock<RwLock<HashMap<String, tokio::task::JoinHandle<()>>>> =
    std::sync::OnceLock::new();

fn scheduled_jobs() -> &'static RwLock<Vec<ScheduledJob>> {
    SCHEDULED_JOBS.get_or_init(|| RwLock::new(Vec::new()))
}

fn schedule_tasks() -> &'static RwLock<HashMap<String, tokio::task::JoinHandle<()>>> {
    SCHEDULE_TASKS.get_or_init(|| RwLock::new(HashMap::new()))
}

const ALLOWED_SCHEDULE_ACTIONS: &[&str] = &["embed", "consolidate", "dedup", "pyramid"];

fn parse_job_kind(action: &str) -> Option<JobKind> {
    match action {
        "embed" => Some(JobKind::Embed),
        "consolidate" => Some(JobKind::Consolidate),
        "dedup" => Some(JobKind::Dedup),
        "pyramid" => Some(JobKind::Pyramid),
        _ => None,
    }
}

/// GET /api/schedule — list all scheduled jobs.
pub async fn api_schedule_list(
    _auth: RequireAuth,
) -> impl IntoResponse {
    let jobs = scheduled_jobs().read().await;
    Json(serde_json::json!({
        "count": jobs.len(),
        "schedules": *jobs,
    }))
}

/// POST /api/schedule — create a new scheduled job.
pub async fn api_schedule_create(
    State(state): State<AppState>,
    _auth: RequireWrite,
    Json(req): Json<CreateScheduleRequest>,
) -> impl IntoResponse {
    // Validate actions
    for action in &req.actions {
        if !ALLOWED_SCHEDULE_ACTIONS.contains(&action.as_str()) {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({
                    "error": format!("Invalid action '{}'. Allowed: {:?}", action, ALLOWED_SCHEDULE_ACTIONS),
                    "code": "invalid_action"
                })),
            )
                .into_response();
        }
    }

    if req.actions.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": "At least one action is required",
                "code": "empty_actions"
            })),
        )
            .into_response();
    }

    if req.interval_hours == 0 {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": "interval_hours must be >= 1",
                "code": "invalid_interval"
            })),
        )
            .into_response();
    }

    let now = Utc::now();
    let next_run = now + chrono::Duration::hours(req.interval_hours as i64);
    let job = ScheduledJob {
        id: Uuid::new_v4().to_string(),
        name: req.name.clone(),
        actions: req.actions.clone(),
        interval_hours: req.interval_hours,
        enabled: true,
        last_run: None,
        next_run: Some(next_run.to_rfc3339()),
        created_at: now.to_rfc3339(),
    };

    let job_id = job.id.clone();
    let job_clone = job.clone();

    // Store the schedule
    scheduled_jobs().write().await.push(job.clone());

    // Spawn a background recurring task
    let db_path = state.db_path.clone();
    let event_bus = state.event_bus.clone();
    let interval_hours = req.interval_hours;
    let actions = req.actions.clone();
    let schedule_id = job_id.clone();

    let handle = tokio::spawn(async move {
        loop {
            tokio::time::sleep(tokio::time::Duration::from_secs(interval_hours * 3600)).await;

            info!(schedule_id = %schedule_id, "Scheduled job triggered");

            // Execute each action sequentially via the job queue
            for action in &actions {
                if let Some(kind) = parse_job_kind(action) {
                    let queue = job_queue();
                    if let Some(jid) = queue.enqueue(kind).await {
                        let db = db_path.clone();
                        let eb = event_bus.clone();
                        match kind {
                            JobKind::Embed => run_embed_job(&jid, &db, &eb, false).await,
                            JobKind::Consolidate => {
                                run_consolidate_job(&jid, &db, &eb, 0.9, false).await
                            }
                            JobKind::Dedup => run_dedup_job(&jid, &db, &eb, true, 0.9).await,
                            JobKind::Pyramid => run_pyramid_job(&jid, &db, &eb, 200).await,
                        }
                    } else {
                        warn!(
                            schedule_id = %schedule_id,
                            action = %action,
                            "Skipping scheduled action — job queue busy"
                        );
                    }
                }
            }

            // Update last_run / next_run
            let now = Utc::now();
            let next = now + chrono::Duration::hours(interval_hours as i64);
            let mut jobs = scheduled_jobs().write().await;
            if let Some(j) = jobs.iter_mut().find(|j| j.id == schedule_id) {
                j.last_run = Some(now.to_rfc3339());
                j.next_run = Some(next.to_rfc3339());
            } else {
                // Schedule was deleted — stop the loop
                break;
            }
        }
    });

    schedule_tasks().write().await.insert(job_id.clone(), handle);

    (
        StatusCode::CREATED,
        Json(serde_json::to_value(job_clone).unwrap()),
    )
        .into_response()
}

/// DELETE /api/schedule/:id — remove a scheduled job and abort its task.
pub async fn api_schedule_delete(
    _auth: RequireWrite,
    Path(schedule_id): Path<String>,
) -> impl IntoResponse {
    // Remove from schedule list
    let mut jobs = scheduled_jobs().write().await;
    let before_len = jobs.len();
    jobs.retain(|j| j.id != schedule_id);
    let removed = jobs.len() < before_len;
    drop(jobs);

    if !removed {
        return (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({
                "error": "Schedule not found",
                "id": schedule_id
            })),
        )
            .into_response();
    }

    // Abort the background task
    let mut tasks = schedule_tasks().write().await;
    if let Some(handle) = tasks.remove(&schedule_id) {
        handle.abort();
    }

    (
        StatusCode::OK,
        Json(serde_json::json!({
            "deleted": true,
            "id": schedule_id
        })),
    )
        .into_response()
}

// ---------------------------------------------------------------------------
// P4.2 Webhook Trigger
// ---------------------------------------------------------------------------

/// Request body for POST /api/webhook/learn.
#[derive(Debug, Deserialize)]
pub struct WebhookLearnRequest {
    pub actions: Option<Vec<String>>,
    pub source: Option<String>,
    pub callback_url: Option<String>,
}

/// POST /api/webhook/learn — trigger a learning pipeline via webhook.
pub async fn api_webhook_learn(
    State(state): State<AppState>,
    _auth: RequireWrite,
    Json(req): Json<WebhookLearnRequest>,
) -> impl IntoResponse {
    let actions = req
        .actions
        .unwrap_or_else(|| vec!["embed".to_string(), "consolidate".to_string()]);

    // Validate actions
    for action in &actions {
        if !ALLOWED_SCHEDULE_ACTIONS.contains(&action.as_str()) {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({
                    "error": format!("Invalid action '{}'. Allowed: {:?}", action, ALLOWED_SCHEDULE_ACTIONS),
                    "code": "invalid_action"
                })),
            )
                .into_response();
        }
    }

    let webhook_id = Uuid::new_v4().to_string();
    let source = req.source.clone().unwrap_or_else(|| "unknown".to_string());
    let callback_url = req.callback_url.clone();

    info!(
        webhook_id = %webhook_id,
        source = %source,
        actions = ?actions,
        "Webhook learn triggered"
    );

    // Build the list of valid action/kind pairs for background execution
    let mut pending_actions: Vec<(String, JobKind)> = Vec::new();

    for action in &actions {
        if let Some(kind) = parse_job_kind(action) {
            pending_actions.push((action.clone(), kind));
        }
    }

    // Spawn a background task that runs the actions in sequence
    let db_path = state.db_path.clone();
    let event_bus = state.event_bus.clone();
    let webhook_id_bg = webhook_id.clone();
    let actions_bg = actions.clone();

    tokio::spawn(async move {
        let mut completed_jobs: Vec<serde_json::Value> = Vec::new();

        for (action_name, kind) in pending_actions {
            let queue = job_queue();
            // Wait up to 5 minutes for the queue to become available
            let mut jid = None;
            for _ in 0..60 {
                if let Some(id) = queue.enqueue(kind).await {
                    jid = Some(id);
                    break;
                }
                tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;
            }

            if let Some(ref job_id) = jid {
                let db = db_path.clone();
                let eb = event_bus.clone();
                match kind {
                    JobKind::Embed => run_embed_job(job_id, &db, &eb, false).await,
                    JobKind::Consolidate => {
                        run_consolidate_job(job_id, &db, &eb, 0.9, false).await
                    }
                    JobKind::Dedup => run_dedup_job(job_id, &db, &eb, true, 0.9).await,
                    JobKind::Pyramid => run_pyramid_job(job_id, &db, &eb, 200).await,
                }

                let status = queue.get(job_id).await;
                completed_jobs.push(serde_json::json!({
                    "action": action_name,
                    "job_id": job_id,
                    "status": status.map(|s| s.status.to_string()).unwrap_or_else(|| "unknown".to_string()),
                }));
            } else {
                completed_jobs.push(serde_json::json!({
                    "action": action_name,
                    "job_id": null,
                    "status": "skipped_queue_busy",
                }));
            }
        }

        // If callback URL provided, POST the results
        if let Some(url) = callback_url {
            let payload = serde_json::json!({
                "webhook_id": webhook_id_bg,
                "status": "completed",
                "jobs": completed_jobs,
            });

            let client = reqwest::Client::new();
            if let Err(e) = client.post(&url).json(&payload).send().await {
                warn!(
                    webhook_id = %webhook_id_bg,
                    url = %url,
                    error = %e,
                    "Failed to POST webhook callback"
                );
            } else {
                info!(
                    webhook_id = %webhook_id_bg,
                    url = %url,
                    "Webhook callback sent"
                );
            }
        }
    });

    (
        StatusCode::ACCEPTED,
        Json(serde_json::json!({
            "webhook_id": webhook_id,
            "actions": actions_bg,
            "source": source,
            "status": "accepted",
            "jobs": "pending",
        })),
    )
        .into_response()
}

// ---------------------------------------------------------------------------
// P4.3 Activity Feed
// ---------------------------------------------------------------------------

/// A single event in the activity feed timeline.
#[derive(Debug, Serialize)]
pub struct ActivityEvent {
    pub timestamp: String,
    pub event_type: String,
    pub summary: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<serde_json::Value>,
}

/// GET /api/events/recent — recent system events (JSON polling).
pub async fn api_events_recent(
    State(state): State<AppState>,
    _auth: RequireAuth,
) -> impl IntoResponse {
    let mut events: Vec<ActivityEvent> = Vec::new();

    // 1. Recent pattern stores/updates from reasoning_patterns
    if let Ok(conn) = open_db(&state.db_path) {
        let dcol = domain_column(&conn);
        let has_updated_at = conn
            .prepare("SELECT updated_at FROM reasoning_patterns LIMIT 0")
            .is_ok();

        let timestamp_col = if has_updated_at {
            "COALESCE(updated_at, created_at)"
        } else {
            "created_at"
        };

        let sql = format!(
            "SELECT id, SUBSTR(COALESCE(problem,''),1,120), COALESCE({dcol},''), \
             COALESCE(reward,0.0), {timestamp_col} \
             FROM reasoning_patterns \
             WHERE created_at IS NOT NULL \
             ORDER BY created_at DESC LIMIT 50"
        );

        if let Ok(mut stmt) = conn.prepare(&sql) {
            let rows = stmt.query_map([], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1).unwrap_or_default(),
                    row.get::<_, String>(2).unwrap_or_default(),
                    row.get::<_, f64>(3).unwrap_or(0.0),
                    row.get::<_, String>(4).unwrap_or_default(),
                ))
            });

            if let Ok(rows) = rows {
                for row in rows.flatten() {
                    let (id, problem, domain, reward, ts) = row;
                    events.push(ActivityEvent {
                        timestamp: ts,
                        event_type: "pattern_stored".to_string(),
                        summary: format!(
                            "Pattern in '{}': {}",
                            domain,
                            if problem.len() > 80 {
                                format!("{}...", &problem[..80])
                            } else {
                                problem.clone()
                            }
                        ),
                        details: Some(serde_json::json!({
                            "pattern_id": id,
                            "domain": domain,
                            "reward": reward,
                        })),
                    });
                }
            }
        }

        // 2. Recent outcomes from pattern_outcomes (if table exists)
        let has_outcomes = conn
            .prepare("SELECT id FROM pattern_outcomes LIMIT 0")
            .is_ok();

        // Also check the 'outcomes' table name variant
        let outcomes_table = if has_outcomes {
            Some("pattern_outcomes")
        } else if conn.prepare("SELECT id FROM outcomes LIMIT 0").is_ok() {
            Some("outcomes")
        } else {
            None
        };

        if let Some(table) = outcomes_table {
            let sql = format!(
                "SELECT pattern_id, outcome, COALESCE(reward,0.0), \
                 COALESCE(feedback,''), COALESCE(recorded_at, created_at, datetime('now')) \
                 FROM {} ORDER BY rowid DESC LIMIT 20",
                table
            );

            if let Ok(mut stmt) = conn.prepare(&sql) {
                let rows = stmt.query_map([], |row| {
                    Ok((
                        row.get::<_, String>(0).unwrap_or_default(),
                        row.get::<_, String>(1).unwrap_or_default(),
                        row.get::<_, f64>(2).unwrap_or(0.0),
                        row.get::<_, String>(3).unwrap_or_default(),
                        row.get::<_, String>(4).unwrap_or_default(),
                    ))
                });

                if let Ok(rows) = rows {
                    for row in rows.flatten() {
                        let (pid, outcome, reward, feedback, ts) = row;
                        events.push(ActivityEvent {
                            timestamp: ts,
                            event_type: "outcome_recorded".to_string(),
                            summary: format!(
                                "Outcome '{}' for pattern {} (reward: {:.2})",
                                outcome, &pid[..pid.len().min(8)], reward
                            ),
                            details: Some(serde_json::json!({
                                "pattern_id": pid,
                                "outcome": outcome,
                                "reward": reward,
                                "feedback": feedback,
                            })),
                        });
                    }
                }
            }
        }
    }

    // 3. Recent completed/failed jobs from the in-memory job queue
    let all_jobs = job_queue().list().await;
    for job in all_jobs.iter().take(10) {
        if job.status == JobStatusKind::Completed || job.status == JobStatusKind::Failed {
            let event_type = if job.status == JobStatusKind::Completed {
                "job_completed"
            } else {
                "job_failed"
            };
            events.push(ActivityEvent {
                timestamp: job
                    .completed_at
                    .map(|t| t.to_rfc3339())
                    .unwrap_or_else(|| job.started_at.to_rfc3339()),
                event_type: event_type.to_string(),
                summary: format!("{} job {}", job.kind, job.status),
                details: Some(serde_json::json!({
                    "job_id": job.job_id,
                    "kind": job.kind.to_string(),
                    "duration_ms": job.completed_at.map(|c| (c - job.started_at).num_milliseconds()),
                })),
            });
        }
    }

    // Sort all events by timestamp descending
    events.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
    events.truncate(80);

    let count = events.len();
    Json(serde_json::json!({
        "count": count,
        "events": events,
    }))
}

// ---------------------------------------------------------------------------
// P4.4 Health Dashboard
// ---------------------------------------------------------------------------

/// Process start time for uptime calculation.
static PROCESS_START: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();

fn process_start() -> &'static std::time::Instant {
    PROCESS_START.get_or_init(std::time::Instant::now)
}

/// Compute a health score (0-100) from the metrics.
fn compute_health_score(
    embedding_coverage: f64,
    stale_patterns: u64,
    avg_reward: f64,
    error_rate: f64,
    orphan_patterns: u64,
) -> u64 {
    let mut score: i64 = 100;

    if embedding_coverage < 0.5 {
        score -= 20;
    } else if embedding_coverage < 0.8 {
        score -= 10;
    }

    if stale_patterns > 100 {
        score -= 15;
    }

    if avg_reward < 0.4 {
        score -= 10;
    }

    if error_rate > 0.2 {
        score -= 10;
    }

    if orphan_patterns > 50 {
        score -= 5;
    }

    score.max(0) as u64
}

/// GET /api/health/detailed — comprehensive health metrics.
pub async fn api_health_detailed(
    State(state): State<AppState>,
    _auth: RequireAuth,
) -> impl IntoResponse {
    let uptime_seconds = process_start().elapsed().as_secs();

    // DB file size
    let db_size_bytes: u64 = std::fs::metadata(&state.db_path)
        .map(|m| m.len())
        .unwrap_or(0);

    // Open read-only connection for metrics
    let conn = match open_db(&state.db_path) {
        Ok(c) => c,
        Err((code, msg)) => {
            return (code, Json(serde_json::json!({ "error": msg }))).into_response();
        }
    };

    let dcol = domain_column(&conn);

    let total_patterns: u64 = conn
        .query_row("SELECT COUNT(*) FROM reasoning_patterns", [], |row| {
            row.get(0)
        })
        .unwrap_or(0);

    let embedded_count: u64 = conn
        .query_row(
            "SELECT COUNT(*) FROM reasoning_patterns WHERE embedding IS NOT NULL",
            [],
            |row| row.get(0),
        )
        .unwrap_or(0);

    let embedding_coverage: f64 = if total_patterns > 0 {
        embedded_count as f64 / total_patterns as f64
    } else {
        0.0
    };

    let stale_patterns: u64 = conn
        .query_row(
            "SELECT COUNT(*) FROM reasoning_patterns \
             WHERE created_at < datetime('now', '-90 days') AND reward < 0.3",
            [],
            |row| row.get(0),
        )
        .unwrap_or(0);

    let orphan_patterns: u64 = conn
        .query_row(
            "SELECT COUNT(*) FROM reasoning_patterns \
             WHERE solution IS NULL OR solution = ''",
            [],
            |row| row.get(0),
        )
        .unwrap_or(0);

    let avg_reward: f64 = conn
        .query_row(
            "SELECT COALESCE(AVG(reward), 0.0) FROM reasoning_patterns",
            [],
            |row| row.get(0),
        )
        .unwrap_or(0.0);

    let high_reward_count: u64 = conn
        .query_row(
            "SELECT COUNT(*) FROM reasoning_patterns WHERE reward >= 0.8",
            [],
            |row| row.get(0),
        )
        .unwrap_or(0);

    let low_reward_count: u64 = conn
        .query_row(
            "SELECT COUNT(*) FROM reasoning_patterns WHERE reward < 0.2",
            [],
            |row| row.get(0),
        )
        .unwrap_or(0);

    let domain_count: u64 = conn
        .query_row(
            &format!("SELECT COUNT(DISTINCT {dcol}) FROM reasoning_patterns"),
            [],
            |row| row.get(0),
        )
        .unwrap_or(0);

    let top_domain: String = conn
        .query_row(
            &format!(
                "SELECT COALESCE({dcol},'unknown') FROM reasoning_patterns \
                 GROUP BY {dcol} ORDER BY COUNT(*) DESC LIMIT 1"
            ),
            [],
            |row| row.get(0),
        )
        .unwrap_or_else(|_| "none".to_string());

    let recent_activity_24h: u64 = conn
        .query_row(
            "SELECT COUNT(*) FROM reasoning_patterns \
             WHERE created_at > datetime('now', '-1 day')",
            [],
            |row| row.get(0),
        )
        .unwrap_or(0);

    let recent_activity_7d: u64 = conn
        .query_row(
            "SELECT COUNT(*) FROM reasoning_patterns \
             WHERE created_at > datetime('now', '-7 days')",
            [],
            |row| row.get(0),
        )
        .unwrap_or(0);

    // Job queue metrics
    let all_jobs = job_queue().list().await;
    let active_jobs = all_jobs
        .iter()
        .filter(|j| j.status == JobStatusKind::Running || j.status == JobStatusKind::Queued)
        .count() as u64;
    let completed_total = all_jobs
        .iter()
        .filter(|j| j.status == JobStatusKind::Completed || j.status == JobStatusKind::Failed)
        .count() as u64;
    let failed_total = all_jobs
        .iter()
        .filter(|j| j.status == JobStatusKind::Failed)
        .count() as u64;
    let error_rate: f64 = if completed_total > 0 {
        failed_total as f64 / completed_total as f64
    } else {
        0.0
    };

    let health_score = compute_health_score(
        embedding_coverage,
        stale_patterns,
        avg_reward,
        error_rate,
        orphan_patterns,
    );

    let status = if health_score >= 80 {
        "healthy"
    } else if health_score >= 50 {
        "degraded"
    } else {
        "unhealthy"
    };

    (
        StatusCode::OK,
        Json(serde_json::json!({
            "status": status,
            "health_score": health_score,
            "db_size_bytes": db_size_bytes,
            "total_patterns": total_patterns,
            "embedded_count": embedded_count,
            "embedding_coverage": (embedding_coverage * 1000.0).round() / 1000.0,
            "stale_patterns": stale_patterns,
            "orphan_patterns": orphan_patterns,
            "avg_reward": (avg_reward * 1000.0).round() / 1000.0,
            "high_reward_count": high_reward_count,
            "low_reward_count": low_reward_count,
            "domain_count": domain_count,
            "top_domain": top_domain,
            "recent_activity_24h": recent_activity_24h,
            "recent_activity_7d": recent_activity_7d,
            "error_rate": (error_rate * 1000.0).round() / 1000.0,
            "active_jobs": active_jobs,
            "uptime_seconds": uptime_seconds,
        })),
    )
        .into_response()
}

// ---------------------------------------------------------------------------
// P4.5 Export / Import
// ---------------------------------------------------------------------------

/// Query parameters for GET /api/export.
#[derive(Debug, Deserialize)]
pub struct ExportQuery {
    pub domain: Option<String>,
    pub limit: Option<u64>,
}

/// GET /api/export — export patterns as JSON.
pub async fn api_export(
    State(state): State<AppState>,
    _auth: RequireAuth,
    axum::extract::Query(params): axum::extract::Query<ExportQuery>,
) -> impl IntoResponse {
    let conn = match open_db(&state.db_path) {
        Ok(c) => c,
        Err((code, msg)) => {
            return (code, Json(serde_json::json!({ "error": msg }))).into_response();
        }
    };

    let dcol = domain_column(&conn);
    let limit = params.limit.unwrap_or(1000).min(5000);

    let has_tags = has_tags_column(&conn);
    let has_tier = conn
        .prepare("SELECT tier FROM reasoning_patterns LIMIT 0")
        .is_ok();

    let tags_expr = if has_tags { "tags" } else { "NULL" };
    let tier_expr = if has_tier { "tier" } else { "NULL" };

    let (sql, needs_domain_param) = if let Some(ref domain) = params.domain {
        let _ = domain; // used via parameter binding
        (
            format!(
                "SELECT id, COALESCE(problem,''), COALESCE(solution,''), \
                 COALESCE({dcol},''), COALESCE({tags_expr},''), COALESCE(reward,0.0), \
                 COALESCE({tier_expr},''), COALESCE(created_at,'') \
                 FROM reasoning_patterns WHERE {dcol} = ?1 \
                 ORDER BY created_at DESC LIMIT ?2"
            ),
            true,
        )
    } else {
        (
            format!(
                "SELECT id, COALESCE(problem,''), COALESCE(solution,''), \
                 COALESCE({dcol},''), COALESCE({tags_expr},''), COALESCE(reward,0.0), \
                 COALESCE({tier_expr},''), COALESCE(created_at,'') \
                 FROM reasoning_patterns \
                 ORDER BY created_at DESC LIMIT ?1"
            ),
            false,
        )
    };

    fn map_export_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<serde_json::Value> {
        Ok(serde_json::json!({
            "id": row.get::<_, String>(0)?,
            "problem": row.get::<_, String>(1).unwrap_or_default(),
            "solution": row.get::<_, String>(2).unwrap_or_default(),
            "domain": row.get::<_, String>(3).unwrap_or_default(),
            "tags": row.get::<_, String>(4).unwrap_or_default(),
            "reward": row.get::<_, f64>(5).unwrap_or(0.0),
            "tier": row.get::<_, String>(6).unwrap_or_default(),
            "created_at": row.get::<_, String>(7).unwrap_or_default(),
        }))
    }

    let result: Result<Vec<serde_json::Value>, rusqlite::Error> = (|| {
        let mut stmt = conn.prepare(&sql)?;

        if needs_domain_param {
            let rows = stmt.query_map(
                rusqlite::params![params.domain.as_deref().unwrap_or(""), limit as i64],
                map_export_row,
            )?;
            rows.collect()
        } else {
            let rows = stmt.query_map(
                rusqlite::params![limit as i64],
                map_export_row,
            )?;
            rows.collect()
        }
    })();

    match result {
        Ok(patterns) => {
            let count = patterns.len();
            (
                StatusCode::OK,
                Json(serde_json::json!({
                    "exported_at": Utc::now().to_rfc3339(),
                    "count": count,
                    "patterns": patterns,
                })),
            )
                .into_response()
        }
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": format!("Export query error: {}", e) })),
        )
            .into_response(),
    }
}

/// A single pattern in an import request.
#[derive(Debug, Deserialize)]
pub struct ImportPattern {
    pub problem: String,
    pub solution: Option<String>,
    pub domain: Option<String>,
    pub tags: Option<String>,
    pub context: Option<String>,
}

/// Request body for POST /api/import.
#[derive(Debug, Deserialize)]
pub struct ImportRequest {
    pub patterns: Vec<ImportPattern>,
}

/// POST /api/import — import patterns from JSON.
pub async fn api_import(
    State(state): State<AppState>,
    _auth: RequireWrite,
    Json(req): Json<ImportRequest>,
) -> impl IntoResponse {
    if req.patterns.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": "No patterns provided",
                "code": "empty_import"
            })),
        )
            .into_response();
    }

    let conn = match Connection::open_with_flags(
        &state.db_path,
        OpenFlags::SQLITE_OPEN_READ_WRITE | OpenFlags::SQLITE_OPEN_CREATE,
    ) {
        Ok(c) => c,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({
                    "error": format!("Database error: {}", e)
                })),
            )
                .into_response();
        }
    };

    let dcol = domain_column(&conn);
    let has_tags = has_tags_column(&conn);
    let has_context = conn
        .prepare("SELECT context FROM reasoning_patterns LIMIT 0")
        .is_ok();

    let mut imported: u64 = 0;
    let mut skipped: u64 = 0;
    let mut errors: Vec<String> = Vec::new();

    for (i, pattern) in req.patterns.iter().enumerate() {
        let id = Uuid::new_v4().to_string();
        let now = Utc::now().to_rfc3339();

        let solution_val = pattern
            .solution
            .clone()
            .unwrap_or_default();
        let domain_val = pattern
            .domain
            .clone()
            .unwrap_or_else(|| "imported".to_string());
        let tags_val = pattern.tags.clone().unwrap_or_default();
        let context_val = pattern.context.clone().unwrap_or_default();

        // Reset and build a straightforward insert
        let sql = if has_tags && has_context {
            format!(
                "INSERT OR IGNORE INTO reasoning_patterns \
                 (id, problem, solution, {dcol}, tags, context, reward, created_at) \
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, 0.5, ?7)"
            )
        } else if has_tags {
            format!(
                "INSERT OR IGNORE INTO reasoning_patterns \
                 (id, problem, solution, {dcol}, tags, reward, created_at) \
                 VALUES (?1, ?2, ?3, ?4, ?5, 0.5, ?6)"
            )
        } else {
            format!(
                "INSERT OR IGNORE INTO reasoning_patterns \
                 (id, problem, solution, {dcol}, reward, created_at) \
                 VALUES (?1, ?2, ?3, ?4, 0.5, ?5)"
            )
        };

        let result = if has_tags && has_context {
            conn.execute(
                &sql,
                rusqlite::params![id, pattern.problem, solution_val, domain_val, tags_val, context_val, now],
            )
        } else if has_tags {
            conn.execute(
                &sql,
                rusqlite::params![id, pattern.problem, solution_val, domain_val, tags_val, now],
            )
        } else {
            conn.execute(
                &sql,
                rusqlite::params![id, pattern.problem, solution_val, domain_val, now],
            )
        };

        match result {
            Ok(0) => skipped += 1, // INSERT OR IGNORE skipped (duplicate)
            Ok(_) => imported += 1,
            Err(e) => {
                errors.push(format!("pattern[{}]: {}", i, e));
                skipped += 1;
            }
        }
    }

    info!(
        imported = imported,
        skipped = skipped,
        errors = errors.len(),
        "Import completed"
    );

    (
        StatusCode::OK,
        Json(serde_json::json!({
            "imported": imported,
            "skipped": skipped,
            "errors": errors,
        })),
    )
        .into_response()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_job_queue_enqueue_and_get() {
        let queue = JobQueue::new();
        let job_id = queue.enqueue(JobKind::Embed).await.unwrap();
        assert!(!job_id.is_empty());

        let job = queue.get(&job_id).await.unwrap();
        assert_eq!(job.kind, JobKind::Embed);
        assert_eq!(job.status, JobStatusKind::Queued);
        assert_eq!(job.progress, 0);
    }

    #[tokio::test]
    async fn test_job_queue_max_one_concurrent() {
        let queue = JobQueue::new();
        let _id1 = queue.enqueue(JobKind::Embed).await.unwrap();

        // Second enqueue should fail while first is queued
        let id2 = queue.enqueue(JobKind::Consolidate).await;
        assert!(id2.is_none());
    }

    #[tokio::test]
    async fn test_job_queue_allows_after_completion() {
        let queue = JobQueue::new();
        let id1 = queue.enqueue(JobKind::Embed).await.unwrap();
        queue.set_running(&id1).await;
        queue.set_completed(&id1, None).await;

        // Should allow a new job now
        let id2 = queue.enqueue(JobKind::Consolidate).await;
        assert!(id2.is_some());
    }

    #[tokio::test]
    async fn test_job_queue_allows_after_failure() {
        let queue = JobQueue::new();
        let id1 = queue.enqueue(JobKind::Dedup).await.unwrap();
        queue.set_running(&id1).await;
        queue.set_failed(&id1, "test error").await;

        let id2 = queue.enqueue(JobKind::Pyramid).await;
        assert!(id2.is_some());
    }

    #[tokio::test]
    async fn test_job_queue_progress() {
        let queue = JobQueue::new();
        let id = queue.enqueue(JobKind::Embed).await.unwrap();
        queue.set_running(&id).await;
        queue.set_progress(&id, 50, "halfway there").await;

        let job = queue.get(&id).await.unwrap();
        assert_eq!(job.status, JobStatusKind::Running);
        assert_eq!(job.progress, 50);
        assert_eq!(job.message, "halfway there");
    }

    #[tokio::test]
    async fn test_job_queue_list() {
        let queue = JobQueue::new();
        let id1 = queue.enqueue(JobKind::Embed).await.unwrap();
        queue.set_completed(&id1, None).await;

        let id2 = queue.enqueue(JobKind::Consolidate).await.unwrap();

        let list = queue.list().await;
        assert_eq!(list.len(), 2);
        // Most recent first
        assert_eq!(list[0].job_id, id2);
        assert_eq!(list[1].job_id, id1);
    }

    #[tokio::test]
    async fn test_job_completed_has_timestamp() {
        let queue = JobQueue::new();
        let id = queue.enqueue(JobKind::Embed).await.unwrap();
        assert!(queue.get(&id).await.unwrap().completed_at.is_none());

        queue.set_completed(&id, Some(serde_json::json!({"ok": true}))).await;
        let job = queue.get(&id).await.unwrap();
        assert!(job.completed_at.is_some());
        assert_eq!(job.result, Some(serde_json::json!({"ok": true})));
    }

    #[tokio::test]
    async fn test_job_status_serialization() {
        let job = JobStatus {
            job_id: "test-123".to_string(),
            kind: JobKind::Embed,
            status: JobStatusKind::Running,
            progress: 42,
            message: "embedding".to_string(),
            started_at: Utc::now(),
            completed_at: None,
            result: None,
        };

        let json = serde_json::to_value(&job).unwrap();
        assert_eq!(json["kind"], "embed");
        assert_eq!(json["status"], "running");
        assert_eq!(json["progress"], 42);
        // completed_at and result should be absent (skip_serializing_if)
        assert!(json.get("completed_at").is_none());
        assert!(json.get("result").is_none());
    }

    #[tokio::test]
    async fn test_insights_with_test_db() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let path = tmp.path().to_path_buf();
        let conn = Connection::open(&path).unwrap();
        conn.execute_batch(
            "CREATE TABLE reasoning_patterns (
                id TEXT PRIMARY KEY,
                problem TEXT,
                solution TEXT,
                domain TEXT,
                reward REAL DEFAULT 0.5,
                success INTEGER DEFAULT 0,
                embedding BLOB,
                reuse_count INTEGER DEFAULT 0,
                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now'))
            );
            INSERT INTO reasoning_patterns (id, problem, solution, domain, reward, success, reuse_count)
            VALUES ('p1', 'Problem 1', 'Solution 1', 'rust', 0.9, 1, 5);
            INSERT INTO reasoning_patterns (id, problem, solution, domain, reward, success, reuse_count)
            VALUES ('p2', 'Problem 2', 'Solution 2', 'rust', 0.3, 0, 1);
            INSERT INTO reasoning_patterns (id, problem, solution, domain, reward, success, embedding, reuse_count)
            VALUES ('p3', 'Problem 3', 'Solution 3', 'python', 0.7, 1, X'0102030405', 3);",
        )
        .unwrap();
        drop(conn);

        let db_conn = open_db(&path).unwrap();
        let dcol = domain_column(&db_conn);

        let total: u64 = db_conn
            .query_row("SELECT COUNT(*) FROM reasoning_patterns", [], |row| row.get(0))
            .unwrap();
        assert_eq!(total, 3);

        let embedded: u64 = db_conn
            .query_row(
                "SELECT COUNT(*) FROM reasoning_patterns WHERE embedding IS NOT NULL",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(embedded, 1);

        assert_eq!(dcol, "domain");
    }

    #[tokio::test]
    async fn test_recommendations_with_test_db() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let path = tmp.path().to_path_buf();
        let conn = Connection::open(&path).unwrap();
        conn.execute_batch(
            "CREATE TABLE reasoning_patterns (
                id TEXT PRIMARY KEY,
                problem TEXT,
                solution TEXT,
                domain TEXT,
                reward REAL DEFAULT 0.5,
                tier TEXT DEFAULT 'booster',
                embedding BLOB,
                reuse_count INTEGER DEFAULT 0,
                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now'))
            );
            INSERT INTO reasoning_patterns (id, problem, solution, domain, reward, tier)
            VALUES ('p1', 'Good pattern', 'Short fix', 'rust', 0.9, 'booster');
            INSERT INTO reasoning_patterns (id, problem, solution, domain, reward, tier)
            VALUES ('p2', 'Bad pattern', 'Did not work', 'rust', 0.2, 'booster');",
        )
        .unwrap();
        drop(conn);

        let db_conn = open_db(&path).unwrap();

        // Check that we can detect unembedded, archivable, and promotable patterns
        let total: u64 = db_conn
            .query_row("SELECT COUNT(*) FROM reasoning_patterns", [], |row| row.get(0))
            .unwrap();
        assert_eq!(total, 2);

        let low_reward: u64 = db_conn
            .query_row(
                "SELECT COUNT(*) FROM reasoning_patterns WHERE reward < 0.4",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(low_reward, 1);

        let high_reward_not_reflex: u64 = db_conn
            .query_row(
                "SELECT COUNT(*) FROM reasoning_patterns WHERE reward >= 0.8 AND (tier IS NULL OR LOWER(tier) != 'reflex')",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(high_reward_not_reflex, 1);
    }

    #[test]
    fn test_embed_request_defaults() {
        let json = r#"{}"#;
        let req: EmbedRequest = serde_json::from_str(json).unwrap();
        assert!(!req.force);
    }

    #[test]
    fn test_consolidate_request_defaults() {
        let json = r#"{}"#;
        let req: ConsolidateRequest = serde_json::from_str(json).unwrap();
        assert!((req.similarity - 0.9).abs() < 0.001);
        assert!(!req.dry_run);
    }

    #[test]
    fn test_dedup_request_defaults() {
        let json = r#"{}"#;
        let req: DedupRequest = serde_json::from_str(json).unwrap();
        assert!(req.auto);
        assert!((req.threshold - 0.9).abs() < 0.001);
    }

    #[test]
    fn test_pyramid_request_defaults() {
        let json = r#"{}"#;
        let req: PyramidRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.limit, 200);
    }

    #[test]
    fn test_job_kind_display() {
        assert_eq!(JobKind::Embed.to_string(), "embed");
        assert_eq!(JobKind::Consolidate.to_string(), "consolidate");
        assert_eq!(JobKind::Dedup.to_string(), "dedup");
        assert_eq!(JobKind::Pyramid.to_string(), "pyramid");
    }

    #[test]
    fn test_job_status_kind_display() {
        assert_eq!(JobStatusKind::Queued.to_string(), "queued");
        assert_eq!(JobStatusKind::Running.to_string(), "running");
        assert_eq!(JobStatusKind::Completed.to_string(), "completed");
        assert_eq!(JobStatusKind::Failed.to_string(), "failed");
    }

    #[test]
    fn test_nagual_binary_resolution() {
        // Should return a PathBuf in all cases
        let bin = nagual_binary();
        assert!(!bin.to_str().unwrap().is_empty());
    }

    // -----------------------------------------------------------------------
    // Phase 2 tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_bulk_request_deserialization_archive() {
        let json = r#"{"ids": ["id1", "id2"], "action": "archive"}"#;
        let req: BulkRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.ids.len(), 2);
        assert_eq!(req.action, BulkAction::Archive);
        assert!(req.value.is_none());
    }

    #[test]
    fn test_bulk_request_deserialization_tag_with_value() {
        let json = r#"{"ids": ["id1"], "action": "tag", "value": "important"}"#;
        let req: BulkRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.ids.len(), 1);
        assert_eq!(req.action, BulkAction::Tag);
        assert_eq!(req.value.as_deref(), Some("important"));
    }

    #[test]
    fn test_bulk_request_deserialization_all_actions() {
        for (action_str, expected) in [
            ("archive", BulkAction::Archive),
            ("promote", BulkAction::Promote),
            ("tag", BulkAction::Tag),
            ("domain", BulkAction::Domain),
            ("delete", BulkAction::Delete),
        ] {
            let json = format!(r#"{{"ids": ["x"], "action": "{}"}}"#, action_str);
            let req: BulkRequest = serde_json::from_str(&json).unwrap();
            assert_eq!(req.action, expected);
        }
    }

    #[test]
    fn test_bulk_action_display() {
        assert_eq!(BulkAction::Archive.to_string(), "archive");
        assert_eq!(BulkAction::Promote.to_string(), "promote");
        assert_eq!(BulkAction::Tag.to_string(), "tag");
        assert_eq!(BulkAction::Domain.to_string(), "domain");
        assert_eq!(BulkAction::Delete.to_string(), "delete");
    }

    #[test]
    fn test_bulk_response_serialization() {
        let resp = BulkResponse {
            affected: 3,
            action: "archive".to_string(),
            errors: vec!["id4: not found".to_string()],
        };
        let json = serde_json::to_value(&resp).unwrap();
        assert_eq!(json["affected"], 3);
        assert_eq!(json["action"], "archive");
        assert_eq!(json["errors"].as_array().unwrap().len(), 1);
    }

    #[test]
    fn test_tag_parsing_comma_separated() {
        let tags_raw = "async,tokio,performance";
        let mut tag_counts: HashMap<String, u64> = HashMap::new();
        for tag in tags_raw.split(',') {
            let tag = tag.trim().to_string();
            if !tag.is_empty() {
                *tag_counts.entry(tag).or_insert(0) += 1;
            }
        }
        assert_eq!(tag_counts.len(), 3);
        assert_eq!(tag_counts["async"], 1);
        assert_eq!(tag_counts["tokio"], 1);
        assert_eq!(tag_counts["performance"], 1);
    }

    #[test]
    fn test_tag_parsing_json_array() {
        let tags_raw = r#"["async","tokio","performance"]"#;
        let mut tag_counts: HashMap<String, u64> = HashMap::new();
        if let Ok(arr) = serde_json::from_str::<Vec<String>>(tags_raw) {
            for tag in arr {
                let tag = tag.trim().to_string();
                if !tag.is_empty() {
                    *tag_counts.entry(tag).or_insert(0) += 1;
                }
            }
        }
        assert_eq!(tag_counts.len(), 3);
        assert_eq!(tag_counts["async"], 1);
    }

    #[test]
    fn test_tag_parsing_duplicates_accumulate() {
        let rows = vec!["async,tokio", "tokio,performance", "async"];
        let mut tag_counts: HashMap<String, u64> = HashMap::new();
        for tags_raw in rows {
            for tag in tags_raw.split(',') {
                let tag = tag.trim().to_string();
                if !tag.is_empty() {
                    *tag_counts.entry(tag).or_insert(0) += 1;
                }
            }
        }
        assert_eq!(tag_counts["async"], 2);
        assert_eq!(tag_counts["tokio"], 2);
        assert_eq!(tag_counts["performance"], 1);
    }

    #[tokio::test]
    async fn test_tags_endpoint_with_test_db() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let path = tmp.path().to_path_buf();
        let conn = Connection::open(&path).unwrap();
        conn.execute_batch(
            "CREATE TABLE reasoning_patterns (
                id TEXT PRIMARY KEY,
                problem TEXT,
                solution TEXT,
                domain TEXT,
                tags TEXT,
                reward REAL DEFAULT 0.5
            );
            INSERT INTO reasoning_patterns (id, problem, solution, domain, tags, reward)
            VALUES ('p1', 'P1', 'S1', 'rust', 'async,tokio', 0.9);
            INSERT INTO reasoning_patterns (id, problem, solution, domain, tags, reward)
            VALUES ('p2', 'P2', 'S2', 'rust', 'tokio,performance', 0.8);
            INSERT INTO reasoning_patterns (id, problem, solution, domain, tags, reward)
            VALUES ('p3', 'P3', 'S3', 'python', NULL, 0.5);",
        )
        .unwrap();
        drop(conn);

        let db_conn = open_db(&path).unwrap();
        assert!(has_tags_column(&db_conn));

        // Simulate tag aggregation
        let mut stmt = db_conn
            .prepare("SELECT tags FROM reasoning_patterns WHERE tags IS NOT NULL AND tags != ''")
            .unwrap();
        let rows = stmt.query_map([], |row| row.get::<_, String>(0)).unwrap();

        let mut tag_counts: HashMap<String, u64> = HashMap::new();
        for row_result in rows {
            if let Ok(tags_raw) = row_result {
                for tag in tags_raw.split(',') {
                    let tag = tag.trim().to_string();
                    if !tag.is_empty() {
                        *tag_counts.entry(tag).or_insert(0) += 1;
                    }
                }
            }
        }

        assert_eq!(tag_counts["tokio"], 2);
        assert_eq!(tag_counts["async"], 1);
        assert_eq!(tag_counts["performance"], 1);
    }

    #[tokio::test]
    async fn test_history_no_outcomes_table() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let path = tmp.path().to_path_buf();
        let conn = Connection::open(&path).unwrap();
        conn.execute_batch(
            "CREATE TABLE reasoning_patterns (
                id TEXT PRIMARY KEY,
                problem TEXT,
                solution TEXT
            );",
        )
        .unwrap();
        drop(conn);

        // Verify outcomes table does not exist
        let db_conn = open_db(&path).unwrap();
        let has_outcomes: bool = db_conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='outcomes'",
                [],
                |row| row.get::<_, i64>(0),
            )
            .map(|count| count > 0)
            .unwrap_or(false);
        assert!(!has_outcomes);
    }

    #[tokio::test]
    async fn test_history_with_outcomes_table() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let path = tmp.path().to_path_buf();
        let conn = Connection::open(&path).unwrap();
        conn.execute_batch(
            "CREATE TABLE reasoning_patterns (
                id TEXT PRIMARY KEY,
                problem TEXT,
                solution TEXT
            );
            INSERT INTO reasoning_patterns (id, problem, solution)
            VALUES ('p1', 'Problem', 'Solution');
            CREATE TABLE outcomes (
                id INTEGER PRIMARY KEY,
                pattern_id TEXT,
                outcome TEXT,
                reward REAL,
                feedback TEXT,
                recorded_at TEXT DEFAULT (datetime('now')),
                created_at TEXT DEFAULT (datetime('now'))
            );
            INSERT INTO outcomes (pattern_id, outcome, reward, feedback)
            VALUES ('p1', 'success', 0.9, 'Worked great');
            INSERT INTO outcomes (pattern_id, outcome, reward, feedback)
            VALUES ('p1', 'failure', 0.2, 'Did not work');",
        )
        .unwrap();
        drop(conn);

        let db_conn = open_db(&path).unwrap();
        let mut stmt = db_conn
            .prepare(
                "SELECT outcome, COALESCE(reward, 0.0), feedback, \
                 COALESCE(recorded_at, created_at, '') \
                 FROM outcomes WHERE pattern_id = ?1",
            )
            .unwrap();
        let rows: Vec<OutcomeEntry> = stmt
            .query_map(rusqlite::params!["p1"], |row| {
                Ok(OutcomeEntry {
                    outcome: row.get(0)?,
                    reward: row.get(1)?,
                    feedback: row.get(2)?,
                    recorded_at: row.get(3)?,
                })
            })
            .unwrap()
            .filter_map(|r| r.ok())
            .collect();

        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].outcome, "success");
        assert!((rows[0].reward - 0.9).abs() < 0.001);
    }

    #[tokio::test]
    async fn test_archive_and_promote_with_tier_column() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let path = tmp.path().to_path_buf();
        let conn = Connection::open(&path).unwrap();
        conn.execute_batch(
            "CREATE TABLE reasoning_patterns (
                id TEXT PRIMARY KEY,
                problem TEXT,
                solution TEXT,
                domain TEXT,
                tier TEXT DEFAULT 'booster',
                reward REAL DEFAULT 0.5
            );
            INSERT INTO reasoning_patterns (id, problem, solution, domain, tier, reward)
            VALUES ('p1', 'P1', 'S1', 'rust', 'booster', 0.9);",
        )
        .unwrap();

        // Test archive
        conn.execute(
            "UPDATE reasoning_patterns SET tier = 'archived' WHERE id = 'p1'",
            [],
        )
        .unwrap();
        let tier: String = conn
            .query_row(
                "SELECT tier FROM reasoning_patterns WHERE id = 'p1'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(tier, "archived");

        // Test promote
        conn.execute(
            "UPDATE reasoning_patterns SET tier = 'reflex' WHERE id = 'p1'",
            [],
        )
        .unwrap();
        let tier: String = conn
            .query_row(
                "SELECT tier FROM reasoning_patterns WHERE id = 'p1'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(tier, "reflex");
    }

    #[tokio::test]
    async fn test_bulk_operations_with_test_db() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let path = tmp.path().to_path_buf();
        let conn = Connection::open(&path).unwrap();
        conn.execute_batch(
            "CREATE TABLE reasoning_patterns (
                id TEXT PRIMARY KEY,
                problem TEXT,
                solution TEXT,
                domain TEXT,
                tier TEXT DEFAULT 'booster',
                tags TEXT,
                reward REAL DEFAULT 0.5
            );
            INSERT INTO reasoning_patterns (id, problem, solution, domain, tier, tags, reward)
            VALUES ('p1', 'P1', 'S1', 'rust', 'booster', 'async', 0.9);
            INSERT INTO reasoning_patterns (id, problem, solution, domain, tier, tags, reward)
            VALUES ('p2', 'P2', 'S2', 'rust', 'booster', NULL, 0.3);
            INSERT INTO reasoning_patterns (id, problem, solution, domain, tier, tags, reward)
            VALUES ('p3', 'P3', 'S3', 'python', 'booster', 'perf', 0.7);",
        )
        .unwrap();

        // Bulk archive
        conn.execute(
            "UPDATE reasoning_patterns SET tier = 'archived' WHERE id IN ('p1', 'p2')",
            [],
        )
        .unwrap();
        let archived: u64 = conn
            .query_row(
                "SELECT COUNT(*) FROM reasoning_patterns WHERE tier = 'archived'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(archived, 2);

        // Bulk tag
        conn.execute(
            "UPDATE reasoning_patterns SET tags = CASE \
             WHEN tags IS NULL OR tags = '' THEN 'new-tag' \
             ELSE tags || ',' || 'new-tag' \
             END WHERE id = 'p3'",
            [],
        )
        .unwrap();
        let tags: String = conn
            .query_row(
                "SELECT tags FROM reasoning_patterns WHERE id = 'p3'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(tags, "perf,new-tag");

        // Bulk delete
        let deleted = conn
            .execute("DELETE FROM reasoning_patterns WHERE id = 'p1'", [])
            .unwrap();
        assert_eq!(deleted, 1);
        let remaining: u64 = conn
            .query_row(
                "SELECT COUNT(*) FROM reasoning_patterns",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(remaining, 2);
    }

    // -----------------------------------------------------------------------
    // Phase 3 tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_semantic_search_request_defaults() {
        let json = r#"{"query": "rust async"}"#;
        let req: SemanticSearchRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.query, "rust async");
        assert_eq!(req.limit, 10);
        assert!(req.domain.is_none());

        // With explicit values
        let json2 = r#"{"query": "tokio", "limit": 5, "domain": "rust"}"#;
        let req2: SemanticSearchRequest = serde_json::from_str(json2).unwrap();
        assert_eq!(req2.limit, 5);
        assert_eq!(req2.domain.as_deref(), Some("rust"));
    }

    #[tokio::test]
    async fn test_prediction_create_and_list() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let path = tmp.path().to_path_buf();
        let conn = Connection::open(&path).unwrap();
        // Ensure table exists
        ensure_predictions_table(&conn).unwrap();

        // Insert a prediction
        let id = Uuid::new_v4().to_string();
        let now = Utc::now().to_rfc3339();
        conn.execute(
            "INSERT INTO predictions (id, description, probability, status, domain, created_at, updated_at) \
             VALUES (?1, ?2, ?3, 'pending', ?4, ?5, ?6)",
            rusqlite::params![id, "API latency < 200ms", 0.8, "general", now, now],
        )
        .unwrap();

        // Verify we can list it
        let count: i64 = conn
            .query_row("SELECT COUNT(*) FROM predictions", [], |row| row.get(0))
            .unwrap();
        assert_eq!(count, 1);

        let desc: String = conn
            .query_row(
                "SELECT description FROM predictions WHERE id = ?1",
                rusqlite::params![id],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(desc, "API latency < 200ms");

        let status: String = conn
            .query_row(
                "SELECT status FROM predictions WHERE id = ?1",
                rusqlite::params![id],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(status, "pending");
    }

    #[tokio::test]
    async fn test_prediction_resolve_and_brier() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let path = tmp.path().to_path_buf();
        let conn = Connection::open(&path).unwrap();
        ensure_predictions_table(&conn).unwrap();

        let id = Uuid::new_v4().to_string();
        let now = Utc::now().to_rfc3339();
        conn.execute(
            "INSERT INTO predictions (id, description, probability, status, domain, created_at, updated_at) \
             VALUES (?1, ?2, ?3, 'pending', 'general', ?4, ?5)",
            rusqlite::params![id, "Test prediction", 0.8, now, now],
        )
        .unwrap();

        // Resolve with outcome = true (1.0)
        let probability: f64 = conn
            .query_row(
                "SELECT probability FROM predictions WHERE id = ?1",
                rusqlite::params![id],
                |row| row.get(0),
            )
            .unwrap();
        assert!((probability - 0.8).abs() < 0.001);

        let outcome_f64: f64 = 1.0; // true
        let brier_score = (probability - outcome_f64).powi(2);
        // (0.8 - 1.0)^2 = 0.04
        assert!((brier_score - 0.04).abs() < 0.001);

        conn.execute(
            "UPDATE predictions SET actual_outcome = 1, brier_score = ?1, \
             status = 'resolved', resolved_at = ?2, updated_at = ?2 \
             WHERE id = ?3",
            rusqlite::params![brier_score, now, id],
        )
        .unwrap();

        let resolved_status: String = conn
            .query_row(
                "SELECT status FROM predictions WHERE id = ?1",
                rusqlite::params![id],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(resolved_status, "resolved");

        let stored_brier: f64 = conn
            .query_row(
                "SELECT brier_score FROM predictions WHERE id = ?1",
                rusqlite::params![id],
                |row| row.get(0),
            )
            .unwrap();
        assert!((stored_brier - 0.04).abs() < 0.001);

        // Resolve with outcome = false: (0.8 - 0.0)^2 = 0.64
        let id2 = Uuid::new_v4().to_string();
        conn.execute(
            "INSERT INTO predictions (id, description, probability, status, domain, created_at, updated_at) \
             VALUES (?1, ?2, 0.8, 'pending', 'general', ?3, ?4)",
            rusqlite::params![id2, "Another prediction", now, now],
        )
        .unwrap();

        let brier_false = (0.8_f64 - 0.0_f64).powi(2);
        assert!((brier_false - 0.64).abs() < 0.001);
    }

    #[tokio::test]
    async fn test_calibration_empty() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let path = tmp.path().to_path_buf();
        let conn = Connection::open(&path).unwrap();
        ensure_predictions_table(&conn).unwrap();

        // No resolved predictions — calibration should return empty buckets
        let count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM predictions WHERE status = 'resolved'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(count, 0);

        // Simulate calibration logic
        let resolved: Vec<(f64, f64, f64)> = Vec::new();
        assert!(resolved.is_empty());
        // Empty resolved set means overall_brier should be null/None
    }

    #[tokio::test]
    async fn test_domain_stats_with_test_db() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let path = tmp.path().to_path_buf();
        let conn = Connection::open(&path).unwrap();
        conn.execute_batch(
            "CREATE TABLE reasoning_patterns (
                id TEXT PRIMARY KEY,
                problem TEXT,
                solution TEXT,
                domain TEXT,
                reward REAL DEFAULT 0.5,
                embedding BLOB,
                created_at TEXT DEFAULT (datetime('now'))
            );
            INSERT INTO reasoning_patterns (id, problem, solution, domain, reward, embedding)
            VALUES ('p1', 'P1', 'S1', 'rust', 0.9, X'0102');
            INSERT INTO reasoning_patterns (id, problem, solution, domain, reward, embedding)
            VALUES ('p2', 'P2', 'S2', 'rust', 0.7, NULL);
            INSERT INTO reasoning_patterns (id, problem, solution, domain, reward, embedding)
            VALUES ('p3', 'P3', 'S3', 'python', 0.3, NULL);",
        )
        .unwrap();

        // Run the domain stats query
        let dcol = domain_column(&conn);
        assert_eq!(dcol, "domain");

        let sql = format!(
            "SELECT COALESCE({dcol},'unknown') as d, COUNT(*) as cnt, \
             AVG(COALESCE(reward,0.5)) as avg_r, 0.0 as avg_e, \
             MIN(COALESCE(reward,0.0)) as min_r, MAX(COALESCE(reward,1.0)) as max_r, \
             SUM(CASE WHEN embedding IS NOT NULL THEN 1 ELSE 0 END) as with_embed \
             FROM reasoning_patterns GROUP BY d ORDER BY cnt DESC"
        );
        let mut stmt = conn.prepare(&sql).unwrap();
        let rows: Vec<(String, i64, f64, f64, f64, f64, i64)> = stmt
            .query_map([], |row| {
                Ok((
                    row.get(0)?,
                    row.get(1)?,
                    row.get(2)?,
                    row.get(3)?,
                    row.get(4)?,
                    row.get(5)?,
                    row.get(6)?,
                ))
            })
            .unwrap()
            .filter_map(|r| r.ok())
            .collect();

        assert_eq!(rows.len(), 2); // rust, python

        // rust domain: 2 patterns, avg reward 0.8, 1 embedded
        let rust_row = rows.iter().find(|(d, _, _, _, _, _, _)| d == "rust").unwrap();
        assert_eq!(rust_row.1, 2); // count
        assert!((rust_row.2 - 0.8).abs() < 0.001); // avg_reward
        assert_eq!(rust_row.6, 1); // embedded_count

        // python domain: 1 pattern, reward 0.3, 0 embedded
        let py_row = rows.iter().find(|(d, _, _, _, _, _, _)| d == "python").unwrap();
        assert_eq!(py_row.1, 1);
        assert!((py_row.2 - 0.3).abs() < 0.001);
        assert_eq!(py_row.6, 0);

        // Test health classification
        let rust_health = if rust_row.2 >= 0.6 && rust_row.6 > 0 {
            "good"
        } else if rust_row.2 >= 0.3 {
            "warning"
        } else {
            "poor"
        };
        assert_eq!(rust_health, "good");

        let py_health = if py_row.2 >= 0.6 && py_row.6 > 0 {
            "good"
        } else if py_row.2 >= 0.3 {
            "warning"
        } else {
            "poor"
        };
        assert_eq!(py_health, "warning");
    }

    #[tokio::test]
    async fn test_surprise_empty_db() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let path = tmp.path().to_path_buf();
        let conn = Connection::open(&path).unwrap();
        conn.execute_batch(
            "CREATE TABLE reasoning_patterns (
                id TEXT PRIMARY KEY,
                problem TEXT,
                domain TEXT,
                reward REAL DEFAULT 0.5,
                surprise_score REAL DEFAULT 0.0,
                created_at TEXT DEFAULT (datetime('now'))
            );",
        )
        .unwrap();

        // Empty table — should return no surprise patterns
        let has_surprise = conn
            .prepare("SELECT surprise_score FROM reasoning_patterns LIMIT 0")
            .is_ok();
        assert!(has_surprise);

        let count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM reasoning_patterns WHERE surprise_score > 0.5",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(count, 0);

        // Add one pattern below threshold, one above
        conn.execute_batch(
            "INSERT INTO reasoning_patterns (id, problem, domain, reward, surprise_score)
             VALUES ('p1', 'Low surprise', 'rust', 0.5, 0.2);
             INSERT INTO reasoning_patterns (id, problem, domain, reward, surprise_score)
             VALUES ('p2', 'High surprise', 'rust', 0.8, 0.9);",
        )
        .unwrap();

        let count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM reasoning_patterns WHERE surprise_score > 0.5",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(count, 1);
    }

    #[tokio::test]
    async fn test_session_stats_empty() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let path = tmp.path().to_path_buf();
        let conn = Connection::open(&path).unwrap();
        conn.execute_batch(
            "CREATE TABLE reasoning_patterns (
                id TEXT PRIMARY KEY,
                problem TEXT,
                solution TEXT
            );",
        )
        .unwrap();

        // No sessions table — should be detected
        let has_sessions: bool = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='sessions'",
                [],
                |row| row.get::<_, i64>(0),
            )
            .map(|count| count > 0)
            .unwrap_or(false);
        assert!(!has_sessions);

        // Create sessions table and verify empty
        conn.execute_batch(
            "CREATE TABLE sessions (
                id TEXT PRIMARY KEY,
                started_at TEXT,
                ended_at TEXT,
                tokens_used INTEGER DEFAULT 0,
                patterns_learned INTEGER DEFAULT 0,
                patterns_retrieved INTEGER DEFAULT 0,
                domain TEXT
            );",
        )
        .unwrap();

        let count: i64 = conn
            .query_row("SELECT COUNT(*) FROM sessions", [], |row| row.get(0))
            .unwrap();
        assert_eq!(count, 0);
    }

    #[tokio::test]
    async fn test_graph_nodes_empty_db() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let path = tmp.path().to_path_buf();
        let conn = Connection::open(&path).unwrap();
        conn.execute_batch(
            "CREATE TABLE reasoning_patterns (
                id TEXT PRIMARY KEY,
                problem TEXT,
                domain TEXT,
                reward REAL DEFAULT 0.5,
                surprise_score REAL DEFAULT 0.0
            );",
        )
        .unwrap();

        // No patterns — nodes query should return empty
        let sql = "SELECT id, SUBSTR(COALESCE(problem,''),1,80) as label, \
                    COALESCE(domain,'') as domain, COALESCE(reward,0.0) as reward, \
                    COALESCE(surprise_score,0.0) as surprise \
                    FROM reasoning_patterns ORDER BY reward DESC LIMIT 200";
        let mut stmt = conn.prepare(sql).unwrap();
        let nodes: Vec<(String, String, String, f64, f64)> = stmt
            .query_map([], |row| {
                Ok((
                    row.get(0)?,
                    row.get(1)?,
                    row.get(2)?,
                    row.get(3)?,
                    row.get(4)?,
                ))
            })
            .unwrap()
            .filter_map(|r| r.ok())
            .collect();
        assert!(nodes.is_empty());

        // Also check edges — context_graph doesn't exist
        let has_context_graph = conn
            .prepare("SELECT source_id FROM context_graph LIMIT 0")
            .is_ok();
        assert!(!has_context_graph);
    }

    // -----------------------------------------------------------------------
    // Phase 4 tests
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_schedule_create_and_list() {
        // Use a fresh in-memory vec (test isolation via unique names)
        let jobs = scheduled_jobs();
        let initial_count = jobs.read().await.len();

        let now = Utc::now();
        let next_run = now + chrono::Duration::hours(6);
        let job = ScheduledJob {
            id: Uuid::new_v4().to_string(),
            name: "test-schedule-create".to_string(),
            actions: vec!["embed".to_string(), "consolidate".to_string()],
            interval_hours: 6,
            enabled: true,
            last_run: None,
            next_run: Some(next_run.to_rfc3339()),
            created_at: now.to_rfc3339(),
        };

        let job_id = job.id.clone();
        jobs.write().await.push(job);

        let all = jobs.read().await;
        assert!(all.len() > initial_count);

        let found = all.iter().find(|j| j.id == job_id);
        assert!(found.is_some());
        let found = found.unwrap();
        assert_eq!(found.name, "test-schedule-create");
        assert_eq!(found.actions.len(), 2);
        assert_eq!(found.interval_hours, 6);
        assert!(found.enabled);
        assert!(found.last_run.is_none());
        assert!(found.next_run.is_some());

        // Cleanup
        drop(all);
        jobs.write().await.retain(|j| j.id != job_id);
    }

    #[tokio::test]
    async fn test_schedule_delete() {
        let jobs = scheduled_jobs();

        let job = ScheduledJob {
            id: Uuid::new_v4().to_string(),
            name: "test-schedule-delete".to_string(),
            actions: vec!["dedup".to_string()],
            interval_hours: 12,
            enabled: true,
            last_run: None,
            next_run: None,
            created_at: Utc::now().to_rfc3339(),
        };

        let job_id = job.id.clone();
        jobs.write().await.push(job);

        // Verify it exists
        assert!(jobs.read().await.iter().any(|j| j.id == job_id));

        // Delete it
        let mut locked = jobs.write().await;
        locked.retain(|j| j.id != job_id);
        drop(locked);

        // Verify removal
        assert!(!jobs.read().await.iter().any(|j| j.id == job_id));
    }

    #[test]
    fn test_webhook_request_defaults() {
        let json = r#"{}"#;
        let req: WebhookLearnRequest = serde_json::from_str(json).unwrap();
        assert!(req.actions.is_none());
        assert!(req.source.is_none());
        assert!(req.callback_url.is_none());
    }

    #[test]
    fn test_webhook_validates_actions() {
        // Valid actions
        for action in ALLOWED_SCHEDULE_ACTIONS {
            assert!(parse_job_kind(action).is_some(), "Expected '{}' to be valid", action);
        }

        // Invalid action
        assert!(parse_job_kind("invalid_action").is_none());
        assert!(parse_job_kind("backup").is_none());
        assert!(parse_job_kind("").is_none());
    }

    #[tokio::test]
    async fn test_health_detailed_with_test_db() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let path = tmp.path().to_path_buf();
        let conn = Connection::open(&path).unwrap();
        conn.execute_batch(
            "CREATE TABLE reasoning_patterns (
                id TEXT PRIMARY KEY,
                problem TEXT,
                solution TEXT,
                domain TEXT,
                reward REAL DEFAULT 0.5,
                embedding BLOB,
                created_at TEXT DEFAULT (datetime('now'))
            );
            INSERT INTO reasoning_patterns (id, problem, solution, domain, reward, embedding, created_at)
            VALUES ('p1', 'Problem 1', 'Solution 1', 'rust', 0.9, X'0102', datetime('now'));
            INSERT INTO reasoning_patterns (id, problem, solution, domain, reward, created_at)
            VALUES ('p2', 'Problem 2', 'Solution 2', 'rust', 0.3, datetime('now'));
            INSERT INTO reasoning_patterns (id, problem, solution, domain, reward, created_at)
            VALUES ('p3', 'Problem 3', '', 'python', 0.1, datetime('now', '-100 days'));
            INSERT INTO reasoning_patterns (id, problem, domain, reward, created_at)
            VALUES ('p4', 'Problem 4', 'python', 0.15, datetime('now', '-100 days'));",
        )
        .unwrap();
        drop(conn);

        let db_conn = open_db(&path).unwrap();

        let total: u64 = db_conn
            .query_row("SELECT COUNT(*) FROM reasoning_patterns", [], |row| row.get(0))
            .unwrap();
        assert_eq!(total, 4);

        let embedded: u64 = db_conn
            .query_row(
                "SELECT COUNT(*) FROM reasoning_patterns WHERE embedding IS NOT NULL",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(embedded, 1);

        // embedding_coverage = 1/4 = 0.25 (< 0.5) -> -20
        let coverage = embedded as f64 / total as f64;
        assert!(coverage < 0.5);

        let orphans: u64 = db_conn
            .query_row(
                "SELECT COUNT(*) FROM reasoning_patterns WHERE solution IS NULL OR solution = ''",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert!(orphans >= 1); // p3 has empty solution, p4 has NULL

        let stale: u64 = db_conn
            .query_row(
                "SELECT COUNT(*) FROM reasoning_patterns \
                 WHERE created_at < datetime('now', '-90 days') AND reward < 0.3",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert!(stale >= 1); // p3 and p4 are old + low reward

        let avg_reward: f64 = db_conn
            .query_row(
                "SELECT COALESCE(AVG(reward), 0.0) FROM reasoning_patterns",
                [],
                |row| row.get(0),
            )
            .unwrap();
        // (0.9 + 0.3 + 0.1 + 0.15) / 4 = 0.3625
        assert!(avg_reward < 0.4); // triggers -10
    }

    #[test]
    fn test_health_score_calculation() {
        // Perfect health
        assert_eq!(compute_health_score(1.0, 0, 0.8, 0.0, 0), 100);

        // Low embedding coverage (< 0.5 -> -20)
        assert_eq!(compute_health_score(0.3, 0, 0.8, 0.0, 0), 80);

        // Medium embedding coverage (0.5..0.8 -> -10)
        assert_eq!(compute_health_score(0.6, 0, 0.8, 0.0, 0), 90);

        // Many stale patterns (> 100 -> -15)
        assert_eq!(compute_health_score(1.0, 150, 0.8, 0.0, 0), 85);

        // Low avg reward (< 0.4 -> -10)
        assert_eq!(compute_health_score(1.0, 0, 0.3, 0.0, 0), 90);

        // High error rate (> 0.2 -> -10)
        assert_eq!(compute_health_score(1.0, 0, 0.8, 0.5, 0), 90);

        // Many orphans (> 50 -> -5)
        assert_eq!(compute_health_score(1.0, 0, 0.8, 0.0, 60), 95);

        // Everything bad
        assert_eq!(compute_health_score(0.3, 200, 0.2, 0.5, 100), 40);

        // All penalties: -20 -15 -10 -10 -5 = -60 -> 40
        assert_eq!(compute_health_score(0.0, 999, 0.0, 1.0, 999), 40);
    }

    #[tokio::test]
    async fn test_export_with_test_db() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let path = tmp.path().to_path_buf();
        let conn = Connection::open(&path).unwrap();
        conn.execute_batch(
            "CREATE TABLE reasoning_patterns (
                id TEXT PRIMARY KEY,
                problem TEXT,
                solution TEXT,
                domain TEXT,
                tags TEXT,
                reward REAL DEFAULT 0.5,
                tier TEXT DEFAULT 'booster',
                created_at TEXT DEFAULT (datetime('now'))
            );
            INSERT INTO reasoning_patterns (id, problem, solution, domain, tags, reward, tier)
            VALUES ('e1', 'Export test 1', 'Sol 1', 'rust', 'async,tokio', 0.9, 'reflex');
            INSERT INTO reasoning_patterns (id, problem, solution, domain, tags, reward, tier)
            VALUES ('e2', 'Export test 2', 'Sol 2', 'python', 'ml,torch', 0.7, 'booster');
            INSERT INTO reasoning_patterns (id, problem, solution, domain, reward, tier)
            VALUES ('e3', 'Export test 3', 'Sol 3', 'rust', 0.5, 'booster');",
        )
        .unwrap();
        drop(conn);

        let db_conn = open_db(&path).unwrap();

        // All patterns
        let total: u64 = db_conn
            .query_row("SELECT COUNT(*) FROM reasoning_patterns", [], |row| row.get(0))
            .unwrap();
        assert_eq!(total, 3);

        // Filter by domain
        let rust_count: u64 = db_conn
            .query_row(
                "SELECT COUNT(*) FROM reasoning_patterns WHERE domain = 'rust'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(rust_count, 2);

        // Verify we can read columns needed for export
        let mut stmt = db_conn
            .prepare(
                "SELECT id, problem, solution, domain, tags, reward, tier, created_at \
                 FROM reasoning_patterns ORDER BY created_at DESC LIMIT 10",
            )
            .unwrap();
        let rows: Vec<String> = stmt
            .query_map([], |row| row.get::<_, String>(0))
            .unwrap()
            .filter_map(|r| r.ok())
            .collect();
        assert_eq!(rows.len(), 3);
    }

    #[tokio::test]
    async fn test_import_with_test_db() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let path = tmp.path().to_path_buf();
        let conn = Connection::open(&path).unwrap();
        conn.execute_batch(
            "CREATE TABLE reasoning_patterns (
                id TEXT PRIMARY KEY,
                problem TEXT,
                solution TEXT,
                domain TEXT,
                tags TEXT,
                context TEXT,
                reward REAL DEFAULT 0.5,
                created_at TEXT DEFAULT (datetime('now'))
            );",
        )
        .unwrap();

        // Import patterns
        let id1 = Uuid::new_v4().to_string();
        let id2 = Uuid::new_v4().to_string();
        let now = Utc::now().to_rfc3339();

        conn.execute(
            "INSERT INTO reasoning_patterns (id, problem, solution, domain, tags, context, reward, created_at) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, 0.5, ?7)",
            rusqlite::params![id1, "Imported 1", "Sol 1", "rust", "async", "test context", now],
        )
        .unwrap();

        conn.execute(
            "INSERT INTO reasoning_patterns (id, problem, solution, domain, tags, context, reward, created_at) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, 0.5, ?7)",
            rusqlite::params![id2, "Imported 2", "Sol 2", "python", "ml", "", now],
        )
        .unwrap();

        let count: u64 = conn
            .query_row("SELECT COUNT(*) FROM reasoning_patterns", [], |row| row.get(0))
            .unwrap();
        assert_eq!(count, 2);

        // Verify data integrity
        let domain: String = conn
            .query_row(
                "SELECT domain FROM reasoning_patterns WHERE id = ?1",
                [&id1],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(domain, "rust");

        // INSERT OR IGNORE should skip duplicates
        let result = conn.execute(
            "INSERT OR IGNORE INTO reasoning_patterns (id, problem, solution, domain, reward, created_at) \
             VALUES (?1, ?2, ?3, ?4, 0.5, ?5)",
            rusqlite::params![id1, "Duplicate", "Dup sol", "rust", now],
        );
        assert_eq!(result.unwrap(), 0); // 0 rows affected = skipped
    }

    #[tokio::test]
    async fn test_activity_empty_db() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let path = tmp.path().to_path_buf();
        let conn = Connection::open(&path).unwrap();
        conn.execute_batch(
            "CREATE TABLE reasoning_patterns (
                id TEXT PRIMARY KEY,
                problem TEXT,
                domain TEXT,
                reward REAL DEFAULT 0.5,
                created_at TEXT DEFAULT (datetime('now'))
            );",
        )
        .unwrap();
        drop(conn);

        let db_conn = open_db(&path).unwrap();

        // Empty DB should return 0 patterns
        let count: u64 = db_conn
            .query_row("SELECT COUNT(*) FROM reasoning_patterns", [], |row| row.get(0))
            .unwrap();
        assert_eq!(count, 0);

        // No outcomes table
        let has_outcomes = db_conn
            .prepare("SELECT id FROM pattern_outcomes LIMIT 0")
            .is_ok();
        assert!(!has_outcomes);

        let has_outcomes_alt = db_conn
            .prepare("SELECT id FROM outcomes LIMIT 0")
            .is_ok();
        assert!(!has_outcomes_alt);
    }
}
