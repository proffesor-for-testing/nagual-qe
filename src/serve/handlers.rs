//! REST API handlers for the Nagual dashboard.
//!
//! Provides JSON endpoints for querying pattern data, domain distributions,
//! tier breakdowns, pulse activity, and knowledge graph data from SQLite.

use std::collections::HashMap;

use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::response::Json;
use rusqlite::{Connection, OpenFlags};
use serde::{Deserialize, Serialize};

use super::auth::RequireAuth;
use super::AppState;

// ---------------------------------------------------------------------------
// Schema detection helpers
// ---------------------------------------------------------------------------

/// Detect which column name is used for "domain" in reasoning_patterns.
/// Older schemas use `category`, newer schemas use `domain`.
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

/// Check if the `tier` column exists in reasoning_patterns.
fn has_tier_column(conn: &Connection) -> bool {
    conn.prepare("SELECT tier FROM reasoning_patterns LIMIT 0")
        .is_ok()
}

// ---------------------------------------------------------------------------
// Response types
// ---------------------------------------------------------------------------

/// Overall system status response.
#[derive(Serialize)]
pub struct StatusResponse {
    pub pattern_count: u64,
    pub domain_count: u64,
    pub avg_reward: f64,
    pub db_size_bytes: u64,
    pub oldest_pattern: Option<String>,
    pub newest_pattern: Option<String>,
}

/// A single pattern entry.
#[derive(Serialize)]
pub struct PatternEntry {
    pub id: String,
    pub problem: String,
    pub solution: String,
    pub domain: String,
    pub reward: f64,
    pub tier: String,
    pub created_at: String,
}

/// Query parameters for patterns endpoint.
#[derive(Deserialize)]
pub struct PatternsQuery {
    pub limit: Option<u32>,
    pub offset: Option<u32>,
    pub domain: Option<String>,
    pub tier: Option<String>,
    pub search: Option<String>,
    pub sort_by: Option<String>,
    pub sort_order: Option<String>,
}

/// Paginated patterns response.
#[derive(Serialize)]
pub struct PaginatedPatterns {
    pub patterns: Vec<PatternEntry>,
    pub total: u64,
    pub limit: u32,
    pub offset: u32,
}

/// Domain distribution entry.
#[derive(Serialize)]
pub struct DomainEntry {
    pub domain: String,
    pub count: u64,
}

/// Tier distribution response.
#[derive(Serialize)]
pub struct TierResponse {
    pub booster: u64,
    pub crystal: u64,
    pub reflex: u64,
    pub unclassified: u64,
}

/// Daily pulse entry (date -> count).
#[derive(Serialize)]
pub struct PulseEntry {
    pub date: String,
    pub count: u64,
}

/// Graph node for visualization.
#[derive(Serialize)]
pub struct GraphNode {
    pub id: String,
    pub label: String,
    pub domain: String,
    pub reward: f64,
}

/// Graph edge for visualization.
#[derive(Serialize)]
pub struct GraphEdge {
    pub source: String,
    pub target: String,
    pub weight: f64,
}

/// Knowledge graph response.
#[derive(Serialize)]
pub struct GraphResponse {
    pub nodes: Vec<GraphNode>,
    pub edges: Vec<GraphEdge>,
}

// ---------------------------------------------------------------------------
// Helper: open read-only SQLite connection
// ---------------------------------------------------------------------------

fn open_db(state: &AppState) -> Result<Connection, (StatusCode, String)> {
    Connection::open_with_flags(&state.db_path, OpenFlags::SQLITE_OPEN_READ_ONLY).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Database error: {}", e),
        )
    })
}

// ---------------------------------------------------------------------------
// GET /api/status
// ---------------------------------------------------------------------------

pub async fn api_status(
    State(state): State<AppState>,
    _auth: RequireAuth,
) -> Result<Json<StatusResponse>, (StatusCode, String)> {
    let conn = open_db(&state)?;

    let pattern_count: u64 = conn
        .query_row(
            "SELECT COUNT(*) FROM reasoning_patterns",
            [],
            |row| row.get(0),
        )
        .unwrap_or(0);

    let dcol = domain_column(&conn);
    let domain_count: u64 = conn
        .query_row(
            &format!("SELECT COUNT(DISTINCT {}) FROM reasoning_patterns", dcol),
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

    let oldest_pattern: Option<String> = conn
        .query_row(
            "SELECT MIN(created_at) FROM reasoning_patterns",
            [],
            |row| row.get(0),
        )
        .unwrap_or(None);

    let newest_pattern: Option<String> = conn
        .query_row(
            "SELECT MAX(created_at) FROM reasoning_patterns",
            [],
            |row| row.get(0),
        )
        .unwrap_or(None);

    // Get file size
    let db_size_bytes = std::fs::metadata(&state.db_path)
        .map(|m| m.len())
        .unwrap_or(0);

    Ok(Json(StatusResponse {
        pattern_count,
        domain_count,
        avg_reward,
        db_size_bytes,
        oldest_pattern,
        newest_pattern,
    }))
}

// ---------------------------------------------------------------------------
// GET /api/patterns
// ---------------------------------------------------------------------------

pub async fn api_patterns(
    State(state): State<AppState>,
    _auth: RequireAuth,
    Query(params): Query<PatternsQuery>,
) -> Result<Json<PaginatedPatterns>, (StatusCode, String)> {
    let conn = open_db(&state)?;

    let limit = params.limit.unwrap_or(50).min(500);
    let offset = params.offset.unwrap_or(0);

    let dcol = domain_column(&conn);
    let tier_expr = if has_tier_column(&conn) {
        "COALESCE(tier, 'booster')"
    } else {
        "'booster'"
    };

    // Build WHERE conditions
    let mut conditions: Vec<String> = vec![];
    let mut bind_values: Vec<String> = vec![];

    if let Some(ref domain) = params.domain {
        if !domain.is_empty() {
            conditions.push(format!("{dcol} = ?{}", bind_values.len() + 1));
            bind_values.push(domain.clone());
        }
    }

    if let Some(ref tier) = params.tier {
        if !tier.is_empty() && has_tier_column(&conn) {
            conditions.push(format!("LOWER(tier) = ?{}", bind_values.len() + 1));
            bind_values.push(tier.to_lowercase());
        }
    }

    if let Some(ref search) = params.search {
        if !search.is_empty() {
            let idx = bind_values.len() + 1;
            conditions.push(format!(
                "(problem LIKE ?{idx} OR solution LIKE ?{idx} OR {dcol} LIKE ?{idx})"
            ));
            bind_values.push(format!("%{}%", search));
        }
    }

    let where_clause = if conditions.is_empty() {
        String::new()
    } else {
        format!("WHERE {}", conditions.join(" AND "))
    };

    // Sorting
    let sort_col = match params.sort_by.as_deref() {
        Some("reward") => "reward",
        Some("tier") => "tier",
        Some("domain") | Some("category") => dcol,
        Some("problem") => "problem",
        _ => "created_at",
    };
    let sort_dir = match params.sort_order.as_deref() {
        Some("asc") => "ASC",
        _ => "DESC",
    };

    // Count total matching
    let count_sql = format!("SELECT COUNT(*) FROM reasoning_patterns {where_clause}");
    let total: u64 = {
        let mut stmt = conn.prepare(&count_sql).map_err(|e| {
            (StatusCode::INTERNAL_SERVER_ERROR, format!("Count error: {}", e))
        })?;
        match bind_values.len() {
            0 => stmt.query_row([], |r| r.get(0)),
            1 => stmt.query_row([&bind_values[0]], |r| r.get(0)),
            2 => stmt.query_row([&bind_values[0], &bind_values[1]], |r| r.get(0)),
            _ => stmt.query_row([&bind_values[0], &bind_values[1], &bind_values[2]], |r| r.get(0)),
        }
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Count error: {}", e)))?
    };

    // Fetch patterns
    let sql = format!(
        "SELECT id, COALESCE(problem, ''), COALESCE(solution, ''), \
         COALESCE({dcol}, ''), COALESCE(reward, 0.0), {tier_expr}, \
         COALESCE(created_at, '') \
         FROM reasoning_patterns {where_clause} \
         ORDER BY {sort_col} {sort_dir} LIMIT {limit} OFFSET {offset}"
    );

    let mut stmt = conn.prepare(&sql).map_err(|e| {
        (StatusCode::INTERNAL_SERVER_ERROR, format!("Query error: {}", e))
    })?;

    fn extract_pattern(row: &rusqlite::Row) -> rusqlite::Result<PatternEntry> {
        Ok(PatternEntry {
            id: row.get(0)?,
            problem: row.get(1)?,
            solution: row.get(2)?,
            domain: row.get(3)?,
            reward: row.get(4)?,
            tier: row.get(5)?,
            created_at: row.get(6)?,
        })
    }

    let patterns: Vec<PatternEntry> = match bind_values.len() {
        0 => {
            let rows = stmt.query_map([], extract_pattern)
                .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Query error: {}", e)))?;
            rows.filter_map(|r| r.ok()).collect()
        }
        1 => {
            let rows = stmt.query_map([&bind_values[0]], extract_pattern)
                .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Query error: {}", e)))?;
            rows.filter_map(|r| r.ok()).collect()
        }
        2 => {
            let rows = stmt.query_map([&bind_values[0], &bind_values[1]], extract_pattern)
                .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Query error: {}", e)))?;
            rows.filter_map(|r| r.ok()).collect()
        }
        _ => {
            let rows = stmt.query_map([&bind_values[0], &bind_values[1], &bind_values[2]], extract_pattern)
                .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Query error: {}", e)))?;
            rows.filter_map(|r| r.ok()).collect()
        }
    };

    Ok(Json(PaginatedPatterns {
        patterns,
        total,
        limit,
        offset,
    }))
}

// ---------------------------------------------------------------------------
// GET /api/domains
// ---------------------------------------------------------------------------

pub async fn api_domains(
    State(state): State<AppState>,
    _auth: RequireAuth,
) -> Result<Json<Vec<DomainEntry>>, (StatusCode, String)> {
    let conn = open_db(&state)?;

    let dcol = domain_column(&conn);
    let mut stmt = conn
        .prepare(&format!(
            "SELECT COALESCE({dcol}, 'unknown') as d, COUNT(*) as cnt \
             FROM reasoning_patterns GROUP BY d ORDER BY cnt DESC LIMIT 50",
        ))
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Query error: {}", e),
            )
        })?;

    let rows = stmt
        .query_map([], |row| {
            Ok(DomainEntry {
                domain: row.get(0)?,
                count: row.get(1)?,
            })
        })
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Query error: {}", e),
            )
        })?;

    let domains: Vec<DomainEntry> = rows.filter_map(|r| r.ok()).collect();
    Ok(Json(domains))
}

// ---------------------------------------------------------------------------
// GET /api/tiers
// ---------------------------------------------------------------------------

pub async fn api_tiers(
    State(state): State<AppState>,
    _auth: RequireAuth,
) -> Result<Json<TierResponse>, (StatusCode, String)> {
    let conn = open_db(&state)?;

    if !has_tier_column(&conn) {
        // No tier column — all patterns are effectively "booster"
        let total: u64 = conn
            .query_row("SELECT COUNT(*) FROM reasoning_patterns", [], |row| row.get(0))
            .unwrap_or(0);
        return Ok(Json(TierResponse {
            booster: total,
            crystal: 0,
            reflex: 0,
            unclassified: 0,
        }));
    }

    let mut tier_counts: HashMap<String, u64> = HashMap::new();

    let mut stmt = conn
        .prepare(
            "SELECT COALESCE(LOWER(tier), ''), COUNT(*) \
             FROM reasoning_patterns GROUP BY COALESCE(LOWER(tier), '')",
        )
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Query error: {}", e),
            )
        })?;

    let rows = stmt
        .query_map([], |row| {
            let tier: String = row.get(0)?;
            let count: u64 = row.get(1)?;
            Ok((tier, count))
        })
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Query error: {}", e),
            )
        })?;

    for row in rows {
        if let Ok((tier, count)) = row {
            tier_counts.insert(tier, count);
        }
    }

    Ok(Json(TierResponse {
        booster: *tier_counts.get("booster").unwrap_or(&0),
        crystal: *tier_counts.get("crystal").unwrap_or(&0),
        reflex: *tier_counts.get("reflex").unwrap_or(&0),
        unclassified: *tier_counts.get("").unwrap_or(&0)
            + *tier_counts.get("unknown").unwrap_or(&0),
    }))
}

// ---------------------------------------------------------------------------
// GET /api/pulse
// ---------------------------------------------------------------------------

pub async fn api_pulse(
    State(state): State<AppState>,
    _auth: RequireAuth,
) -> Result<Json<Vec<PulseEntry>>, (StatusCode, String)> {
    let conn = open_db(&state)?;

    let mut stmt = conn
        .prepare(
            "SELECT DATE(created_at) as day, COUNT(*) as cnt \
             FROM reasoning_patterns \
             WHERE created_at >= DATE('now', '-364 days') \
             GROUP BY DATE(created_at) ORDER BY day",
        )
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Query error: {}", e),
            )
        })?;

    let rows = stmt
        .query_map([], |row| {
            Ok(PulseEntry {
                date: row.get(0)?,
                count: row.get(1)?,
            })
        })
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Query error: {}", e),
            )
        })?;

    let pulse: Vec<PulseEntry> = rows.filter_map(|r| r.ok()).collect();
    Ok(Json(pulse))
}

// ---------------------------------------------------------------------------
// GET /api/graph
// ---------------------------------------------------------------------------

pub async fn api_graph(
    State(state): State<AppState>,
    _auth: RequireAuth,
) -> Result<Json<GraphResponse>, (StatusCode, String)> {
    let conn = open_db(&state)?;

    // Get top patterns as nodes (limit to 100 for visualization)
    let dcol = domain_column(&conn);
    let mut node_stmt = conn
        .prepare(&format!(
            "SELECT id, COALESCE(problem, ''), COALESCE({dcol}, ''), COALESCE(reward, 0.0) \
             FROM reasoning_patterns ORDER BY reward DESC LIMIT 100",
        ))
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Query error: {}", e),
            )
        })?;

    let node_rows = node_stmt
        .query_map([], |row| {
            Ok(GraphNode {
                id: row.get(0)?,
                label: row.get::<_, String>(1)?
                    .chars()
                    .take(60)
                    .collect(),
                domain: row.get(2)?,
                reward: row.get(3)?,
            })
        })
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Query error: {}", e),
            )
        })?;

    let nodes: Vec<GraphNode> = node_rows.filter_map(|r| r.ok()).collect();

    // Try context_graph (actual SQLite edge table)
    let mut edges: Vec<GraphEdge> = match conn.prepare(
        "SELECT source_id, target_id, COALESCE(strength, 0.5) \
         FROM context_graph LIMIT 500",
    ) {
        Ok(mut edge_stmt) => match edge_stmt.query_map([], |row| {
            Ok(GraphEdge {
                source: row.get(0)?,
                target: row.get(1)?,
                weight: row.get(2)?,
            })
        }) {
            Ok(rows) => rows.filter_map(|r| r.ok()).collect(),
            Err(_) => Vec::new(),
        },
        Err(_) => Vec::new(),
    };

    // If no stored edges, generate domain-based edges dynamically:
    // patterns sharing the same domain are connected.
    if edges.is_empty() {
        let node_ids: Vec<&str> = nodes.iter().map(|n| n.id.as_str()).collect();
        let mut domain_groups: HashMap<String, Vec<usize>> = HashMap::new();
        for (i, n) in nodes.iter().enumerate() {
            if !n.domain.is_empty() {
                domain_groups.entry(n.domain.clone()).or_default().push(i);
            }
        }
        for (_domain, members) in &domain_groups {
            if members.len() < 2 || members.len() > 30 { continue; }
            // Connect each node to the next within the domain (chain)
            for w in members.windows(2) {
                edges.push(GraphEdge {
                    source: node_ids[w[0]].to_string(),
                    target: node_ids[w[1]].to_string(),
                    weight: 0.5,
                });
            }
        }
    }

    Ok(Json(GraphResponse { nodes, edges }))
}

// ---------------------------------------------------------------------------
// 3D Graph types and handler
// ---------------------------------------------------------------------------

/// 3D graph node for force-graph visualization.
#[derive(Serialize)]
pub struct Graph3DNode {
    pub id: String,
    pub label: String,
    pub domain: String,
    pub tier: String,
    pub reward: f64,
    pub reuse_count: u32,
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub zone: String,
    pub size: f64,
    pub color: String,
}

/// 3D graph edge.
#[derive(Serialize)]
pub struct Graph3DEdge {
    pub source: String,
    pub target: String,
    pub weight: f64,
    pub edge_type: String,
}

/// 3D graph response.
#[derive(Serialize)]
pub struct Graph3DResponse {
    pub nodes: Vec<Graph3DNode>,
    pub edges: Vec<Graph3DEdge>,
    pub stats: Graph3DStats,
}

/// Summary stats for the 3D graph.
#[derive(Serialize)]
pub struct Graph3DStats {
    pub total_nodes: usize,
    pub total_edges: usize,
    pub domains: usize,
    pub core_count: usize,
    pub future_count: usize,
    pub history_count: usize,
}

/// GET /api/graph/3d - 3D knowledge graph data for force-graph visualization.
///
/// Assigns each pattern to one of three zones:
///   - **Core**: `reward >= 0.7 AND reuse_count >= 5` (active, proven patterns)
///   - **Future**: created in last 7 days AND `reuse_count < 3` (new, unproven)
///   - **History**: everything else
///
/// Queries multiple edge sources: context_graph, profdag_edges, wormholes,
/// and generates domain-based edges as fallback.
pub async fn api_graph_3d(
    State(state): State<AppState>,
    _auth: RequireAuth,
) -> Result<Json<Graph3DResponse>, (StatusCode, String)> {
    let conn = open_db(&state)?;

    let dcol = domain_column(&conn);
    let has_tier = has_tier_column(&conn);

    let has_reuse_count = conn
        .prepare("SELECT reuse_count FROM reasoning_patterns LIMIT 0")
        .is_ok();

    let tier_expr = if has_tier { "COALESCE(tier, 'booster')" } else { "'booster'" };
    let reuse_expr = if has_reuse_count { "COALESCE(reuse_count, 0)" } else { "0" };

    // Fetch up to 2000 patterns (enough for force-graph, not overwhelming)
    let sql = format!(
        "SELECT id, COALESCE(problem, ''), COALESCE({dcol}, 'unknown'), \
         {tier_expr}, COALESCE(reward, 0.0), \
         {reuse_expr}, COALESCE(created_at, '') \
         FROM reasoning_patterns ORDER BY reward DESC LIMIT 2000"
    );

    let mut stmt = conn.prepare(&sql).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Query error: {}", e),
        )
    })?;

    let rows = stmt
        .query_map([], |row| {
            let id: String = row.get(0)?;
            let problem: String = row.get(1)?;
            let domain: String = row.get(2)?;
            let tier: String = row.get(3)?;
            let reward: f64 = row.get(4)?;
            let reuse_count: u32 = row.get::<_, i64>(5).unwrap_or(0) as u32;
            let created_at: String = row.get(6)?;
            Ok((id, problem, domain, tier, reward, reuse_count, created_at))
        })
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Query error: {}", e),
            )
        })?;

    let mut raw: Vec<(String, String, String, String, f64, u32, String)> = Vec::new();
    for r in rows {
        if let Ok(v) = r {
            raw.push(v);
        }
    }

    // Determine 7-day-ago cutoff (ISO 8601 string comparison works)
    let cutoff_7d = {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let seven_days = 7 * 24 * 3600;
        let cutoff = now.saturating_sub(seven_days);
        let secs_per_day = 86400u64;
        let days_since_epoch = cutoff / secs_per_day;
        let (y, m, d) = epoch_days_to_ymd(days_since_epoch);
        format!("{:04}-{:02}-{:02}", y, m, d)
    };

    // Simple hash for deterministic initial positions (force-graph will relayout)
    fn domain_hash(s: &str) -> f64 {
        let mut h: u64 = 5381;
        for b in s.bytes() {
            h = h.wrapping_mul(33).wrapping_add(b as u64);
        }
        ((h % 10000) as f64 / 5000.0) - 1.0
    }

    fn id_hash(s: &str) -> f64 {
        let mut h: u64 = 0x517cc1b727220a95;
        for b in s.bytes() {
            h = h.wrapping_mul(6364136223846793005).wrapping_add(b as u64);
        }
        ((h % 10000) as f64 / 5000.0) - 1.0
    }

    // Build node set for edge filtering
    let node_ids: std::collections::HashSet<String> = raw.iter().map(|r| r.0.clone()).collect();

    // Zone color mapping
    fn zone_color(zone: &str) -> &'static str {
        match zone {
            "core" => "#4ecca3",
            "future" => "#4488ff",
            _ => "#888888",
        }
    }

    let mut core_count = 0usize;
    let mut future_count = 0usize;
    let mut history_count = 0usize;

    let nodes: Vec<Graph3DNode> = raw
        .iter()
        .map(|(id, problem, domain, tier, reward, reuse_count, created_at)| {
            let zone = if *reward >= 0.7 && *reuse_count >= 5 {
                "core"
            } else if created_at.as_str() >= cutoff_7d.as_str() && *reuse_count < 3 {
                "future"
            } else {
                "history"
            };

            match zone {
                "core" => core_count += 1,
                "future" => future_count += 1,
                _ => history_count += 1,
            }

            // Initial positions hint for force-graph (it will re-layout)
            let y_base = match zone {
                "history" => -0.65,
                "core" => 0.0,
                "future" => 0.65,
                _ => 0.0,
            };
            let jitter = id_hash(id) * 0.3;
            let y = (y_base + jitter * 0.15) * 100.0;

            let dh = domain_hash(domain);
            let ih = id_hash(id);
            let x = (dh * 0.6 + ih * 0.4) * 100.0;
            let z = domain_hash(&format!("{}{}", domain, id)) * 70.0;

            let rc = (*reuse_count).max(1) as f64;
            let size = reward * (1.0 + rc.ln()) * 4.0 + 2.0;

            Graph3DNode {
                id: id.clone(),
                label: problem.chars().take(80).collect(),
                domain: domain.clone(),
                tier: tier.clone(),
                reward: *reward,
                reuse_count: *reuse_count,
                x,
                y,
                z,
                zone: zone.to_string(),
                size,
                color: zone_color(zone).to_string(),
            }
        })
        .collect();

    let unique_domains: std::collections::HashSet<&str> =
        raw.iter().map(|r| r.2.as_str()).collect();

    // --- Collect edges from all sources ---
    let mut edges: Vec<Graph3DEdge> = Vec::new();

    // 1. context_graph edges
    if let Ok(mut edge_stmt) = conn.prepare(
        "SELECT source_id, target_id, COALESCE(strength, 0.5), COALESCE(edge_type, 'related_to') \
         FROM context_graph LIMIT 2000",
    ) {
        if let Ok(rows) = edge_stmt.query_map([], |row| {
            Ok(Graph3DEdge {
                source: row.get(0)?,
                target: row.get(1)?,
                weight: row.get(2)?,
                edge_type: row.get(3)?,
            })
        }) {
            edges.extend(rows.filter_map(|r| r.ok()).filter(|e| {
                node_ids.contains(&e.source) && node_ids.contains(&e.target)
            }));
        }
    }

    // 2. profdag_edges (leads_to, similar_to, derived_from, wormhole, temporal_link)
    if let Ok(mut pd_stmt) = conn.prepare(
        "SELECT source_id, target_id, COALESCE(weight, 0.5), edge_type \
         FROM profdag_edges ORDER BY weight DESC LIMIT 2000",
    ) {
        if let Ok(rows) = pd_stmt.query_map([], |row| {
            Ok(Graph3DEdge {
                source: row.get(0)?,
                target: row.get(1)?,
                weight: row.get(2)?,
                edge_type: row.get(3)?,
            })
        }) {
            // profdag_edges reference profdag_nodes, not reasoning_patterns directly.
            // Include them if both endpoints are in our node set.
            edges.extend(rows.filter_map(|r| r.ok()).filter(|e| {
                node_ids.contains(&e.source) && node_ids.contains(&e.target)
            }));
        }
    }

    // 3. Wormhole edges from wormholes table
    if let Ok(mut wh_stmt) = conn.prepare(
        "SELECT source_id, target_id, COALESCE(strength, 0.8) \
         FROM wormholes WHERE is_active = 1 LIMIT 500",
    ) {
        if let Ok(wh_rows) = wh_stmt.query_map([], |row| {
            Ok(Graph3DEdge {
                source: row.get(0)?,
                target: row.get(1)?,
                weight: row.get(2)?,
                edge_type: "wormhole".to_string(),
            })
        }) {
            edges.extend(wh_rows.filter_map(|r| r.ok()).filter(|e| {
                node_ids.contains(&e.source) && node_ids.contains(&e.target)
            }));
        }
    }

    // 4. Always generate domain-based edges (they show clustering even with real edges)
    {
        let mut domain_groups: HashMap<String, Vec<usize>> = HashMap::new();
        for (i, r) in raw.iter().enumerate() {
            let domain = &r.2;
            if !domain.is_empty() && domain != "unknown" {
                domain_groups.entry(domain.clone()).or_default().push(i);
            }
        }
        for (_domain, members) in &domain_groups {
            if members.len() < 2 { continue; }
            // For large domains, only connect nearest neighbors (max 30 edges per domain)
            let max_edges = 30.min(members.len() - 1);
            for w in members.windows(2).take(max_edges) {
                edges.push(Graph3DEdge {
                    source: raw[w[0]].0.clone(),
                    target: raw[w[1]].0.clone(),
                    weight: 0.3,
                    edge_type: "domain".to_string(),
                });
            }
        }
    }

    let stats = Graph3DStats {
        total_nodes: nodes.len(),
        total_edges: edges.len(),
        domains: unique_domains.len(),
        core_count,
        future_count,
        history_count,
    };

    Ok(Json(Graph3DResponse { nodes, edges, stats }))
}

/// Convert days since Unix epoch to (year, month, day).
fn epoch_days_to_ymd(days: u64) -> (u64, u64, u64) {
    // Algorithm from http://howardhinnant.github.io/date_algorithms.html
    let z = days + 719468;
    let era = z / 146097;
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    (y, m, d)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::serve::auth::AuthIdentity;
    use std::path::PathBuf;

    fn test_auth() -> RequireAuth {
        RequireAuth(AuthIdentity::LocalOnly)
    }

    fn test_state(db_path: PathBuf) -> AppState {
        AppState {
            db_path,
            event_bus: std::sync::Arc::new(crate::events::EventBus::new()),
            storage: None,
            auth_token: None,
            key_store: None,
            user_store: None,
            session_secret: vec![0u8; 32],
            login_required: false,
        }
    }

    fn create_test_db() -> (tempfile::NamedTempFile, PathBuf) {
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
                tier TEXT DEFAULT '',
                created_at TEXT DEFAULT (datetime('now'))
            );
            INSERT INTO reasoning_patterns (id, problem, solution, domain, reward, tier)
            VALUES ('p1', 'Test problem', 'Test solution', 'rust', 0.8, 'booster');
            INSERT INTO reasoning_patterns (id, problem, solution, domain, reward, tier)
            VALUES ('p2', 'Another problem', 'Another solution', 'python', 0.6, 'crystal');
            INSERT INTO reasoning_patterns (id, problem, solution, domain, reward, tier)
            VALUES ('p3', 'Third problem', 'Third solution', 'rust', 0.9, 'reflex');",
        )
        .unwrap();
        (tmp, path)
    }

    #[test]
    fn test_open_db_success() {
        let (_tmp, path) = create_test_db();
        let state = test_state(path);
        assert!(open_db(&state).is_ok());
    }

    #[test]
    fn test_open_db_nonexistent() {
        let state = test_state(PathBuf::from("/nonexistent/path/test.db"));
        assert!(open_db(&state).is_err());
    }

    #[tokio::test]
    async fn test_api_status_handler() {
        let (_tmp, path) = create_test_db();
        let state = test_state(path);
        let result = api_status(State(state), test_auth()).await;
        assert!(result.is_ok());
        let json = result.unwrap();
        assert_eq!(json.pattern_count, 3);
        assert_eq!(json.domain_count, 2);
    }

    #[tokio::test]
    async fn test_api_domains_handler() {
        let (_tmp, path) = create_test_db();
        let state = test_state(path);
        let result = api_domains(State(state), test_auth()).await;
        assert!(result.is_ok());
        let domains = result.unwrap().0;
        assert_eq!(domains.len(), 2);
        // rust has 2 patterns, should be first
        assert_eq!(domains[0].domain, "rust");
        assert_eq!(domains[0].count, 2);
    }

    #[tokio::test]
    async fn test_api_tiers_handler() {
        let (_tmp, path) = create_test_db();
        let state = test_state(path);
        let result = api_tiers(State(state), test_auth()).await;
        assert!(result.is_ok());
        let tiers = result.unwrap().0;
        assert_eq!(tiers.booster, 1);
        assert_eq!(tiers.crystal, 1);
        assert_eq!(tiers.reflex, 1);
    }

    #[tokio::test]
    async fn test_api_patterns_handler() {
        let (_tmp, path) = create_test_db();
        let state = test_state(path);
        let params = PatternsQuery {
            limit: Some(10),
            offset: None,
            domain: None,
            tier: None,
            search: None,
            sort_by: None,
            sort_order: None,
        };
        let result = api_patterns(State(state), test_auth(), Query(params)).await;
        assert!(result.is_ok());
        let resp = result.unwrap().0;
        assert_eq!(resp.patterns.len(), 3);
    }

    #[tokio::test]
    async fn test_api_patterns_with_domain_filter() {
        let (_tmp, path) = create_test_db();
        let state = test_state(path);
        let params = PatternsQuery {
            limit: Some(10),
            offset: None,
            domain: Some("rust".to_string()),
            tier: None,
            search: None,
            sort_by: None,
            sort_order: None,
        };
        let result = api_patterns(State(state), test_auth(), Query(params)).await;
        assert!(result.is_ok());
        let resp = result.unwrap().0;
        assert_eq!(resp.patterns.len(), 2);
    }

    #[tokio::test]
    async fn test_api_pulse_handler() {
        let (_tmp, path) = create_test_db();
        let state = test_state(path);
        let result = api_pulse(State(state), test_auth()).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_api_graph_handler() {
        let (_tmp, path) = create_test_db();
        let state = test_state(path);
        let result = api_graph(State(state), test_auth()).await;
        assert!(result.is_ok());
        let graph = result.unwrap().0;
        assert_eq!(graph.nodes.len(), 3);
        // No context_graph table, so domain-based edges are generated dynamically.
        // "rust" domain has 2 patterns → 1 chain edge; "python" has 1 → no edge.
        assert_eq!(graph.edges.len(), 1);
        assert_eq!(graph.edges[0].weight, 0.5);
    }

    #[tokio::test]
    async fn test_api_graph_3d_handler() {
        let (_tmp, path) = create_test_db();
        let state = test_state(path);
        let result = api_graph_3d(State(state), test_auth()).await;
        assert!(result.is_ok());
        let graph = result.unwrap().0;
        assert_eq!(graph.nodes.len(), 3);
        // All nodes should have valid zone assignments and colors
        for node in &graph.nodes {
            assert!(
                node.zone == "core" || node.zone == "future" || node.zone == "history",
                "Invalid zone: {}",
                node.zone
            );
            assert!(node.size > 0.0, "Node size should be positive");
            assert!(!node.color.is_empty(), "Node color should be set");
        }
        // No context_graph/wormholes/profdag tables → domain-based edges generated.
        // "rust" domain has 2 patterns → 1 chain edge with type "domain".
        assert_eq!(graph.edges.len(), 1);
        assert_eq!(graph.edges[0].edge_type, "domain");
        // Stats should reflect the data
        assert_eq!(graph.stats.total_nodes, 3);
        assert_eq!(graph.stats.total_edges, 1);
    }

    #[test]
    fn test_epoch_days_to_ymd() {
        // 2026-02-07 = day 20,491 since epoch
        // 2024-01-01 = day 19,723 since epoch
        let (y, m, d) = epoch_days_to_ymd(0); // 1970-01-01
        assert_eq!((y, m, d), (1970, 1, 1));

        let (y, m, d) = epoch_days_to_ymd(365); // 1971-01-01
        assert_eq!((y, m, d), (1971, 1, 1));
    }
}
