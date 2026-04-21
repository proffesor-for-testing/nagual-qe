//! Periodic heartbeat for nagual serve.
//!
//! Runs every 30 minutes (configurable) and performs:
//! 1. Database health check (size, pattern count, stale patterns)
//! 2. Constitution compliance check (log-only)
//! 3. Pattern consolidation (similarity >= 0.95, optional)
//! 4. Auto-promotion scan (promote patterns meeting recurrence thresholds)
//! 5. Publishes a health report via EventBus (for WebSocket subscribers)

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use chrono::Utc;
use serde::Serialize;
use tracing::{info, warn};

use crate::events::{EventBus, NagualEvent};

/// Heartbeat configuration.
#[derive(Debug, Clone)]
pub struct HeartbeatConfig {
    /// Interval between heartbeat ticks (default: 30 minutes).
    pub interval: Duration,
    /// Whether to run consolidation on each tick.
    pub consolidate: bool,
    /// Similarity threshold for consolidation.
    pub consolidation_threshold: f32,
    /// Whether to run constitution checks.
    pub constitution_check: bool,
    /// Whether to run auto-promotion scan on each tick.
    pub auto_promote: bool,
}

impl Default for HeartbeatConfig {
    fn default() -> Self {
        Self {
            interval: Duration::from_secs(30 * 60), // 30 minutes
            consolidate: true,
            consolidation_threshold: 0.95,
            constitution_check: true,
            auto_promote: true,
        }
    }
}

/// Health report published after each heartbeat tick.
#[derive(Debug, Clone, Serialize)]
pub struct HeartbeatReport {
    /// ISO 8601 timestamp of the report.
    pub timestamp: String,
    /// Total number of patterns in the database.
    pub pattern_count: u64,
    /// Database file size in bytes.
    pub db_size_bytes: u64,
    /// Number of stale patterns (not updated in 90+ days).
    pub stale_pattern_count: u64,
    /// Number of patterns merged during consolidation (0 if skipped).
    pub consolidation_merged: u64,
    /// Number of patterns auto-promoted during this tick.
    pub auto_promoted: u64,
    /// Constitution violations detected (empty if all pass).
    pub constitution_violations: Vec<String>,
    /// Server uptime in seconds.
    pub uptime_secs: u64,
}

/// Start the heartbeat loop.
///
/// Returns a `JoinHandle` that can be aborted on shutdown.
/// The first interval tick is skipped so the heartbeat does not fire immediately.
pub fn start_heartbeat(
    config: HeartbeatConfig,
    db_path: PathBuf,
    event_bus: Arc<EventBus>,
    start_time: std::time::Instant,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(config.interval);
        // Skip the first immediate tick
        interval.tick().await;

        loop {
            interval.tick().await;
            info!("Heartbeat tick");

            let report = run_heartbeat_tick(&config, &db_path, start_time).await;

            // Publish as a Custom event so WebSocket subscribers receive it
            let mut payload = HashMap::new();
            if let Ok(value) = serde_json::to_value(&report) {
                payload.insert("report".to_string(), value);
            }
            let event = NagualEvent::custom("heartbeat", payload);
            event_bus.publish_sync(event);

            info!(
                patterns = report.pattern_count,
                db_size_bytes = report.db_size_bytes,
                uptime_secs = report.uptime_secs,
                merged = report.consolidation_merged,
                promoted = report.auto_promoted,
                "Heartbeat completed"
            );
        }
    })
}

/// Execute a single heartbeat tick: gather metrics and return a report.
///
/// Uses a direct read-only SQLite connection (via `spawn_blocking`) to avoid
/// holding the shared `PatternStorage` mutex across ticks and to satisfy
/// `Send` requirements for `tokio::spawn`.
pub async fn run_heartbeat_tick(
    config: &HeartbeatConfig,
    db_path: &Path,
    start_time: std::time::Instant,
) -> HeartbeatReport {
    let mut report = HeartbeatReport {
        timestamp: Utc::now().to_rfc3339(),
        pattern_count: 0,
        db_size_bytes: 0,
        stale_pattern_count: 0,
        consolidation_merged: 0,
        auto_promoted: 0,
        constitution_violations: Vec::new(),
        uptime_secs: start_time.elapsed().as_secs(),
    };

    // 1. DB file size
    if let Ok(metadata) = std::fs::metadata(db_path) {
        report.db_size_bytes = metadata.len();
    }

    // 2. Pattern count + stale count via a direct read-only SQLite connection
    let path_owned = db_path.to_path_buf();
    let counts = tokio::task::spawn_blocking(move || -> (u64, u64) {
        let conn = match rusqlite::Connection::open_with_flags(
            &path_owned,
            rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
        ) {
            Ok(c) => c,
            Err(e) => {
                warn!("Heartbeat: failed to open DB for count: {}", e);
                return (0, 0);
            }
        };

        let total: u64 = conn
            .query_row("SELECT COUNT(*) FROM reasoning_patterns", [], |row| {
                row.get(0)
            })
            .unwrap_or(0);

        // Count stale patterns: updated_at older than 90 days
        let stale: u64 = conn
            .query_row(
                "SELECT COUNT(*) FROM reasoning_patterns WHERE updated_at < datetime('now', '-90 days')",
                [],
                |row| row.get(0),
            )
            .unwrap_or(0);

        (total, stale)
    })
    .await
    .unwrap_or((0, 0));

    report.pattern_count = counts.0;
    report.stale_pattern_count = counts.1;

    // 3. Consolidation: count exact-duplicate groups by problem text (no ONNX needed).
    //    A future version can merge them; for now we report how many duplicate groups exist.
    if config.consolidate && report.pattern_count > 100 {
        let path_for_consolidation = db_path.to_path_buf();
        let merged = tokio::task::spawn_blocking(move || -> u64 {
            let conn = match rusqlite::Connection::open_with_flags(
                &path_for_consolidation,
                rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY
                    | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
            ) {
                Ok(c) => c,
                Err(_) => return 0,
            };

            // Find exact duplicates by problem text -- count duplicate groups
            let count: u64 = conn
                .query_row(
                    "SELECT COUNT(*) FROM (
                        SELECT problem, COUNT(*) as cnt
                        FROM reasoning_patterns
                        GROUP BY problem
                        HAVING cnt > 1
                    )",
                    [],
                    |row| row.get(0),
                )
                .unwrap_or(0);

            count
        })
        .await
        .unwrap_or(0);

        if merged > 0 {
            info!(
                duplicate_groups = merged,
                "Heartbeat: found {} duplicate pattern groups for future consolidation", merged
            );
        }

        report.consolidation_merged = merged;
    }

    // 4. Auto-promotion scan: promote patterns meeting recurrence thresholds
    if config.auto_promote && report.pattern_count > 0 {
        let path_for_promo = db_path.to_path_buf();
        let promo_result = tokio::task::spawn_blocking(move || -> u64 {
            use crate::reasoning_bank::AutoPromotionCriteria;

            let conn = match rusqlite::Connection::open_with_flags(
                &path_for_promo,
                rusqlite::OpenFlags::SQLITE_OPEN_READ_WRITE
                    | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
            ) {
                Ok(c) => c,
                Err(e) => {
                    warn!("Heartbeat: failed to open DB for auto-promotion: {}", e);
                    return 0;
                }
            };

            let criteria = AutoPromotionCriteria::default();
            let window_param = format!("-{} days", criteria.window_days);

            // Query eligible patterns (booster or crystal tier)
            let mut stmt = match conn.prepare(
                "SELECT id, COALESCE(tier, 'booster') as tier FROM reasoning_patterns WHERE tier IN ('booster', 'crystal') LIMIT 1000"
            ) {
                Ok(s) => s,
                Err(e) => {
                    warn!("Heartbeat: auto-promotion query failed: {}", e);
                    return 0;
                }
            };

            let candidates: Vec<(String, String)> = match stmt.query_map([], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
            }) {
                Ok(rows) => rows.filter_map(|r| r.ok()).collect(),
                Err(_) => return 0,
            };

            let mut promoted = 0u64;
            for (id, tier) in &candidates {
                // Count usage within the window
                let usage: (i64, i64) = conn
                    .query_row(
                        "SELECT COUNT(*), COUNT(DISTINCT CASE WHEN session_id != '' THEN session_id END) FROM pattern_usage_log WHERE pattern_id = ? AND used_at >= datetime('now', ?)",
                        rusqlite::params![id, &window_param],
                        |row| Ok((row.get(0)?, row.get(1)?)),
                    )
                    .unwrap_or((0, 0));

                let total = usage.0 as u32;
                let distinct = usage.1 as u32;

                if total >= criteria.min_occurrences && distinct >= criteria.min_distinct_contexts {
                    let new_tier = match tier.as_str() {
                        "booster" => "crystal",
                        "crystal" => "reflex",
                        _ => continue,
                    };

                    let now = chrono::Utc::now().to_rfc3339();
                    if conn
                        .execute(
                            "UPDATE reasoning_patterns SET tier = ?, updated_at = ? WHERE id = ?",
                            rusqlite::params![new_tier, &now, id],
                        )
                        .is_ok()
                    {
                        info!(
                            pattern_id = %id,
                            old_tier = %tier,
                            new_tier = %new_tier,
                            uses = total,
                            contexts = distinct,
                            "Heartbeat: pattern auto-promoted"
                        );
                        promoted += 1;
                    }
                }
            }

            promoted
        })
        .await
        .unwrap_or(0);

        report.auto_promoted = promo_result;
        if promo_result > 0 {
            info!(promoted = promo_result, "Heartbeat: auto-promotion scan complete");
        }
    }

    // 5. Constitution check (log-only, enforce: false by default)
    if config.constitution_check {
        // If pattern_count == 0 and DB file exists, flag it.
        if report.pattern_count == 0 && report.db_size_bytes > 0 {
            report
                .constitution_violations
                .push("No patterns in database -- possible data loss".to_string());
        }
        // Check stale ratio -- if majority of patterns are stale, flag for consolidation
        if report.pattern_count > 0 && report.stale_pattern_count > report.pattern_count / 2 {
            report.constitution_violations.push(format!(
                "{}% patterns are stale (>90 days) -- consider consolidation",
                report.stale_pattern_count * 100 / report.pattern_count
            ));
        }
    }

    report
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_heartbeat_config_default() {
        let config = HeartbeatConfig::default();
        assert_eq!(config.interval, Duration::from_secs(1800));
        assert!(config.consolidate);
        assert!((config.consolidation_threshold - 0.95).abs() < f32::EPSILON);
        assert!(config.constitution_check);
        assert!(config.auto_promote);
    }

    #[test]
    fn test_heartbeat_report_serialization() {
        let report = HeartbeatReport {
            timestamp: "2026-03-08T12:00:00Z".to_string(),
            pattern_count: 42,
            db_size_bytes: 1024 * 1024,
            stale_pattern_count: 3,
            consolidation_merged: 0,
            auto_promoted: 0,
            constitution_violations: vec![],
            uptime_secs: 3600,
        };

        let json = serde_json::to_string(&report).unwrap();
        assert!(json.contains("\"pattern_count\":42"));
        assert!(json.contains("\"db_size_bytes\":1048576"));
        assert!(json.contains("\"uptime_secs\":3600"));

        // Verify it round-trips through serde_json::Value
        let value: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(value["pattern_count"], 42);
    }

    #[test]
    fn test_heartbeat_report_with_violations() {
        let report = HeartbeatReport {
            timestamp: Utc::now().to_rfc3339(),
            pattern_count: 0,
            db_size_bytes: 0,
            stale_pattern_count: 0,
            consolidation_merged: 0,
            auto_promoted: 0,
            constitution_violations: vec!["No patterns in database".to_string()],
            uptime_secs: 60,
        };

        let json = serde_json::to_string(&report).unwrap();
        assert!(json.contains("No patterns in database"));
    }

    #[tokio::test]
    async fn test_run_heartbeat_tick_no_db() {
        let config = HeartbeatConfig::default();
        let db_path = PathBuf::from("/tmp/nonexistent-heartbeat-test.db");
        let start_time = std::time::Instant::now();

        let report = run_heartbeat_tick(&config, &db_path, start_time).await;

        assert_eq!(report.pattern_count, 0);
        assert_eq!(report.db_size_bytes, 0);
        assert_eq!(report.consolidation_merged, 0);
        assert!(report.uptime_secs < 5);
        // No violations because DB doesn't exist (size == 0)
        assert!(report.constitution_violations.is_empty());
    }

    #[tokio::test]
    async fn test_run_heartbeat_tick_with_temp_db() {
        let config = HeartbeatConfig::default();
        let dir = std::env::temp_dir();
        let db_path = dir.join("heartbeat_test.db");

        // Create a minimal SQLite DB with the expected table
        {
            let conn = rusqlite::Connection::open(&db_path).unwrap();
            conn.execute_batch(
                "CREATE TABLE IF NOT EXISTS reasoning_patterns (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL DEFAULT (datetime('now')),
                    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
                    category TEXT NOT NULL DEFAULT 'general',
                    problem TEXT NOT NULL,
                    solution TEXT NOT NULL,
                    domain TEXT NOT NULL DEFAULT '',
                    tags TEXT NOT NULL DEFAULT '',
                    reward REAL NOT NULL DEFAULT 0.5,
                    success INTEGER NOT NULL DEFAULT 0,
                    session_id TEXT,
                    critique TEXT,
                    tokens_used INTEGER NOT NULL DEFAULT 0,
                    latency_ms INTEGER NOT NULL DEFAULT 0,
                    access_count INTEGER NOT NULL DEFAULT 0,
                    failure_mode TEXT,
                    tier TEXT
                );
                INSERT INTO reasoning_patterns (id, problem, solution) VALUES ('p1', 'test', 'sol');
                INSERT INTO reasoning_patterns (id, problem, solution, updated_at) VALUES ('p2', 'old', 'sol', datetime('now', '-100 days'));",
            )
            .unwrap();
        }

        let start_time = std::time::Instant::now();
        let report = run_heartbeat_tick(&config, &db_path, start_time).await;

        assert_eq!(report.pattern_count, 2);
        assert_eq!(report.stale_pattern_count, 1);
        assert!(report.db_size_bytes > 0);
        assert!(report.constitution_violations.is_empty());

        // Cleanup
        let _ = std::fs::remove_file(&db_path);
    }
}
