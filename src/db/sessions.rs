//! Session management and analytics module.
//!
//! Provides session lifecycle management and token efficiency tracking
//! for learning analytics. Sessions track tokens used, patterns learned,
//! and patterns retrieved during a development session.
//!
//! # Example
//!
//! ```ignore
//! use nagual::db::sessions::SessionManager;
//!
//! let manager = SessionManager::new(db)?;
//!
//! // Start a new session
//! let session = manager.start_session(Some("rust")).await?;
//!
//! // Record activity
//! manager.record_tokens(&session.id, 1500).await?;
//! manager.record_pattern_learned(&session.id).await?;
//!
//! // End the session
//! manager.end_session(&session.id).await?;
//!
//! // Get analytics
//! let stats = manager.get_stats().await?;
//! println!("Efficiency: {:.2} patterns/1K tokens", stats.efficiency);
//! ```

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use uuid::Uuid;

use crate::db::SqliteDb;
use crate::error::{NagualError, Result};

/// A session representing a development activity period.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Session {
    /// Unique session identifier.
    pub id: String,
    /// When the session started.
    pub started_at: DateTime<Utc>,
    /// When the session ended (None if still active).
    pub ended_at: Option<DateTime<Utc>>,
    /// Total tokens consumed during this session.
    pub tokens_used: u64,
    /// Number of patterns learned (stored) during this session.
    pub patterns_learned: u32,
    /// Number of patterns retrieved during this session.
    pub patterns_retrieved: u32,
    /// Optional domain focus for this session.
    pub domain: Option<String>,
}

impl Session {
    /// Calculate the duration of the session in seconds.
    /// Returns None if the session has not ended.
    pub fn duration_secs(&self) -> Option<i64> {
        self.ended_at.map(|end| {
            (end - self.started_at).num_seconds()
        })
    }

    /// Calculate token efficiency (patterns learned per 1K tokens).
    pub fn efficiency(&self) -> f64 {
        if self.tokens_used > 0 {
            (self.patterns_learned as f64 * 1000.0) / self.tokens_used as f64
        } else {
            0.0
        }
    }

    /// Check if the session is still active.
    pub fn is_active(&self) -> bool {
        self.ended_at.is_none()
    }
}

/// Aggregated session statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionStats {
    /// Total number of sessions.
    pub total_sessions: u32,
    /// Total tokens used across all sessions.
    pub total_tokens: u64,
    /// Total patterns learned across all sessions.
    pub total_patterns_learned: u32,
    /// Total patterns retrieved across all sessions.
    pub total_patterns_retrieved: u32,
    /// Average tokens per session.
    pub avg_tokens_per_session: f64,
    /// Average patterns learned per session.
    pub avg_patterns_per_session: f64,
    /// Token efficiency (patterns learned per 1K tokens).
    pub efficiency: f64,
    /// Number of active (unclosed) sessions.
    pub active_sessions: u32,
    /// Average session duration in seconds.
    pub avg_duration_secs: f64,
}

impl Default for SessionStats {
    fn default() -> Self {
        Self {
            total_sessions: 0,
            total_tokens: 0,
            total_patterns_learned: 0,
            total_patterns_retrieved: 0,
            avg_tokens_per_session: 0.0,
            avg_patterns_per_session: 0.0,
            efficiency: 0.0,
            active_sessions: 0,
            avg_duration_secs: 0.0,
        }
    }
}

/// Session manager for lifecycle and analytics operations.
pub struct SessionManager {
    db: Arc<SqliteDb>,
}

impl SessionManager {
    /// Create a new session manager with the given database connection.
    pub fn new(db: Arc<SqliteDb>) -> Self {
        Self { db }
    }

    /// Start a new session.
    ///
    /// # Arguments
    /// * `domain` - Optional domain focus for this session.
    ///
    /// # Returns
    /// The newly created session.
    pub async fn start_session(&self, domain: Option<&str>) -> Result<Session> {
        let id = Uuid::new_v4().to_string();
        let now = Utc::now();
        let started_at_str = now.to_rfc3339();

        self.db.execute(
            "INSERT INTO sessions (id, started_at, domain) VALUES (?, ?, ?)",
            &[&id as &dyn rusqlite::ToSql, &started_at_str, &domain],
        ).await?;

        Ok(Session {
            id,
            started_at: now,
            ended_at: None,
            tokens_used: 0,
            patterns_learned: 0,
            patterns_retrieved: 0,
            domain: domain.map(String::from),
        })
    }

    /// End a session.
    ///
    /// # Arguments
    /// * `session_id` - The ID of the session to end.
    pub async fn end_session(&self, session_id: &str) -> Result<()> {
        let now = Utc::now().to_rfc3339();

        let rows = self.db.execute(
            "UPDATE sessions SET ended_at = ? WHERE id = ?",
            &[&now as &dyn rusqlite::ToSql, &session_id],
        ).await?;

        if rows == 0 {
            return Err(NagualError::internal(format!(
                "Session not found: {}",
                session_id
            )));
        }

        Ok(())
    }

    /// Record token usage for a session.
    ///
    /// # Arguments
    /// * `session_id` - The session ID.
    /// * `tokens` - Number of tokens to add.
    pub async fn record_tokens(&self, session_id: &str, tokens: u64) -> Result<()> {
        let tokens_i64 = tokens as i64;

        let rows = self.db.execute(
            "UPDATE sessions SET tokens_used = tokens_used + ? WHERE id = ?",
            &[&tokens_i64 as &dyn rusqlite::ToSql, &session_id],
        ).await?;

        if rows == 0 {
            return Err(NagualError::internal(format!(
                "Session not found: {}",
                session_id
            )));
        }

        Ok(())
    }

    /// Record a pattern learned during a session.
    ///
    /// # Arguments
    /// * `session_id` - The session ID.
    pub async fn record_pattern_learned(&self, session_id: &str) -> Result<()> {
        let rows = self.db.execute(
            "UPDATE sessions SET patterns_learned = patterns_learned + 1 WHERE id = ?",
            &[&session_id as &dyn rusqlite::ToSql],
        ).await?;

        if rows == 0 {
            return Err(NagualError::internal(format!(
                "Session not found: {}",
                session_id
            )));
        }

        Ok(())
    }

    /// Record a pattern retrieved during a session.
    ///
    /// # Arguments
    /// * `session_id` - The session ID.
    pub async fn record_pattern_retrieved(&self, session_id: &str) -> Result<()> {
        let rows = self.db.execute(
            "UPDATE sessions SET patterns_retrieved = patterns_retrieved + 1 WHERE id = ?",
            &[&session_id as &dyn rusqlite::ToSql],
        ).await?;

        if rows == 0 {
            return Err(NagualError::internal(format!(
                "Session not found: {}",
                session_id
            )));
        }

        Ok(())
    }

    /// Get a session by ID.
    ///
    /// # Arguments
    /// * `session_id` - The session ID.
    ///
    /// # Returns
    /// The session if found, None otherwise.
    pub async fn get_session(&self, session_id: &str) -> Result<Option<Session>> {
        self.db.query_one(
            r#"SELECT id, started_at, ended_at, tokens_used, patterns_learned, patterns_retrieved, domain
               FROM sessions WHERE id = ?"#,
            &[&session_id as &dyn rusqlite::ToSql],
            |row| {
                let started_at_str: String = row.get(1)?;
                let ended_at_str: Option<String> = row.get(2)?;

                Ok(Session {
                    id: row.get(0)?,
                    started_at: DateTime::parse_from_rfc3339(&started_at_str)
                        .map(|dt| dt.with_timezone(&Utc))
                        .unwrap_or_else(|_| Utc::now()),
                    ended_at: ended_at_str.and_then(|s| {
                        DateTime::parse_from_rfc3339(&s)
                            .map(|dt| dt.with_timezone(&Utc))
                            .ok()
                    }),
                    tokens_used: row.get::<_, i64>(3)? as u64,
                    patterns_learned: row.get::<_, i64>(4)? as u32,
                    patterns_retrieved: row.get::<_, i64>(5)? as u32,
                    domain: row.get(6)?,
                })
            },
        ).await
    }

    /// Get the current active session (if any).
    ///
    /// If multiple active sessions exist, returns the most recently started one.
    pub async fn get_active_session(&self) -> Result<Option<Session>> {
        self.db.query_one(
            r#"SELECT id, started_at, ended_at, tokens_used, patterns_learned, patterns_retrieved, domain
               FROM sessions WHERE ended_at IS NULL
               ORDER BY started_at DESC LIMIT 1"#,
            &[],
            |row| {
                let started_at_str: String = row.get(1)?;

                Ok(Session {
                    id: row.get(0)?,
                    started_at: DateTime::parse_from_rfc3339(&started_at_str)
                        .map(|dt| dt.with_timezone(&Utc))
                        .unwrap_or_else(|_| Utc::now()),
                    ended_at: None,
                    tokens_used: row.get::<_, i64>(3)? as u64,
                    patterns_learned: row.get::<_, i64>(4)? as u32,
                    patterns_retrieved: row.get::<_, i64>(5)? as u32,
                    domain: row.get(6)?,
                })
            },
        ).await
    }

    /// List recent sessions.
    ///
    /// # Arguments
    /// * `limit` - Maximum number of sessions to return.
    ///
    /// # Returns
    /// List of sessions, most recent first.
    pub async fn list_sessions(&self, limit: usize) -> Result<Vec<Session>> {
        let limit_i64 = limit as i64;

        self.db.query(
            r#"SELECT id, started_at, ended_at, tokens_used, patterns_learned, patterns_retrieved, domain
               FROM sessions ORDER BY started_at DESC LIMIT ?"#,
            &[&limit_i64 as &dyn rusqlite::ToSql],
            |row| {
                let started_at_str: String = row.get(1)?;
                let ended_at_str: Option<String> = row.get(2)?;

                Ok(Session {
                    id: row.get(0)?,
                    started_at: DateTime::parse_from_rfc3339(&started_at_str)
                        .map(|dt| dt.with_timezone(&Utc))
                        .unwrap_or_else(|_| Utc::now()),
                    ended_at: ended_at_str.and_then(|s| {
                        DateTime::parse_from_rfc3339(&s)
                            .map(|dt| dt.with_timezone(&Utc))
                            .ok()
                    }),
                    tokens_used: row.get::<_, i64>(3)? as u64,
                    patterns_learned: row.get::<_, i64>(4)? as u32,
                    patterns_retrieved: row.get::<_, i64>(5)? as u32,
                    domain: row.get(6)?,
                })
            },
        ).await
    }

    /// List sessions filtered by domain.
    ///
    /// # Arguments
    /// * `domain` - Domain to filter by.
    /// * `limit` - Maximum number of sessions to return.
    pub async fn list_sessions_by_domain(&self, domain: &str, limit: usize) -> Result<Vec<Session>> {
        let limit_i64 = limit as i64;

        self.db.query(
            r#"SELECT id, started_at, ended_at, tokens_used, patterns_learned, patterns_retrieved, domain
               FROM sessions WHERE domain = ? ORDER BY started_at DESC LIMIT ?"#,
            &[&domain as &dyn rusqlite::ToSql, &limit_i64],
            |row| {
                let started_at_str: String = row.get(1)?;
                let ended_at_str: Option<String> = row.get(2)?;

                Ok(Session {
                    id: row.get(0)?,
                    started_at: DateTime::parse_from_rfc3339(&started_at_str)
                        .map(|dt| dt.with_timezone(&Utc))
                        .unwrap_or_else(|_| Utc::now()),
                    ended_at: ended_at_str.and_then(|s| {
                        DateTime::parse_from_rfc3339(&s)
                            .map(|dt| dt.with_timezone(&Utc))
                            .ok()
                    }),
                    tokens_used: row.get::<_, i64>(3)? as u64,
                    patterns_learned: row.get::<_, i64>(4)? as u32,
                    patterns_retrieved: row.get::<_, i64>(5)? as u32,
                    domain: row.get(6)?,
                })
            },
        ).await
    }

    /// Get aggregated session statistics.
    pub async fn get_stats(&self) -> Result<SessionStats> {
        self.db.query_one(
            r#"SELECT
                COUNT(*) as total,
                COALESCE(SUM(tokens_used), 0) as tokens,
                COALESCE(SUM(patterns_learned), 0) as learned,
                COALESCE(SUM(patterns_retrieved), 0) as retrieved,
                COALESCE(AVG(tokens_used), 0) as avg_tokens,
                COALESCE(AVG(patterns_learned), 0) as avg_learned,
                (SELECT COUNT(*) FROM sessions WHERE ended_at IS NULL) as active,
                COALESCE(AVG(
                    CASE WHEN ended_at IS NOT NULL
                    THEN (julianday(ended_at) - julianday(started_at)) * 86400
                    ELSE NULL END
                ), 0) as avg_duration
            FROM sessions"#,
            &[],
            |row| {
                let total: i64 = row.get(0)?;
                let tokens: i64 = row.get(1)?;
                let learned: i64 = row.get(2)?;
                let retrieved: i64 = row.get(3)?;
                let avg_tokens: f64 = row.get(4)?;
                let avg_learned: f64 = row.get(5)?;
                let active: i64 = row.get(6)?;
                let avg_duration: f64 = row.get(7)?;

                let efficiency = if tokens > 0 {
                    (learned as f64 * 1000.0) / tokens as f64
                } else {
                    0.0
                };

                Ok(SessionStats {
                    total_sessions: total as u32,
                    total_tokens: tokens as u64,
                    total_patterns_learned: learned as u32,
                    total_patterns_retrieved: retrieved as u32,
                    avg_tokens_per_session: avg_tokens,
                    avg_patterns_per_session: avg_learned,
                    efficiency,
                    active_sessions: active as u32,
                    avg_duration_secs: avg_duration,
                })
            },
        ).await.map(|opt| opt.unwrap_or_default())
    }

    /// Get statistics for a specific time window.
    ///
    /// # Arguments
    /// * `days` - Number of days to look back.
    pub async fn get_stats_for_window(&self, days: u32) -> Result<SessionStats> {
        let days_str = format!("-{} days", days);

        self.db.query_one(
            r#"SELECT
                COUNT(*) as total,
                COALESCE(SUM(tokens_used), 0) as tokens,
                COALESCE(SUM(patterns_learned), 0) as learned,
                COALESCE(SUM(patterns_retrieved), 0) as retrieved,
                COALESCE(AVG(tokens_used), 0) as avg_tokens,
                COALESCE(AVG(patterns_learned), 0) as avg_learned,
                (SELECT COUNT(*) FROM sessions WHERE ended_at IS NULL AND started_at > datetime('now', ?)) as active,
                COALESCE(AVG(
                    CASE WHEN ended_at IS NOT NULL
                    THEN (julianday(ended_at) - julianday(started_at)) * 86400
                    ELSE NULL END
                ), 0) as avg_duration
            FROM sessions
            WHERE started_at > datetime('now', ?)"#,
            &[&days_str as &dyn rusqlite::ToSql, &days_str],
            |row| {
                let total: i64 = row.get(0)?;
                let tokens: i64 = row.get(1)?;
                let learned: i64 = row.get(2)?;
                let retrieved: i64 = row.get(3)?;
                let avg_tokens: f64 = row.get(4)?;
                let avg_learned: f64 = row.get(5)?;
                let active: i64 = row.get(6)?;
                let avg_duration: f64 = row.get(7)?;

                let efficiency = if tokens > 0 {
                    (learned as f64 * 1000.0) / tokens as f64
                } else {
                    0.0
                };

                Ok(SessionStats {
                    total_sessions: total as u32,
                    total_tokens: tokens as u64,
                    total_patterns_learned: learned as u32,
                    total_patterns_retrieved: retrieved as u32,
                    avg_tokens_per_session: avg_tokens,
                    avg_patterns_per_session: avg_learned,
                    efficiency,
                    active_sessions: active as u32,
                    avg_duration_secs: avg_duration,
                })
            },
        ).await.map(|opt| opt.unwrap_or_default())
    }

    /// Delete a session.
    ///
    /// # Arguments
    /// * `session_id` - The session ID to delete.
    pub async fn delete_session(&self, session_id: &str) -> Result<bool> {
        let rows = self.db.execute(
            "DELETE FROM sessions WHERE id = ?",
            &[&session_id as &dyn rusqlite::ToSql],
        ).await?;

        Ok(rows > 0)
    }

    /// Clean up old sessions.
    ///
    /// # Arguments
    /// * `older_than_days` - Delete sessions older than this many days.
    ///
    /// # Returns
    /// Number of sessions deleted.
    pub async fn cleanup_old_sessions(&self, older_than_days: u32) -> Result<usize> {
        let days_str = format!("-{} days", older_than_days);

        let rows = self.db.execute(
            "DELETE FROM sessions WHERE ended_at IS NOT NULL AND started_at < datetime('now', ?)",
            &[&days_str as &dyn rusqlite::ToSql],
        ).await?;

        Ok(rows)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn create_test_db() -> Arc<SqliteDb> {
        let db = SqliteDb::open_in_memory().unwrap();

        // Create the sessions table
        db.execute_batch(
            r#"CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                started_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                ended_at TEXT,
                tokens_used INTEGER DEFAULT 0,
                patterns_learned INTEGER DEFAULT 0,
                patterns_retrieved INTEGER DEFAULT 0,
                domain TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_sessions_started_at ON sessions(started_at);
            CREATE INDEX IF NOT EXISTS idx_sessions_domain ON sessions(domain);"#,
        ).await.unwrap();

        Arc::new(db)
    }

    #[tokio::test]
    async fn test_start_session() {
        let db = create_test_db().await;
        let manager = SessionManager::new(db);

        let session = manager.start_session(Some("rust")).await.unwrap();

        assert!(!session.id.is_empty());
        assert!(session.is_active());
        assert_eq!(session.tokens_used, 0);
        assert_eq!(session.patterns_learned, 0);
        assert_eq!(session.domain, Some("rust".to_string()));
    }

    #[tokio::test]
    async fn test_end_session() {
        let db = create_test_db().await;
        let manager = SessionManager::new(db);

        let session = manager.start_session(None).await.unwrap();
        manager.end_session(&session.id).await.unwrap();

        let ended = manager.get_session(&session.id).await.unwrap().unwrap();
        assert!(!ended.is_active());
        assert!(ended.ended_at.is_some());
    }

    #[tokio::test]
    async fn test_record_tokens() {
        let db = create_test_db().await;
        let manager = SessionManager::new(db);

        let session = manager.start_session(None).await.unwrap();
        manager.record_tokens(&session.id, 1000).await.unwrap();
        manager.record_tokens(&session.id, 500).await.unwrap();

        let updated = manager.get_session(&session.id).await.unwrap().unwrap();
        assert_eq!(updated.tokens_used, 1500);
    }

    #[tokio::test]
    async fn test_record_patterns() {
        let db = create_test_db().await;
        let manager = SessionManager::new(db);

        let session = manager.start_session(None).await.unwrap();
        manager.record_pattern_learned(&session.id).await.unwrap();
        manager.record_pattern_learned(&session.id).await.unwrap();
        manager.record_pattern_retrieved(&session.id).await.unwrap();

        let updated = manager.get_session(&session.id).await.unwrap().unwrap();
        assert_eq!(updated.patterns_learned, 2);
        assert_eq!(updated.patterns_retrieved, 1);
    }

    #[tokio::test]
    async fn test_get_active_session() {
        let db = create_test_db().await;
        let manager = SessionManager::new(db);

        // Initially no active session
        assert!(manager.get_active_session().await.unwrap().is_none());

        // Start a session
        let session = manager.start_session(None).await.unwrap();
        let active = manager.get_active_session().await.unwrap().unwrap();
        assert_eq!(active.id, session.id);

        // End the session
        manager.end_session(&session.id).await.unwrap();
        assert!(manager.get_active_session().await.unwrap().is_none());
    }

    #[tokio::test]
    async fn test_list_sessions() {
        let db = create_test_db().await;
        let manager = SessionManager::new(db);

        // Create a few sessions
        for _ in 0..5 {
            manager.start_session(None).await.unwrap();
        }

        let sessions = manager.list_sessions(10).await.unwrap();
        assert_eq!(sessions.len(), 5);

        let sessions_limited = manager.list_sessions(3).await.unwrap();
        assert_eq!(sessions_limited.len(), 3);
    }

    #[tokio::test]
    async fn test_get_stats() {
        let db = create_test_db().await;
        let manager = SessionManager::new(db);

        // Create sessions with various metrics
        let s1 = manager.start_session(Some("rust")).await.unwrap();
        manager.record_tokens(&s1.id, 1000).await.unwrap();
        manager.record_pattern_learned(&s1.id).await.unwrap();
        manager.end_session(&s1.id).await.unwrap();

        let s2 = manager.start_session(Some("python")).await.unwrap();
        manager.record_tokens(&s2.id, 2000).await.unwrap();
        manager.record_pattern_learned(&s2.id).await.unwrap();
        manager.record_pattern_learned(&s2.id).await.unwrap();
        // s2 remains active

        let stats = manager.get_stats().await.unwrap();

        assert_eq!(stats.total_sessions, 2);
        assert_eq!(stats.total_tokens, 3000);
        assert_eq!(stats.total_patterns_learned, 3);
        assert_eq!(stats.active_sessions, 1);
        assert!(stats.efficiency > 0.0);
    }

    #[tokio::test]
    async fn test_session_efficiency() {
        let session = Session {
            id: "test".to_string(),
            started_at: Utc::now(),
            ended_at: None,
            tokens_used: 2000,
            patterns_learned: 4,
            patterns_retrieved: 10,
            domain: None,
        };

        // 4 patterns per 2000 tokens = 2 patterns per 1K tokens
        assert!((session.efficiency() - 2.0).abs() < 0.001);
    }

    #[tokio::test]
    async fn test_delete_session() {
        let db = create_test_db().await;
        let manager = SessionManager::new(db);

        let session = manager.start_session(None).await.unwrap();
        assert!(manager.get_session(&session.id).await.unwrap().is_some());

        let deleted = manager.delete_session(&session.id).await.unwrap();
        assert!(deleted);

        assert!(manager.get_session(&session.id).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn test_list_sessions_by_domain() {
        let db = create_test_db().await;
        let manager = SessionManager::new(db);

        manager.start_session(Some("rust")).await.unwrap();
        manager.start_session(Some("rust")).await.unwrap();
        manager.start_session(Some("python")).await.unwrap();

        let rust_sessions = manager.list_sessions_by_domain("rust", 10).await.unwrap();
        assert_eq!(rust_sessions.len(), 2);

        let python_sessions = manager.list_sessions_by_domain("python", 10).await.unwrap();
        assert_eq!(python_sessions.len(), 1);
    }
}
