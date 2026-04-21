//! Sync state tracking in local SQLite.
//!
//! Tracks the last push/pull timestamps per remote URL so that
//! subsequent syncs only transfer changed patterns.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::db::SqliteDb;
use crate::error::Result;

/// Sync state for a remote endpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncState {
    pub remote_url: String,
    pub last_push_at: Option<DateTime<Utc>>,
    pub last_pull_at: Option<DateTime<Utc>>,
    pub last_push_count: i64,
    pub last_pull_count: i64,
    pub updated_at: DateTime<Utc>,
}

/// Initialize the cloud_sync_state table.
pub async fn init_sync_state_table(db: &SqliteDb) -> Result<()> {
    let sql = r#"
        CREATE TABLE IF NOT EXISTS cloud_sync_state (
            remote_url TEXT PRIMARY KEY,
            last_push_at TEXT,
            last_pull_at TEXT,
            last_push_count INTEGER DEFAULT 0,
            last_pull_count INTEGER DEFAULT 0,
            updated_at TEXT NOT NULL DEFAULT (datetime('now'))
        )
    "#;
    db.execute(sql, &[]).await?;
    Ok(())
}

/// Get sync state for a remote URL.
pub async fn get_sync_state(db: &SqliteDb, remote_url: &str) -> Result<Option<SyncState>> {
    let sql = "SELECT remote_url, last_push_at, last_pull_at, last_push_count, last_pull_count, updated_at FROM cloud_sync_state WHERE remote_url = ?";
    let url = remote_url.to_string();

    let state = db.with_connection(move |conn| {
        let mut stmt = conn.prepare(sql).map_err(crate::error::DatabaseError::from)?;
        let mut rows = stmt.query(rusqlite::params![url]).map_err(crate::error::DatabaseError::from)?;
        match rows.next().map_err(crate::error::DatabaseError::from)? {
            Some(row) => {
                let remote_url: String = row.get(0).map_err(crate::error::DatabaseError::from)?;
                let push_str: Option<String> = row.get(1).map_err(crate::error::DatabaseError::from)?;
                let pull_str: Option<String> = row.get(2).map_err(crate::error::DatabaseError::from)?;
                let push_count: i64 = row.get(3).map_err(crate::error::DatabaseError::from)?;
                let pull_count: i64 = row.get(4).map_err(crate::error::DatabaseError::from)?;
                let updated_str: String = row.get(5).map_err(crate::error::DatabaseError::from)?;

                let last_push_at = push_str.and_then(|s| DateTime::parse_from_rfc3339(&s).ok().map(|dt| dt.with_timezone(&Utc)));
                let last_pull_at = pull_str.and_then(|s| DateTime::parse_from_rfc3339(&s).ok().map(|dt| dt.with_timezone(&Utc)));
                let updated_at = DateTime::parse_from_rfc3339(&updated_str)
                    .map(|dt| dt.with_timezone(&Utc))
                    .unwrap_or_else(|_| Utc::now());

                Ok(Some(SyncState {
                    remote_url,
                    last_push_at,
                    last_pull_at,
                    last_push_count: push_count,
                    last_pull_count: pull_count,
                    updated_at,
                }))
            }
            None => Ok(None),
        }
    }).await?;

    Ok(state)
}

/// Update push state after a successful push.
pub async fn update_push_state(
    db: &SqliteDb,
    remote_url: &str,
    timestamp: DateTime<Utc>,
    count: i64,
) -> Result<()> {
    let sql = r#"
        INSERT INTO cloud_sync_state (remote_url, last_push_at, last_push_count, updated_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(remote_url) DO UPDATE SET
            last_push_at = excluded.last_push_at,
            last_push_count = excluded.last_push_count,
            updated_at = excluded.updated_at
    "#;
    let ts = timestamp.to_rfc3339();
    let now = Utc::now().to_rfc3339();
    let url = remote_url.to_string();

    db.with_connection(move |conn| {
        conn.execute(sql, rusqlite::params![url, ts, count, now])
            .map_err(crate::error::DatabaseError::from)?;
        Ok(())
    }).await?;

    Ok(())
}

/// Update pull state after a successful pull.
pub async fn update_pull_state(
    db: &SqliteDb,
    remote_url: &str,
    timestamp: DateTime<Utc>,
    count: i64,
) -> Result<()> {
    let sql = r#"
        INSERT INTO cloud_sync_state (remote_url, last_pull_at, last_pull_count, updated_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(remote_url) DO UPDATE SET
            last_pull_at = excluded.last_pull_at,
            last_pull_count = excluded.last_pull_count,
            updated_at = excluded.updated_at
    "#;
    let ts = timestamp.to_rfc3339();
    let now = Utc::now().to_rfc3339();
    let url = remote_url.to_string();

    db.with_connection(move |conn| {
        conn.execute(sql, rusqlite::params![url, ts, count, now])
            .map_err(crate::error::DatabaseError::from)?;
        Ok(())
    }).await?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    async fn setup_db() -> (SqliteDb, TempDir) {
        let temp = TempDir::new().unwrap();
        let db_path = temp.path().join("test.db");
        let db = SqliteDb::open(&db_path).unwrap();
        init_sync_state_table(&db).await.unwrap();
        (db, temp)
    }

    #[tokio::test]
    async fn test_get_sync_state_empty() {
        let (db, _temp) = setup_db().await;
        let state = get_sync_state(&db, "https://example.com").await.unwrap();
        assert!(state.is_none());
    }

    #[tokio::test]
    async fn test_update_push_state() {
        let (db, _temp) = setup_db().await;
        let now = Utc::now();
        update_push_state(&db, "https://example.com", now, 42).await.unwrap();

        let state = get_sync_state(&db, "https://example.com").await.unwrap().unwrap();
        assert_eq!(state.remote_url, "https://example.com");
        assert!(state.last_push_at.is_some());
        assert_eq!(state.last_push_count, 42);
        assert!(state.last_pull_at.is_none());
    }

    #[tokio::test]
    async fn test_update_pull_state() {
        let (db, _temp) = setup_db().await;
        let now = Utc::now();
        update_pull_state(&db, "https://example.com", now, 10).await.unwrap();

        let state = get_sync_state(&db, "https://example.com").await.unwrap().unwrap();
        assert_eq!(state.last_pull_count, 10);
        assert!(state.last_pull_at.is_some());
    }

    #[tokio::test]
    async fn test_update_push_then_pull() {
        let (db, _temp) = setup_db().await;
        let now = Utc::now();

        update_push_state(&db, "https://example.com", now, 5).await.unwrap();
        update_pull_state(&db, "https://example.com", now, 3).await.unwrap();

        let state = get_sync_state(&db, "https://example.com").await.unwrap().unwrap();
        assert_eq!(state.last_push_count, 5);
        assert_eq!(state.last_pull_count, 3);
    }
}
