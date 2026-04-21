//! Database abstraction layer for dual-write persistence.
//!
//! Provides unified access to both SQLite (local) and PostgreSQL (cloud)
//! databases with connection management and health checking.
//!
//! # Modules
//!
//! - [`crypto`]: Cryptographic key derivation for database encryption
//! - [`sqlite`]: Encrypted SQLite with SQLCipher support
//! - [`fts`]: Full-text search with FTS5 virtual tables
//! - [`postgres`]: PostgreSQL configuration with TLS support
//! - [`conflicts`]: Conflict logging and resolution
//! - [`dual_write`]: Dual-write adapter with circuit breaker and DLQ
//! - [`sessions`]: Session management and analytics

mod conflicts;
pub mod crypto;
mod dual_write;
pub mod fts;
pub mod pg_notify;
mod postgres;
pub mod sessions;
pub mod sqlite;
pub mod users;

pub use conflicts::{
    log_conflict, AutoResolveResult, ConflictLog, ConflictLogEntry, ConflictResolution,
    ConflictStats,
};
pub use crypto::{
    derive_key_from_password, derive_key_with_salt, Argon2Params, CryptoError, CryptoResult,
    DerivedKey, KeyDerivation, Salt,
};
pub use dual_write::{
    ConflictWinner, DlqProcessResult, DualWritable, DualWriteAdapter, DualWriteConfig,
    DualWriteDlqMeta, DualWriteResult, OperationType,
};
pub use fts::{
    create_patterns_table, fts_search, fts_search_patterns, init_patterns_fts, Fts5Config,
    Fts5Tokenizer, FtsSearchOptions, FtsSearchResult, PatternFts,
};
pub use pg_notify::{
    parse_notification, notification_to_event, PgNotification, PgNotifyHandle, PgNotifyListener,
    ConsolidationCompletePayload, PatternPromotedPayload, PatternStoredPayload,
    CHANNEL_CONSOLIDATION_COMPLETE, CHANNEL_PATTERN_PROMOTED, CHANNEL_PATTERN_STORED,
};
pub use postgres::{PoolConfig, PostgresConfig, TlsConfig, TlsVerifyMode};
pub use sessions::{Session, SessionManager, SessionStats};
pub use users::{User, UserStore};
pub use sqlite::{
    is_database_encrypted, migrate_to_encrypted, EncryptedSqliteDb, SharedEncryptedSqliteDb,
    SqliteConfig,
};

use std::path::Path;
use std::sync::Arc;
use tokio::sync::Mutex;

use rusqlite::Connection as SqliteConnection;
use sqlx::postgres::{PgPool, PgPoolOptions};
use sqlx::Row;

use crate::error::{DatabaseError, Result};

/// Configuration for database connections.
#[derive(Debug, Clone)]
pub struct DatabaseConfig {
    /// Path to the SQLite database file.
    pub sqlite_path: String,
    /// PostgreSQL connection string (optional for local-only mode).
    pub postgres_url: Option<String>,
    /// Maximum PostgreSQL pool connections.
    pub max_pg_connections: u32,
    /// Connection timeout in seconds.
    pub connection_timeout_secs: u64,
}

impl Default for DatabaseConfig {
    fn default() -> Self {
        Self {
            sqlite_path: "nagual.db".to_string(),
            postgres_url: None,
            max_pg_connections: 5,
            connection_timeout_secs: 30,
        }
    }
}

/// SQLite connection wrapper with thread-safe access.
pub struct SqliteDb {
    conn: Mutex<SqliteConnection>,
    path: String,
}

impl SqliteDb {
    /// Open or create a SQLite database at the given path.
    pub fn open(path: impl AsRef<Path>) -> std::result::Result<Self, DatabaseError> {
        let path_str = path.as_ref().to_string_lossy().to_string();
        let conn = SqliteConnection::open(&path)?;

        // Enable WAL mode for better concurrency
        conn.execute_batch(
            "PRAGMA journal_mode=WAL;
             PRAGMA synchronous=NORMAL;
             PRAGMA foreign_keys=ON;
             PRAGMA busy_timeout=5000;",
        )?;

        Ok(Self {
            conn: Mutex::new(conn),
            path: path_str,
        })
    }

    /// Open an in-memory SQLite database (for testing).
    pub fn open_in_memory() -> std::result::Result<Self, DatabaseError> {
        let conn = SqliteConnection::open_in_memory()?;
        conn.execute_batch(
            "PRAGMA foreign_keys=ON;
             PRAGMA busy_timeout=5000;",
        )?;

        Ok(Self {
            conn: Mutex::new(conn),
            path: ":memory:".to_string(),
        })
    }

    /// Get the database path.
    pub fn path(&self) -> &str {
        &self.path
    }

    /// Execute a SQL statement with no return value.
    pub async fn execute(&self, sql: &str, params: &[&dyn rusqlite::ToSql]) -> Result<usize> {
        let conn = self.conn.lock().await;
        let rows = conn.execute(sql, params).map_err(DatabaseError::from)?;
        Ok(rows)
    }

    /// Execute a batch of SQL statements.
    pub async fn execute_batch(&self, sql: &str) -> Result<()> {
        let conn = self.conn.lock().await;
        conn.execute_batch(sql).map_err(DatabaseError::from)?;
        Ok(())
    }

    /// Query and map results.
    pub async fn query<T, F>(&self, sql: &str, params: &[&dyn rusqlite::ToSql], f: F) -> Result<Vec<T>>
    where
        F: FnMut(&rusqlite::Row<'_>) -> rusqlite::Result<T>,
    {
        let conn = self.conn.lock().await;
        let mut stmt = conn.prepare(sql).map_err(DatabaseError::from)?;
        let rows = stmt.query_map(params, f).map_err(DatabaseError::from)?;

        let mut results = Vec::new();
        for row in rows {
            results.push(row.map_err(DatabaseError::from)?);
        }
        Ok(results)
    }

    /// Query for a single optional row.
    pub async fn query_one<T, F>(
        &self,
        sql: &str,
        params: &[&dyn rusqlite::ToSql],
        f: F,
    ) -> Result<Option<T>>
    where
        F: FnOnce(&rusqlite::Row<'_>) -> rusqlite::Result<T>,
    {
        let conn = self.conn.lock().await;
        let mut stmt = conn.prepare(sql).map_err(DatabaseError::from)?;
        let mut rows = stmt.query(params).map_err(DatabaseError::from)?;

        match rows.next().map_err(DatabaseError::from)? {
            Some(row) => Ok(Some(f(row).map_err(DatabaseError::from)?)),
            None => Ok(None),
        }
    }

    /// Check if a table exists.
    pub async fn table_exists(&self, table_name: &str) -> Result<bool> {
        let result = self
            .query_one(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                &[&table_name],
                |row| row.get::<_, String>(0),
            )
            .await?;
        Ok(result.is_some())
    }

    /// Begin a transaction and execute a closure.
    pub async fn transaction<T, F>(&self, f: F) -> Result<T>
    where
        F: FnOnce(&rusqlite::Transaction<'_>) -> std::result::Result<T, DatabaseError>,
    {
        let mut conn = self.conn.lock().await;
        let tx = conn.transaction().map_err(DatabaseError::from)?;
        let result = f(&tx)?;
        tx.commit().map_err(DatabaseError::from)?;
        Ok(result)
    }

    /// Get the raw connection for advanced operations (use with caution).
    pub async fn with_connection<T, F>(&self, f: F) -> Result<T>
    where
        F: FnOnce(&SqliteConnection) -> std::result::Result<T, DatabaseError>,
    {
        let conn = self.conn.lock().await;
        Ok(f(&conn)?)
    }

    /// Get mutable access to the raw connection (use with caution).
    pub async fn with_connection_mut<T, F>(&self, f: F) -> Result<T>
    where
        F: FnOnce(&mut SqliteConnection) -> std::result::Result<T, DatabaseError>,
    {
        let mut conn = self.conn.lock().await;
        Ok(f(&mut conn)?)
    }
}

/// PostgreSQL connection pool wrapper.
pub struct PostgresDb {
    pool: PgPool,
    url: String,
}

impl PostgresDb {
    /// Connect to a PostgreSQL database.
    pub async fn connect(url: &str, max_connections: u32) -> std::result::Result<Self, DatabaseError> {
        let pool = PgPoolOptions::new()
            .max_connections(max_connections)
            .acquire_timeout(std::time::Duration::from_secs(30))
            .connect(url)
            .await
            .map_err(DatabaseError::from)?;

        Ok(Self {
            pool,
            url: url.to_string(),
        })
    }

    /// Get the connection URL (with password masked).
    pub fn url_masked(&self) -> String {
        // Mask password in URL for logging
        if let Some(at_pos) = self.url.find('@') {
            if let Some(colon_pos) = self.url[..at_pos].rfind(':') {
                let prefix = &self.url[..colon_pos + 1];
                let suffix = &self.url[at_pos..];
                return format!("{}****{}", prefix, suffix);
            }
        }
        self.url.clone()
    }

    /// Get the underlying pool.
    pub fn pool(&self) -> &PgPool {
        &self.pool
    }

    /// Execute a SQL statement.
    pub async fn execute(&self, sql: &str) -> Result<u64> {
        let result = sqlx::query(sql)
            .execute(&self.pool)
            .await
            .map_err(DatabaseError::from)?;
        Ok(result.rows_affected())
    }

    /// Check if a table exists.
    pub async fn table_exists(&self, table_name: &str) -> Result<bool> {
        let row = sqlx::query(
            "SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_schema = 'public'
                AND table_name = $1
            )",
        )
        .bind(table_name)
        .fetch_one(&self.pool)
        .await
        .map_err(DatabaseError::from)?;

        Ok(row.get::<bool, _>(0))
    }

    /// Check connection health.
    pub async fn is_healthy(&self) -> bool {
        sqlx::query("SELECT 1")
            .fetch_one(&self.pool)
            .await
            .is_ok()
    }
}

/// Dual database manager for coordinated access.
pub struct DualDb {
    /// SQLite database (always available).
    pub sqlite: Arc<SqliteDb>,
    /// PostgreSQL database (optional).
    pub postgres: Option<Arc<PostgresDb>>,
}

impl DualDb {
    /// Create a new dual database manager.
    pub async fn new(config: &DatabaseConfig) -> Result<Self> {
        let sqlite = Arc::new(SqliteDb::open(&config.sqlite_path)?);

        let postgres = if let Some(ref url) = config.postgres_url {
            Some(Arc::new(
                PostgresDb::connect(url, config.max_pg_connections).await?,
            ))
        } else {
            None
        };

        Ok(Self { sqlite, postgres })
    }

    /// Create a dual database manager for testing (in-memory SQLite, no PostgreSQL).
    pub fn new_for_testing() -> Result<Self> {
        Ok(Self {
            sqlite: Arc::new(SqliteDb::open_in_memory()?),
            postgres: None,
        })
    }

    /// Check if PostgreSQL is configured.
    pub fn has_postgres(&self) -> bool {
        self.postgres.is_some()
    }

    /// Get health status of both databases.
    pub async fn health_check(&self) -> (bool, Option<bool>) {
        let sqlite_healthy = self.sqlite.table_exists("sqlite_master").await.is_ok();
        let postgres_healthy = if let Some(ref pg) = self.postgres {
            Some(pg.is_healthy().await)
        } else {
            None
        };
        (sqlite_healthy, postgres_healthy)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_sqlite_open_in_memory() {
        let db = SqliteDb::open_in_memory().unwrap();
        assert_eq!(db.path(), ":memory:");
    }

    #[tokio::test]
    async fn test_sqlite_execute_batch() {
        let db = SqliteDb::open_in_memory().unwrap();
        db.execute_batch("CREATE TABLE test (id INTEGER PRIMARY KEY)")
            .await
            .unwrap();
        assert!(db.table_exists("test").await.unwrap());
    }

    #[tokio::test]
    async fn test_dual_db_testing() {
        let db = DualDb::new_for_testing().unwrap();
        assert!(!db.has_postgres());
    }
}
