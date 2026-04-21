//! SQLite database with SQLCipher encryption support.
//!
//! Provides encrypted database connections using SQLCipher,
//! with key derivation, password rotation, and WAL mode configuration.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use rusqlite::Connection;
use tokio::sync::Mutex;
use tracing::{debug, info, warn};

use super::crypto::{derive_key_from_password, CryptoError, DerivedKey, Salt};
use crate::error::{DatabaseError, NagualError, Result};

/// Configuration for encrypted SQLite database.
#[derive(Debug, Clone)]
pub struct SqliteConfig {
    /// Path to the database file.
    pub path: PathBuf,

    /// Path to the salt file (stored separately from database).
    pub salt_path: Option<PathBuf>,

    /// Enable WAL mode (recommended for concurrency).
    pub wal_mode: bool,

    /// Busy timeout in milliseconds.
    pub busy_timeout_ms: u32,

    /// Enable foreign keys.
    pub foreign_keys: bool,

    /// SQLCipher page size (must match between encryption and decryption).
    pub cipher_page_size: u32,

    /// SQLCipher KDF iterations (higher = slower but more secure).
    pub kdf_iterations: u32,

    /// SQLCipher memory security (prevents key from being paged to disk).
    pub cipher_memory_security: bool,
}

impl Default for SqliteConfig {
    fn default() -> Self {
        Self {
            path: PathBuf::from("nagual.db"),
            salt_path: None,
            wal_mode: true,
            busy_timeout_ms: 5000,
            foreign_keys: true,
            cipher_page_size: 4096,
            kdf_iterations: 256000, // SQLCipher default
            cipher_memory_security: true,
        }
    }
}

impl SqliteConfig {
    /// Create a new config with the given path.
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            ..Default::default()
        }
    }

    /// Get the default salt path (database path + ".salt").
    pub fn default_salt_path(&self) -> PathBuf {
        let mut salt_path = self.path.clone();
        let filename = salt_path
            .file_name()
            .map(|s| format!("{}.salt", s.to_string_lossy()))
            .unwrap_or_else(|| "nagual.db.salt".to_string());
        salt_path.set_file_name(filename);
        salt_path
    }

    /// Get the configured salt path or the default.
    pub fn salt_path(&self) -> PathBuf {
        self.salt_path
            .clone()
            .unwrap_or_else(|| self.default_salt_path())
    }

    /// Set the database path.
    pub fn with_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.path = path.into();
        self
    }

    /// Set the salt path.
    pub fn with_salt_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.salt_path = Some(path.into());
        self
    }

    /// Enable or disable WAL mode.
    pub fn with_wal_mode(mut self, enabled: bool) -> Self {
        self.wal_mode = enabled;
        self
    }

    /// Set busy timeout.
    pub fn with_busy_timeout(mut self, timeout_ms: u32) -> Self {
        self.busy_timeout_ms = timeout_ms;
        self
    }

    /// Create a config for testing (in-memory).
    pub fn for_testing() -> Self {
        Self {
            path: PathBuf::from(":memory:"),
            salt_path: None,
            wal_mode: false, // WAL not supported for in-memory
            busy_timeout_ms: 5000,
            foreign_keys: true,
            cipher_page_size: 4096,
            kdf_iterations: 64000, // Faster for tests
            cipher_memory_security: false,
        }
    }
}

/// Encrypted SQLite database connection wrapper.
pub struct EncryptedSqliteDb {
    conn: Mutex<Connection>,
    config: SqliteConfig,
    salt: Salt,
}

impl EncryptedSqliteDb {
    /// Open or create an encrypted database at the given path.
    ///
    /// If the database doesn't exist, it will be created with the given password.
    /// If the database exists, it will be opened with the given password.
    pub fn open_encrypted(config: &SqliteConfig, password: &str) -> Result<Self> {
        let salt = load_or_create_salt(config)?;
        let key = derive_key_for_db(password, &salt)?;

        let conn = if config.path.to_string_lossy() == ":memory:" {
            Connection::open_in_memory().map_err(DatabaseError::from)?
        } else {
            Connection::open(&config.path).map_err(DatabaseError::from)?
        };

        // Set encryption key and configure SQLCipher
        configure_encryption(&conn, &key, config)?;

        // Apply standard pragmas
        configure_pragmas(&conn, config)?;

        // Verify the database is accessible (this will fail if password is wrong)
        verify_encryption(&conn)?;

        info!(
            path = %config.path.display(),
            wal_mode = config.wal_mode,
            "Opened encrypted SQLite database"
        );

        Ok(Self {
            conn: Mutex::new(conn),
            config: config.clone(),
            salt,
        })
    }

    /// Initialize a new encrypted database.
    ///
    /// Creates the database file and salt file, then returns the connection.
    pub fn init_encrypted(config: &SqliteConfig, password: &str) -> Result<Self> {
        // Ensure parent directory exists
        if let Some(parent) = config.path.parent() {
            if !parent.exists() {
                fs::create_dir_all(parent).map_err(NagualError::from)?;
            }
        }

        // Remove existing database if present (for fresh init)
        if config.path.exists() && config.path.to_string_lossy() != ":memory:" {
            warn!(path = %config.path.display(), "Removing existing database for fresh init");
            fs::remove_file(&config.path).map_err(NagualError::from)?;
        }

        // Generate new salt
        let salt = Salt::generate().map_err(|e| {
            NagualError::config(format!("Failed to generate salt: {}", e))
        })?;

        // Save salt to file
        save_salt(&salt, &config.salt_path())?;

        let key = derive_key_for_db(password, &salt)?;

        let conn = if config.path.to_string_lossy() == ":memory:" {
            Connection::open_in_memory().map_err(DatabaseError::from)?
        } else {
            Connection::open(&config.path).map_err(DatabaseError::from)?
        };

        configure_encryption(&conn, &key, config)?;
        configure_pragmas(&conn, config)?;

        // Create metadata table to mark as encrypted
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS _nagual_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            INSERT OR REPLACE INTO _nagual_meta (key, value) VALUES
                ('encrypted', 'true'),
                ('version', '1');",
        )
        .map_err(DatabaseError::from)?;

        info!(
            path = %config.path.display(),
            salt_path = %config.salt_path().display(),
            "Initialized new encrypted SQLite database"
        );

        Ok(Self {
            conn: Mutex::new(conn),
            config: config.clone(),
            salt,
        })
    }

    /// Change the encryption password.
    ///
    /// This re-encrypts the database with a new key derived from the new password.
    /// The salt remains the same.
    pub async fn change_password(&self, new_password: &str) -> Result<()> {
        let new_key = derive_key_for_db(new_password, &self.salt)?;

        let conn = self.conn.lock().await;

        // Use PRAGMA rekey to change the encryption key
        let hex_key = new_key.as_hex();
        conn.execute_batch(&format!("PRAGMA rekey = \"x'{}'\";", hex_key))
            .map_err(DatabaseError::from)?;

        info!(
            path = %self.config.path.display(),
            "Changed database encryption password"
        );

        Ok(())
    }

    /// Get the database path.
    pub fn path(&self) -> &Path {
        &self.config.path
    }

    /// Get the salt.
    pub fn salt(&self) -> &Salt {
        &self.salt
    }

    /// Get the configuration.
    pub fn config(&self) -> &SqliteConfig {
        &self.config
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
    pub async fn query<T, F>(
        &self,
        sql: &str,
        params: &[&dyn rusqlite::ToSql],
        f: F,
    ) -> Result<Vec<T>>
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
        F: FnOnce(&Connection) -> std::result::Result<T, DatabaseError>,
    {
        let conn = self.conn.lock().await;
        Ok(f(&conn)?)
    }

    /// Get mutable access to the raw connection (use with caution).
    pub async fn with_connection_mut<T, F>(&self, f: F) -> Result<T>
    where
        F: FnOnce(&mut Connection) -> std::result::Result<T, DatabaseError>,
    {
        let mut conn = self.conn.lock().await;
        Ok(f(&mut conn)?)
    }
}

/// Detect if a database file is encrypted.
pub fn is_database_encrypted(path: &Path) -> Result<bool> {
    if !path.exists() {
        return Ok(false);
    }

    // Read first 16 bytes of the file
    let bytes = fs::read(path).map_err(NagualError::from)?;

    if bytes.len() < 16 {
        return Ok(false);
    }

    // SQLite header magic: "SQLite format 3\0"
    let sqlite_magic = b"SQLite format 3\0";

    // If the header matches SQLite magic, it's NOT encrypted
    // Encrypted databases have scrambled headers
    Ok(&bytes[..16] != sqlite_magic)
}

/// Migrate an unencrypted database to encrypted.
///
/// This uses a row-by-row copy approach since SQLCipher encrypted connections
/// cannot directly attach unencrypted databases.
pub async fn migrate_to_encrypted(
    unencrypted_path: &Path,
    config: &SqliteConfig,
    password: &str,
) -> Result<EncryptedSqliteDb> {
    if !unencrypted_path.exists() {
        return Err(NagualError::config(format!(
            "Source database not found: {}",
            unencrypted_path.display()
        )));
    }

    // Open unencrypted database
    let source = Connection::open(unencrypted_path).map_err(DatabaseError::from)?;

    // Get list of tables to migrate
    let tables: Vec<String> = {
        let mut stmt = source
            .prepare("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' AND name NOT LIKE '_nagual_%'")
            .map_err(DatabaseError::from)?;
        let rows = stmt
            .query_map([], |row| row.get(0))
            .map_err(DatabaseError::from)?;
        rows.map(|r| r.map_err(DatabaseError::from))
            .collect::<std::result::Result<Vec<_>, _>>()?
    };

    // Get table schemas and data before creating encrypted db
    let mut table_data: Vec<(String, String, Vec<Vec<rusqlite::types::Value>>)> = Vec::new();

    for table in &tables {
        // Get table schema
        let schema: String = source
            .query_row(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name=?",
                [&table],
                |row| row.get(0),
            )
            .map_err(DatabaseError::from)?;

        // Get column count for this table
        let col_info: Vec<String> = {
            let mut stmt = source
                .prepare(&format!("PRAGMA table_info({})", table))
                .map_err(DatabaseError::from)?;
            let rows = stmt
                .query_map([], |row| row.get::<_, String>(1))
                .map_err(DatabaseError::from)?;
            rows.map(|r| r.map_err(DatabaseError::from))
                .collect::<std::result::Result<Vec<_>, _>>()?
        };

        // Get all rows
        let mut stmt = source
            .prepare(&format!("SELECT * FROM {}", table))
            .map_err(DatabaseError::from)?;
        let col_count = col_info.len();
        let rows: Vec<Vec<rusqlite::types::Value>> = {
            let mapped = stmt
                .query_map([], |row| {
                    let mut values = Vec::with_capacity(col_count);
                    for i in 0..col_count {
                        values.push(row.get(i)?);
                    }
                    Ok(values)
                })
                .map_err(DatabaseError::from)?;
            mapped
                .map(|r| r.map_err(DatabaseError::from))
                .collect::<std::result::Result<Vec<_>, _>>()?
        };

        table_data.push((table.clone(), schema, rows));
    }

    let table_count = table_data.len();

    // Create new encrypted database
    let encrypted = EncryptedSqliteDb::init_encrypted(config, password)?;

    // Migrate each table
    for (table, schema, rows) in table_data {
        debug!(table = %table, rows = rows.len(), "Migrating table to encrypted database");

        // Create table in encrypted db
        encrypted.execute_batch(&schema).await?;

        // Insert rows one by one
        if !rows.is_empty() {
            let col_count = rows[0].len();
            let placeholders: Vec<&str> = (0..col_count).map(|_| "?").collect();
            let insert_sql = format!(
                "INSERT INTO {} VALUES ({})",
                table,
                placeholders.join(", ")
            );

            for row in rows {
                let params: Vec<&dyn rusqlite::ToSql> = row
                    .iter()
                    .map(|v| v as &dyn rusqlite::ToSql)
                    .collect();
                encrypted.execute(&insert_sql, params.as_slice()).await?;
            }
        }
    }

    info!(
        source = %unencrypted_path.display(),
        target = %config.path.display(),
        tables_migrated = table_count,
        "Migrated database to encrypted format"
    );

    Ok(encrypted)
}

// Helper functions

fn derive_key_for_db(password: &str, salt: &Salt) -> Result<DerivedKey> {
    derive_key_from_password(password, salt).map_err(|e: CryptoError| {
        NagualError::config(format!("Key derivation failed: {}", e))
    })
}

fn load_or_create_salt(config: &SqliteConfig) -> Result<Salt> {
    let salt_path = config.salt_path();

    if salt_path.exists() {
        load_salt(&salt_path)
    } else if config.path.exists() && config.path.to_string_lossy() != ":memory:" {
        // Database exists but no salt - this is an error for encrypted databases
        Err(NagualError::config(format!(
            "Salt file not found: {}. Cannot decrypt database without salt.",
            salt_path.display()
        )))
    } else {
        // Generate new salt for new database
        let salt = Salt::generate().map_err(|e| {
            NagualError::config(format!("Failed to generate salt: {}", e))
        })?;
        save_salt(&salt, &salt_path)?;
        Ok(salt)
    }
}

fn load_salt(path: &Path) -> Result<Salt> {
    let hex = fs::read_to_string(path).map_err(NagualError::from)?;
    Salt::from_hex(hex.trim()).map_err(|e| {
        NagualError::config(format!("Invalid salt file: {}", e))
    })
}

fn save_salt(salt: &Salt, path: &Path) -> Result<()> {
    // Ensure parent directory exists
    if let Some(parent) = path.parent() {
        if !parent.exists() {
            fs::create_dir_all(parent).map_err(NagualError::from)?;
        }
    }

    fs::write(path, salt.as_hex()).map_err(NagualError::from)?;
    debug!(path = %path.display(), "Saved salt file");
    Ok(())
}

fn configure_encryption(conn: &Connection, key: &DerivedKey, config: &SqliteConfig) -> Result<()> {
    // Set the encryption key (must be first pragma after opening)
    let hex_key = key.as_hex();
    conn.execute_batch(&format!("PRAGMA key = \"x'{}'\";", hex_key))
        .map_err(DatabaseError::from)?;

    // Configure SQLCipher settings
    conn.execute_batch(&format!(
        "PRAGMA cipher_page_size = {};
         PRAGMA kdf_iter = {};
         PRAGMA cipher_memory_security = {};",
        config.cipher_page_size,
        config.kdf_iterations,
        if config.cipher_memory_security { "ON" } else { "OFF" }
    ))
    .map_err(DatabaseError::from)?;

    Ok(())
}

fn configure_pragmas(conn: &Connection, config: &SqliteConfig) -> Result<()> {
    let mut pragmas = vec![
        format!("PRAGMA busy_timeout = {};", config.busy_timeout_ms),
        format!(
            "PRAGMA foreign_keys = {};",
            if config.foreign_keys { "ON" } else { "OFF" }
        ),
    ];

    if config.wal_mode && config.path.to_string_lossy() != ":memory:" {
        pragmas.push("PRAGMA journal_mode = WAL;".to_string());
        pragmas.push("PRAGMA synchronous = NORMAL;".to_string());
    }

    conn.execute_batch(&pragmas.join("\n"))
        .map_err(DatabaseError::from)?;

    Ok(())
}

fn verify_encryption(conn: &Connection) -> Result<()> {
    // Try to read from the database - this will fail if the key is wrong
    conn.query_row("SELECT COUNT(*) FROM sqlite_master", [], |_| Ok(()))
        .map_err(|e| {
            NagualError::config(format!(
                "Failed to verify database encryption (wrong password?): {}",
                e
            ))
        })?;
    Ok(())
}

/// Arc wrapper for encrypted SQLite database.
pub type SharedEncryptedSqliteDb = Arc<EncryptedSqliteDb>;

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn test_config(temp_dir: &TempDir) -> SqliteConfig {
        SqliteConfig {
            path: temp_dir.path().join("test.db"),
            salt_path: Some(temp_dir.path().join("test.salt")),
            wal_mode: true,
            busy_timeout_ms: 5000,
            foreign_keys: true,
            cipher_page_size: 4096,
            kdf_iterations: 64000, // Faster for tests
            cipher_memory_security: false,
        }
    }

    #[tokio::test]
    async fn test_init_encrypted_database() {
        let temp_dir = TempDir::new().unwrap();
        let config = test_config(&temp_dir);

        let db = EncryptedSqliteDb::init_encrypted(&config, "test_password_123").unwrap();

        // Verify database is accessible
        assert!(db.table_exists("_nagual_meta").await.unwrap());

        // Verify salt file was created
        assert!(config.salt_path().exists());
    }

    #[tokio::test]
    async fn test_open_encrypted_database() {
        let temp_dir = TempDir::new().unwrap();
        let config = test_config(&temp_dir);

        // Create database
        {
            let db = EncryptedSqliteDb::init_encrypted(&config, "test_password_123").unwrap();
            db.execute_batch("CREATE TABLE test (id INTEGER PRIMARY KEY, name TEXT);")
                .await
                .unwrap();
            db.execute("INSERT INTO test (name) VALUES (?)", &[&"Alice"])
                .await
                .unwrap();
        }

        // Reopen database
        let db = EncryptedSqliteDb::open_encrypted(&config, "test_password_123").unwrap();

        // Verify data persisted
        let names: Vec<String> = db
            .query("SELECT name FROM test", &[], |row| row.get(0))
            .await
            .unwrap();

        assert_eq!(names, vec!["Alice"]);
    }

    #[tokio::test]
    async fn test_wrong_password_fails() {
        let temp_dir = TempDir::new().unwrap();
        let config = test_config(&temp_dir);

        // Create database with one password
        EncryptedSqliteDb::init_encrypted(&config, "correct_password_here").unwrap();

        // Try to open with wrong password
        let result = EncryptedSqliteDb::open_encrypted(&config, "wrong_password_here");

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_change_password() {
        let temp_dir = TempDir::new().unwrap();
        let config = test_config(&temp_dir);

        // Create database
        let db = EncryptedSqliteDb::init_encrypted(&config, "old_password_here").unwrap();
        db.execute_batch("CREATE TABLE test (id INTEGER PRIMARY KEY);")
            .await
            .unwrap();

        // Change password
        db.change_password("new_password_here").await.unwrap();
        drop(db);

        // Verify old password no longer works
        let result = EncryptedSqliteDb::open_encrypted(&config, "old_password_here");
        assert!(result.is_err());

        // Verify new password works
        let db = EncryptedSqliteDb::open_encrypted(&config, "new_password_here").unwrap();
        assert!(db.table_exists("test").await.unwrap());
    }

    #[test]
    fn test_is_database_encrypted() {
        let temp_dir = TempDir::new().unwrap();

        // Unencrypted database
        let unenc_path = temp_dir.path().join("unencrypted.db");
        let conn = Connection::open(&unenc_path).unwrap();
        conn.execute_batch("CREATE TABLE test (id INTEGER);").unwrap();
        drop(conn);

        assert!(!is_database_encrypted(&unenc_path).unwrap());

        // Encrypted database
        let config = SqliteConfig::new(temp_dir.path().join("encrypted.db"));
        EncryptedSqliteDb::init_encrypted(&config, "password123!").unwrap();

        assert!(is_database_encrypted(&config.path).unwrap());
    }

    #[tokio::test]
    async fn test_migrate_to_encrypted() {
        let temp_dir = TempDir::new().unwrap();

        // Create unencrypted database with data
        let unenc_path = temp_dir.path().join("source.db");
        {
            let conn = Connection::open(&unenc_path).unwrap();
            conn.execute_batch(
                "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT);
                 INSERT INTO users (name) VALUES ('Alice'), ('Bob');",
            )
            .unwrap();
        }

        // Migrate to encrypted
        let config = SqliteConfig::new(temp_dir.path().join("encrypted.db"))
            .with_salt_path(temp_dir.path().join("encrypted.salt"));

        let db = migrate_to_encrypted(&unenc_path, &config, "secure_password!").await.unwrap();

        // Verify data was migrated
        let names: Vec<String> = db
            .query("SELECT name FROM users ORDER BY name", &[], |row| row.get(0))
            .await
            .unwrap();

        assert_eq!(names, vec!["Alice", "Bob"]);

        // Verify new database is encrypted
        assert!(is_database_encrypted(&config.path).unwrap());
    }

    #[test]
    fn test_default_salt_path() {
        let config = SqliteConfig::new("/path/to/database.db");
        assert_eq!(
            config.default_salt_path(),
            PathBuf::from("/path/to/database.db.salt")
        );
    }
}
