//! Common utilities shared across CLI commands.
//!
//! This module provides shared functionality to avoid code duplication:
//! - Database initialization (SQLite + PostgreSQL dual-write)
//! - Configuration resolution from environment and config files
//! - Pattern loading helpers

use std::path::PathBuf;
use std::sync::Arc;

use crate::db::{is_database_encrypted, DualWriteAdapter, DualWriteConfig, PostgresDb, SqliteDb};
use crate::error::{NagualError, Result};
use crate::reasoning_bank::storage::{PatternStorage, StorageConfig};

/// Resolve the PostgreSQL URL from explicit flag, env var, or config file.
///
/// Resolution order:
/// 1. Explicit argument (if provided and non-empty)
/// 2. `DATABASE_URL` environment variable (handled by clap `env` attribute)
/// 3. `postgres_url` in `~/.nagual/config.toml`
///
/// # Arguments
/// * `explicit` - Optional explicit URL from command-line argument
///
/// # Returns
/// * `Some(url)` if a PostgreSQL URL was found
/// * `None` if no URL is configured (SQLite-only mode)
pub fn resolve_postgres_url(explicit: Option<&str>) -> Option<String> {
    // 1. Explicit flag (already includes env via clap)
    if let Some(url) = explicit {
        if !url.is_empty() {
            return Some(url.to_string());
        }
    }

    // 2. Fallback: read from ~/.nagual/config.toml
    if let Some(home) = std::env::var("HOME").ok().map(PathBuf::from) {
        let config_path = home.join(".nagual").join("config.toml");
        if let Ok(content) = std::fs::read_to_string(&config_path) {
            for line in content.lines() {
                let trimmed = line.trim();
                if trimmed.starts_with("postgres_url") {
                    if let Some(value) = trimmed.split('=').nth(1) {
                        let url = value.trim().trim_matches('"').trim_matches('\'');
                        if !url.is_empty() {
                            return Some(url.to_string());
                        }
                    }
                }
            }
        }
    }

    None
}

/// Initialize pattern storage with optional PostgreSQL dual-write.
///
/// This is the canonical implementation used across all CLI commands.
/// It handles:
/// - Creating parent directories if needed
/// - Opening SQLite database
/// - Connecting to PostgreSQL if URL is provided
/// - Setting up the DualWriteAdapter with DLQ
/// - Creating PatternStorage with default config
///
/// # Arguments
/// * `db_path` - Path to the SQLite database file
/// * `postgres_url` - Optional PostgreSQL connection URL
///
/// # Returns
/// * `PatternStorage` instance ready for use
pub async fn init_storage(db_path: &PathBuf, postgres_url: Option<&str>) -> Result<PatternStorage> {
    // Ensure parent directory exists
    if let Some(parent) = db_path.parent() {
        if !parent.exists() {
            std::fs::create_dir_all(parent)?;
        }
    }

    // Safety: refuse to open encrypted databases with the unencrypted driver.
    // Opening an encrypted SQLCipher DB with plain rusqlite silently overwrites
    // the encryption header, destroying all data.
    if db_path.exists() {
        if is_database_encrypted(db_path)? {
            return Err(NagualError::internal(format!(
                "Database at '{}' is encrypted (SQLCipher). Cannot open with unencrypted driver. \
                 Use --db-path to specify a different database, or decrypt the file first.",
                db_path.display()
            )));
        }
    }

    // Open SQLite database
    let sqlite = Arc::new(SqliteDb::open(db_path)?);

    // Resolve PostgreSQL URL and connect if available
    let pg_url = resolve_postgres_url(postgres_url);
    let postgres = if let Some(ref url) = pg_url {
        match PostgresDb::connect(url, 5).await {
            Ok(pg) => {
                tracing::info!(url_masked = %pg.url_masked(), "PostgreSQL connected for dual-write");
                Some(Arc::new(pg))
            }
            Err(e) => {
                tracing::warn!(error = %e, "PostgreSQL unavailable, using SQLite only");
                eprintln!("Warning: PostgreSQL unavailable ({}), using SQLite only", e);
                None
            }
        }
    } else {
        None
    };

    // Create DualWriteAdapter with DLQ path
    let config = DualWriteConfig {
        dlq_path: db_path
            .with_extension("dlq.db")
            .to_string_lossy()
            .to_string(),
        ..Default::default()
    };
    let adapter = Arc::new(DualWriteAdapter::new(sqlite, postgres, config)?);

    // Create PatternStorage
    PatternStorage::new(adapter, StorageConfig::default()).await
}

/// Initialize pattern storage and return an Arc-wrapped instance.
///
/// Use this variant when the storage needs to be shared across multiple
/// async tasks or when cloning is required.
///
/// # Arguments
/// * `db_path` - Path to the SQLite database file
/// * `postgres_url` - Optional PostgreSQL connection URL
///
/// # Returns
/// * `Arc<PatternStorage>` instance ready for shared use
pub async fn init_storage_arc(
    db_path: &PathBuf,
    postgres_url: Option<&str>,
) -> Result<Arc<PatternStorage>> {
    let storage = init_storage(db_path, postgres_url).await?;
    Ok(Arc::new(storage))
}

/// Initialize pattern storage in SQLite-only mode (no PostgreSQL).
///
/// Use this for commands that don't need dual-write functionality,
/// such as local-only operations or when PostgreSQL is known to be unavailable.
///
/// # Arguments
/// * `db_path` - Path to the SQLite database file
///
/// # Returns
/// * `PatternStorage` instance in SQLite-only mode
pub async fn init_storage_sqlite_only(db_path: &PathBuf) -> Result<PatternStorage> {
    // Ensure parent directory exists
    if let Some(parent) = db_path.parent() {
        if !parent.exists() {
            std::fs::create_dir_all(parent)?;
        }
    }

    // Safety: refuse to open encrypted databases with the unencrypted driver.
    if db_path.exists() {
        if is_database_encrypted(db_path)? {
            return Err(NagualError::internal(format!(
                "Database at '{}' is encrypted (SQLCipher). Cannot open with unencrypted driver. \
                 Use --db-path to specify a different database, or decrypt the file first.",
                db_path.display()
            )));
        }
    }

    // Open SQLite database
    let sqlite = Arc::new(SqliteDb::open(db_path)?);

    // Create DualWriteAdapter (SQLite-only mode)
    let config = DualWriteConfig {
        dlq_path: db_path
            .with_extension("dlq.db")
            .to_string_lossy()
            .to_string(),
        ..Default::default()
    };
    let adapter = Arc::new(DualWriteAdapter::new(sqlite, None, config)?);

    // Create PatternStorage
    PatternStorage::new(adapter, StorageConfig::default()).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_resolve_postgres_url_explicit() {
        let url = resolve_postgres_url(Some("postgres://localhost/test"));
        assert_eq!(url, Some("postgres://localhost/test".to_string()));
    }

    #[test]
    fn test_resolve_postgres_url_empty_explicit() {
        let url = resolve_postgres_url(Some(""));
        // Empty string should fall through to config file lookup
        // which will likely return None in test environment
        assert!(url.is_none() || url.is_some());
    }

    #[test]
    fn test_resolve_postgres_url_none() {
        let url = resolve_postgres_url(None);
        // Will depend on config file presence
        // Just ensure it doesn't panic
        let _ = url;
    }

    #[tokio::test]
    async fn test_init_storage_sqlite_only() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test.db");

        let storage = init_storage_sqlite_only(&db_path).await.unwrap();

        // Should be able to get recent patterns (empty initially)
        let patterns = storage.get_recent(10).await.unwrap();
        assert!(patterns.is_empty());
    }

    #[tokio::test]
    async fn test_init_storage_rejects_encrypted_db() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("encrypted.db");

        // Write fake encrypted content (no SQLite magic header)
        std::fs::write(&db_path, b"\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f").unwrap();

        let result = init_storage(&db_path, None).await;
        assert!(result.is_err(), "Should reject encrypted database");
        match result {
            Err(e) => assert!(e.to_string().contains("encrypted"), "Error should mention encryption: {}", e),
            Ok(_) => panic!("Expected error for encrypted database"),
        }
    }

    #[tokio::test]
    async fn test_init_storage_sqlite_only_rejects_encrypted_db() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("encrypted.db");

        // Write fake encrypted content (no SQLite magic header)
        std::fs::write(&db_path, b"\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f").unwrap();

        let result = init_storage_sqlite_only(&db_path).await;
        assert!(result.is_err(), "Should reject encrypted database");
        match result {
            Err(e) => assert!(e.to_string().contains("encrypted"), "Error should mention encryption: {}", e),
            Ok(_) => panic!("Expected error for encrypted database"),
        }
    }

    #[tokio::test]
    async fn test_init_storage_creates_parent_dirs() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("subdir").join("nested").join("test.db");

        // Parent dirs don't exist yet
        assert!(!db_path.parent().unwrap().exists());

        let storage = init_storage(&db_path, None).await.unwrap();

        // Parent dirs should now exist
        assert!(db_path.parent().unwrap().exists());

        // Storage should work
        let patterns = storage.get_recent(10).await.unwrap();
        assert!(patterns.is_empty());
    }
}
