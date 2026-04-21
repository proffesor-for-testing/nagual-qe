//! API key management for per-agent authentication.
//!
//! Keys use `ngk_<32-hex-chars>` format, hashed with BLAKE3 before storage.
//! Supports scopes (`read`, `write`, `admin`) for fine-grained access control.

use std::sync::Arc;

use chrono::{DateTime, Utc};
use serde::Serialize;

use crate::db::SqliteDb;
use crate::error::Result;

/// An API key record (never contains the raw key).
#[derive(Debug, Clone, Serialize)]
pub struct ApiKeyRecord {
    pub id: String,
    pub name: String,
    pub key_prefix: String,
    pub scopes: Vec<String>,
    pub created_at: DateTime<Utc>,
    pub last_used_at: Option<DateTime<Utc>>,
    pub revoked_at: Option<DateTime<Utc>>,
    pub created_by: Option<String>,
}

impl ApiKeyRecord {
    /// Check if this key has the given scope.
    pub fn has_scope(&self, scope: &str) -> bool {
        self.scopes.iter().any(|s| s == scope || s == "admin")
    }

    /// Check if this key is currently active (not revoked).
    pub fn is_active(&self) -> bool {
        self.revoked_at.is_none()
    }
}

/// Store and manage API keys in SQLite.
pub struct ApiKeyStore {
    db: Arc<SqliteDb>,
}

impl ApiKeyStore {
    /// Create a new store, ensuring the schema exists.
    pub async fn new(db: Arc<SqliteDb>) -> Result<Self> {
        let store = Self { db };
        store.ensure_schema().await?;
        Ok(store)
    }

    async fn ensure_schema(&self) -> Result<()> {
        self.db
            .execute_batch(
                "CREATE TABLE IF NOT EXISTS api_keys (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL UNIQUE,
                    key_prefix TEXT NOT NULL,
                    key_hash TEXT NOT NULL UNIQUE,
                    scopes TEXT NOT NULL DEFAULT 'read,write',
                    created_at TEXT NOT NULL,
                    last_used_at TEXT,
                    revoked_at TEXT,
                    created_by TEXT
                )",
            )
            .await
    }

    /// Generate a new API key. Returns `(plaintext_key, record)`.
    /// The plaintext is shown once and never stored.
    pub async fn create_key(
        &self,
        name: &str,
        scopes: &[String],
        created_by: Option<&str>,
    ) -> Result<(String, ApiKeyRecord)> {
        let plaintext = generate_key();
        let key_hash = hash_key(&plaintext);
        let prefix = format!("ngk_{}", &plaintext[4..8]);
        let id = uuid::Uuid::new_v4().to_string();
        let now = Utc::now();
        let scopes_str = scopes.join(",");

        self.db
            .execute(
                "INSERT INTO api_keys (id, name, key_prefix, key_hash, scopes, created_at, created_by)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                &[
                    &id as &dyn rusqlite::ToSql,
                    &name,
                    &prefix,
                    &key_hash,
                    &scopes_str,
                    &now.to_rfc3339(),
                    &created_by.map(|s| s.to_string()),
                ],
            )
            .await?;

        let record = ApiKeyRecord {
            id,
            name: name.to_string(),
            key_prefix: prefix,
            scopes: scopes.to_vec(),
            created_at: now,
            last_used_at: None,
            revoked_at: None,
            created_by: created_by.map(|s| s.to_string()),
        };

        Ok((plaintext, record))
    }

    /// Validate a plaintext key. Returns the record if active, None if missing/revoked.
    ///
    /// Uses `with_connection` to keep `&dyn ToSql` refs inside a sync closure,
    /// which avoids the non-`Send` future that axum handlers require.
    pub async fn validate_key(&self, plaintext: &str) -> Result<Option<ApiKeyRecord>> {
        let key_hash = hash_key(plaintext);

        self.db
            .with_connection(move |conn| {
                let mut stmt = conn
                    .prepare(
                        "SELECT id, name, key_prefix, scopes, created_at, last_used_at, revoked_at, created_by
                         FROM api_keys WHERE key_hash = ?1 AND revoked_at IS NULL",
                    )
                    .map_err(crate::error::DatabaseError::from)?;

                let mut rows = stmt
                    .query(rusqlite::params![key_hash])
                    .map_err(crate::error::DatabaseError::from)?;

                match rows.next().map_err(crate::error::DatabaseError::from)? {
                    Some(row) => {
                        let scopes_str: String = row.get(3).map_err(crate::error::DatabaseError::from)?;
                        Ok(Some(ApiKeyRecord {
                            id: row.get(0).map_err(crate::error::DatabaseError::from)?,
                            name: row.get(1).map_err(crate::error::DatabaseError::from)?,
                            key_prefix: row.get(2).map_err(crate::error::DatabaseError::from)?,
                            scopes: scopes_str.split(',').map(|s| s.trim().to_string()).collect(),
                            created_at: parse_dt(row.get::<_, String>(4).map_err(crate::error::DatabaseError::from)?),
                            last_used_at: row.get::<_, Option<String>>(5).map_err(crate::error::DatabaseError::from)?.map(parse_dt),
                            revoked_at: row.get::<_, Option<String>>(6).map_err(crate::error::DatabaseError::from)?.map(parse_dt),
                            created_by: row.get(7).map_err(crate::error::DatabaseError::from)?,
                        }))
                    }
                    None => Ok(None),
                }
            })
            .await
    }

    /// Update last_used_at for a key (fire-and-forget).
    ///
    /// Uses `with_connection` for Send-safety in axum handlers.
    pub async fn touch_last_used(&self, id: &str) -> Result<()> {
        let now = Utc::now().to_rfc3339();
        let id = id.to_string();
        self.db
            .with_connection(move |conn| {
                conn.execute(
                    "UPDATE api_keys SET last_used_at = ?1 WHERE id = ?2",
                    rusqlite::params![now, id],
                )
                .map_err(crate::error::DatabaseError::from)?;
                Ok(())
            })
            .await
    }

    /// List all keys, optionally including revoked ones.
    pub async fn list_keys(&self, include_revoked: bool) -> Result<Vec<ApiKeyRecord>> {
        let sql = if include_revoked {
            "SELECT id, name, key_prefix, scopes, created_at, last_used_at, revoked_at, created_by
             FROM api_keys ORDER BY created_at DESC"
        } else {
            "SELECT id, name, key_prefix, scopes, created_at, last_used_at, revoked_at, created_by
             FROM api_keys WHERE revoked_at IS NULL ORDER BY created_at DESC"
        };

        self.db
            .query(sql, &[], |row| {
                let scopes_str: String = row.get(3)?;
                Ok(ApiKeyRecord {
                    id: row.get(0)?,
                    name: row.get(1)?,
                    key_prefix: row.get(2)?,
                    scopes: scopes_str.split(',').map(|s| s.trim().to_string()).collect(),
                    created_at: parse_dt(row.get::<_, String>(4)?),
                    last_used_at: row.get::<_, Option<String>>(5)?.map(parse_dt),
                    revoked_at: row.get::<_, Option<String>>(6)?.map(parse_dt),
                    created_by: row.get(7)?,
                })
            })
            .await
    }

    /// Revoke a key by name or ID (soft-delete via revoked_at).
    pub async fn revoke_key(&self, name_or_id: &str) -> Result<bool> {
        let now = Utc::now().to_rfc3339();
        let rows = self
            .db
            .execute(
                "UPDATE api_keys SET revoked_at = ?1
                 WHERE (name = ?2 OR id = ?2) AND revoked_at IS NULL",
                &[&now as &dyn rusqlite::ToSql, &name_or_id],
            )
            .await?;
        Ok(rows > 0)
    }
}

/// Generate a random API key in `ngk_<32-hex-chars>` format.
pub fn generate_key() -> String {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let mut bytes = [0u8; 16];
    rng.fill(&mut bytes);
    format!("ngk_{}", hex::encode(bytes))
}

/// BLAKE3-hash a plaintext key for storage.
pub fn hash_key(key: &str) -> String {
    let hash = blake3::hash(key.as_bytes());
    hash.to_hex().to_string()
}

/// Parse an RFC3339 datetime string, falling back to epoch on error.
fn parse_dt(s: String) -> DateTime<Utc> {
    DateTime::parse_from_rfc3339(&s)
        .map(|dt| dt.with_timezone(&Utc))
        .unwrap_or_else(|_| DateTime::UNIX_EPOCH)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_key_format() {
        let key = generate_key();
        assert!(key.starts_with("ngk_"), "Key should start with ngk_");
        assert_eq!(key.len(), 4 + 32, "Key should be ngk_ + 32 hex chars");
        // Verify hex chars after prefix
        assert!(key[4..].chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn test_hash_key_deterministic() {
        let key = "ngk_0123456789abcdef0123456789abcdef";
        let h1 = hash_key(key);
        let h2 = hash_key(key);
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_hash_key_different_inputs() {
        let h1 = hash_key("ngk_aaaa");
        let h2 = hash_key("ngk_bbbb");
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_api_key_record_has_scope() {
        let rec = ApiKeyRecord {
            id: "test".into(),
            name: "test".into(),
            key_prefix: "ngk_1234".into(),
            scopes: vec!["read".into(), "write".into()],
            created_at: Utc::now(),
            last_used_at: None,
            revoked_at: None,
            created_by: None,
        };
        assert!(rec.has_scope("read"));
        assert!(rec.has_scope("write"));
        assert!(!rec.has_scope("admin"));
    }

    #[test]
    fn test_admin_scope_grants_all() {
        let rec = ApiKeyRecord {
            id: "test".into(),
            name: "admin-key".into(),
            key_prefix: "ngk_5678".into(),
            scopes: vec!["admin".into()],
            created_at: Utc::now(),
            last_used_at: None,
            revoked_at: None,
            created_by: None,
        };
        assert!(rec.has_scope("read"));
        assert!(rec.has_scope("write"));
        assert!(rec.has_scope("admin"));
    }

    #[test]
    fn test_is_active() {
        let mut rec = ApiKeyRecord {
            id: "test".into(),
            name: "test".into(),
            key_prefix: "ngk_1234".into(),
            scopes: vec!["read".into()],
            created_at: Utc::now(),
            last_used_at: None,
            revoked_at: None,
            created_by: None,
        };
        assert!(rec.is_active());
        rec.revoked_at = Some(Utc::now());
        assert!(!rec.is_active());
    }

    #[tokio::test]
    async fn test_store_create_and_validate() {
        let db = Arc::new(SqliteDb::open_in_memory().unwrap());
        let store = ApiKeyStore::new(db).await.unwrap();

        let (plaintext, record) = store
            .create_key("test-agent", &["read".into(), "write".into()], None)
            .await
            .unwrap();

        assert!(plaintext.starts_with("ngk_"));
        assert_eq!(record.name, "test-agent");
        assert_eq!(record.scopes, vec!["read", "write"]);

        // Validate
        let found = store.validate_key(&plaintext).await.unwrap();
        assert!(found.is_some());
        let found = found.unwrap();
        assert_eq!(found.name, "test-agent");
    }

    #[tokio::test]
    async fn test_store_revoke_and_validate() {
        let db = Arc::new(SqliteDb::open_in_memory().unwrap());
        let store = ApiKeyStore::new(db).await.unwrap();

        let (plaintext, _) = store
            .create_key("revoke-me", &["read".into()], None)
            .await
            .unwrap();

        // Revoke
        let revoked = store.revoke_key("revoke-me").await.unwrap();
        assert!(revoked);

        // Should no longer validate
        let found = store.validate_key(&plaintext).await.unwrap();
        assert!(found.is_none());
    }

    #[tokio::test]
    async fn test_store_list_keys() {
        let db = Arc::new(SqliteDb::open_in_memory().unwrap());
        let store = ApiKeyStore::new(db).await.unwrap();

        store.create_key("key-a", &["read".into()], None).await.unwrap();
        store.create_key("key-b", &["write".into()], None).await.unwrap();

        let active = store.list_keys(false).await.unwrap();
        assert_eq!(active.len(), 2);

        store.revoke_key("key-a").await.unwrap();

        let active = store.list_keys(false).await.unwrap();
        assert_eq!(active.len(), 1);

        let all = store.list_keys(true).await.unwrap();
        assert_eq!(all.len(), 2);
    }

    #[tokio::test]
    async fn test_store_touch_last_used() {
        let db = Arc::new(SqliteDb::open_in_memory().unwrap());
        let store = ApiKeyStore::new(db).await.unwrap();

        let (plaintext, record) = store
            .create_key("touch-me", &["read".into()], None)
            .await
            .unwrap();
        assert!(record.last_used_at.is_none());

        store.touch_last_used(&record.id).await.unwrap();

        let found = store.validate_key(&plaintext).await.unwrap().unwrap();
        assert!(found.last_used_at.is_some());
    }

    #[tokio::test]
    async fn test_store_duplicate_name_fails() {
        let db = Arc::new(SqliteDb::open_in_memory().unwrap());
        let store = ApiKeyStore::new(db).await.unwrap();

        store.create_key("unique", &["read".into()], None).await.unwrap();
        let result = store.create_key("unique", &["read".into()], None).await;
        assert!(result.is_err());
    }
}
