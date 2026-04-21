//! User management for dashboard login.
//!
//! Stores users in SQLite with argon2-hashed passwords.
//! Used by both the `nagual user` CLI and `nagual serve` login flow.
//!
//! All DB operations use `with_connection` to avoid holding `&dyn ToSql`
//! references across async boundaries (which would make futures !Send).

use std::sync::Arc;

use argon2::{
    password_hash::{rand_core::OsRng, PasswordHash, PasswordHasher, PasswordVerifier, SaltString},
    Argon2,
};
use chrono::{DateTime, Utc};
use serde::Serialize;

use crate::db::SqliteDb;
use crate::error::{DatabaseError, Result};

/// A dashboard user.
#[derive(Debug, Clone, Serialize)]
pub struct User {
    pub id: String,
    pub username: String,
    #[serde(skip)]
    pub password_hash: String,
    pub role: String,
    pub created_at: DateTime<Utc>,
    pub last_login: Option<DateTime<Utc>>,
}

/// Manages users in the SQLite database.
pub struct UserStore {
    db: Arc<SqliteDb>,
}

fn parse_datetime(s: &str) -> DateTime<Utc> {
    DateTime::parse_from_rfc3339(s)
        .map(|dt| dt.with_timezone(&Utc))
        .unwrap_or_else(|_| Utc::now())
}

fn parse_optional_datetime(s: &Option<String>) -> Option<DateTime<Utc>> {
    s.as_ref().and_then(|v| {
        DateTime::parse_from_rfc3339(v)
            .ok()
            .map(|dt| dt.with_timezone(&Utc))
    })
}

impl UserStore {
    /// Create a new UserStore, initializing the `users` table if needed.
    pub async fn new(db: Arc<SqliteDb>) -> Result<Self> {
        db.execute_batch(
            "CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'viewer',
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                last_login TEXT
            )",
        )
        .await?;
        Ok(Self { db })
    }

    /// Create a new user with an argon2-hashed password.
    pub async fn create_user(
        &self,
        username: &str,
        password: &str,
        role: &str,
    ) -> Result<User> {
        let id = uuid::Uuid::new_v4().to_string();
        let salt = SaltString::generate(&mut OsRng);
        let hash = Argon2::default()
            .hash_password(password.as_bytes(), &salt)
            .map_err(|e| crate::error::NagualError::Internal {
                message: format!("Password hash error: {e}"),
            })?
            .to_string();
        let now = Utc::now();
        let now_str = now.to_rfc3339();

        // Clone owned values for the closure
        let id_c = id.clone();
        let username_c = username.to_string();
        let hash_c = hash.clone();
        let role_c = role.to_string();

        self.db
            .with_connection(move |conn| {
                conn.execute(
                    "INSERT INTO users (id, username, password_hash, role, created_at) \
                     VALUES (?1, ?2, ?3, ?4, ?5)",
                    rusqlite::params![id_c, username_c, hash_c, role_c, now_str],
                )
                .map_err(DatabaseError::from)?;
                Ok(())
            })
            .await?;

        Ok(User {
            id,
            username: username.to_string(),
            password_hash: hash,
            role: role.to_string(),
            created_at: now,
            last_login: None,
        })
    }

    /// Verify a username/password pair. Returns the user on success, None on failure.
    /// Updates `last_login` on successful verification.
    pub async fn verify_user(
        &self,
        username: &str,
        password: &str,
    ) -> Result<Option<User>> {
        let username_owned = username.to_string();

        // Query user in a sync closure
        let row: Option<(String, String, String, String, String, Option<String>)> = self
            .db
            .with_connection(move |conn| {
                let mut stmt = conn
                    .prepare(
                        "SELECT id, username, password_hash, role, created_at, last_login \
                         FROM users WHERE username = ?1",
                    )
                    .map_err(DatabaseError::from)?;
                let mut rows = stmt
                    .query(rusqlite::params![username_owned])
                    .map_err(DatabaseError::from)?;
                match rows.next().map_err(DatabaseError::from)? {
                    Some(row) => Ok(Some((
                        row.get::<_, String>(0).map_err(DatabaseError::from)?,
                        row.get::<_, String>(1).map_err(DatabaseError::from)?,
                        row.get::<_, String>(2).map_err(DatabaseError::from)?,
                        row.get::<_, String>(3).map_err(DatabaseError::from)?,
                        row.get::<_, String>(4).map_err(DatabaseError::from)?,
                        row.get::<_, Option<String>>(5).map_err(DatabaseError::from)?,
                    ))),
                    None => Ok(None),
                }
            })
            .await?;

        let (id, uname, hash_str, role, created_str, login_str) = match row {
            Some(r) => r,
            None => return Ok(None),
        };

        // Verify password (CPU-bound, but fast enough inline)
        let parsed_hash = PasswordHash::new(&hash_str).map_err(|e| {
            crate::error::NagualError::Internal {
                message: format!("Hash parse error: {e}"),
            }
        })?;

        if Argon2::default()
            .verify_password(password.as_bytes(), &parsed_hash)
            .is_err()
        {
            return Ok(None);
        }

        // Update last_login
        let now = Utc::now();
        let now_str = now.to_rfc3339();
        let id_c = id.clone();
        let _ = self
            .db
            .with_connection(move |conn| {
                conn.execute(
                    "UPDATE users SET last_login = ?1 WHERE id = ?2",
                    rusqlite::params![now_str, id_c],
                )
                .map_err(DatabaseError::from)?;
                Ok(())
            })
            .await;

        Ok(Some(User {
            id,
            username: uname,
            password_hash: hash_str,
            role,
            created_at: parse_datetime(&created_str),
            last_login: parse_optional_datetime(&login_str),
        }))
    }

    /// List all users (without password hashes).
    pub async fn list_users(&self) -> Result<Vec<User>> {
        self.db
            .with_connection(|conn| {
                let mut stmt = conn
                    .prepare(
                        "SELECT id, username, role, created_at, last_login \
                         FROM users ORDER BY created_at",
                    )
                    .map_err(DatabaseError::from)?;
                let rows = stmt
                    .query_map([], |row| {
                        Ok((
                            row.get::<_, String>(0)?,
                            row.get::<_, String>(1)?,
                            row.get::<_, String>(2)?,
                            row.get::<_, String>(3)?,
                            row.get::<_, Option<String>>(4)?,
                        ))
                    })
                    .map_err(DatabaseError::from)?;

                let mut users = Vec::new();
                for r in rows {
                    let (id, username, role, created_str, login_str) =
                        r.map_err(DatabaseError::from)?;
                    users.push(User {
                        id,
                        username,
                        password_hash: String::new(),
                        role,
                        created_at: parse_datetime(&created_str),
                        last_login: parse_optional_datetime(&login_str),
                    });
                }
                Ok(users)
            })
            .await
    }

    /// Delete a user by username. Returns true if a user was deleted.
    pub async fn delete_user(&self, username: &str) -> Result<bool> {
        let username_owned = username.to_string();
        self.db
            .with_connection(move |conn| {
                let rows = conn
                    .execute(
                        "DELETE FROM users WHERE username = ?1",
                        rusqlite::params![username_owned],
                    )
                    .map_err(DatabaseError::from)?;
                Ok(rows > 0)
            })
            .await
    }

    /// Check if any users exist (determines if login is required).
    pub async fn has_users(&self) -> Result<bool> {
        self.db
            .with_connection(|conn| {
                let count: i64 = conn
                    .query_row("SELECT COUNT(*) FROM users", [], |row| row.get(0))
                    .map_err(DatabaseError::from)?;
                Ok(count > 0)
            })
            .await
    }
}

/// Session cookie operations using HMAC-SHA256.
pub mod session {
    use base64::Engine;
    use ring::hmac;

    const COOKIE_NAME: &str = "nagual_session";
    const SESSION_DURATION_SECS: i64 = 86400; // 24 hours

    /// Generate a session secret key for HMAC signing.
    pub fn generate_secret() -> Vec<u8> {
        use rand::RngCore;
        let mut key = vec![0u8; 32];
        rand::thread_rng().fill_bytes(&mut key);
        key
    }

    /// Create a signed session cookie value.
    pub fn create_cookie(username: &str, role: &str, secret: &[u8]) -> String {
        let expiry = chrono::Utc::now().timestamp() + SESSION_DURATION_SECS;
        let payload = format!("{username}:{role}:{expiry}");
        let key = hmac::Key::new(hmac::HMAC_SHA256, secret);
        let tag = hmac::sign(&key, payload.as_bytes());
        let sig = hex::encode(tag.as_ref());
        let raw = format!("{payload}:{sig}");
        base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(raw.as_bytes())
    }

    /// Verify a session cookie value. Returns `(username, role)` on success.
    pub fn verify_cookie(cookie_value: &str, secret: &[u8]) -> Option<(String, String)> {
        let decoded = base64::engine::general_purpose::URL_SAFE_NO_PAD
            .decode(cookie_value)
            .ok()?;
        let raw = String::from_utf8(decoded).ok()?;
        let parts: Vec<&str> = raw.splitn(4, ':').collect();
        if parts.len() != 4 {
            return None;
        }

        let (username, role, expiry_str, sig_hex) = (parts[0], parts[1], parts[2], parts[3]);

        // Check expiry
        let expiry: i64 = expiry_str.parse().ok()?;
        if chrono::Utc::now().timestamp() > expiry {
            return None;
        }

        // Verify HMAC
        let payload = format!("{username}:{role}:{expiry_str}");
        let key = hmac::Key::new(hmac::HMAC_SHA256, secret);
        let sig_bytes = hex::decode(sig_hex).ok()?;
        hmac::verify(&key, payload.as_bytes(), &sig_bytes).ok()?;

        Some((username.to_string(), role.to_string()))
    }

    /// Build a `Set-Cookie` header value.
    pub fn set_cookie_header(cookie_value: &str) -> String {
        format!(
            "{COOKIE_NAME}={cookie_value}; Path=/; HttpOnly; SameSite=Lax; Max-Age={SESSION_DURATION_SECS}"
        )
    }

    /// Build a `Set-Cookie` header that clears the session.
    pub fn clear_cookie_header() -> String {
        format!("{COOKIE_NAME}=; Path=/; HttpOnly; SameSite=Lax; Max-Age=0")
    }

    /// Extract the session cookie value from a Cookie header string.
    pub fn extract_from_cookie_header(header: &str) -> Option<&str> {
        for part in header.split(';') {
            let trimmed = part.trim();
            if let Some(value) = trimmed.strip_prefix("nagual_session=") {
                if !value.is_empty() {
                    return Some(value);
                }
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_create_and_verify_user() {
        let db = Arc::new(SqliteDb::open_in_memory().unwrap());
        let store = UserStore::new(db).await.unwrap();

        let user = store
            .create_user("alice", "hunter2", "admin")
            .await
            .unwrap();
        assert_eq!(user.username, "alice");
        assert_eq!(user.role, "admin");

        // Correct password
        let verified = store.verify_user("alice", "hunter2").await.unwrap();
        assert!(verified.is_some());
        assert_eq!(verified.unwrap().username, "alice");

        // Wrong password
        let bad = store.verify_user("alice", "wrong").await.unwrap();
        assert!(bad.is_none());

        // Unknown user
        let unknown = store.verify_user("bob", "x").await.unwrap();
        assert!(unknown.is_none());
    }

    #[tokio::test]
    async fn test_list_and_delete_users() {
        let db = Arc::new(SqliteDb::open_in_memory().unwrap());
        let store = UserStore::new(db).await.unwrap();

        assert!(!store.has_users().await.unwrap());

        store.create_user("alice", "pw1", "admin").await.unwrap();
        store.create_user("bob", "pw2", "viewer").await.unwrap();

        assert!(store.has_users().await.unwrap());

        let users = store.list_users().await.unwrap();
        assert_eq!(users.len(), 2);

        assert!(store.delete_user("bob").await.unwrap());
        assert!(!store.delete_user("bob").await.unwrap()); // already gone

        let users = store.list_users().await.unwrap();
        assert_eq!(users.len(), 1);
        assert_eq!(users[0].username, "alice");
    }

    #[test]
    fn test_session_cookie_roundtrip() {
        let secret = session::generate_secret();
        let cookie = session::create_cookie("alice", "admin", &secret);
        let result = session::verify_cookie(&cookie, &secret);
        assert!(result.is_some());
        let (username, role) = result.unwrap();
        assert_eq!(username, "alice");
        assert_eq!(role, "admin");
    }

    #[test]
    fn test_session_cookie_wrong_secret() {
        let secret1 = session::generate_secret();
        let secret2 = session::generate_secret();
        let cookie = session::create_cookie("alice", "admin", &secret1);
        assert!(session::verify_cookie(&cookie, &secret2).is_none());
    }

    #[test]
    fn test_session_cookie_tampered() {
        let secret = session::generate_secret();
        let cookie = session::create_cookie("alice", "admin", &secret);
        let mut tampered = cookie.clone();
        tampered.push('X');
        assert!(session::verify_cookie(&tampered, &secret).is_none());
    }

    #[test]
    fn test_extract_from_cookie_header() {
        let header = "other=foo; nagual_session=abc123; another=bar";
        assert_eq!(
            session::extract_from_cookie_header(header),
            Some("abc123")
        );
        assert!(session::extract_from_cookie_header("other=foo").is_none());
    }

    #[test]
    fn test_set_cookie_header() {
        let header = session::set_cookie_header("test_value");
        assert!(header.contains("nagual_session=test_value"));
        assert!(header.contains("HttpOnly"));
        assert!(header.contains("SameSite=Lax"));
    }
}
