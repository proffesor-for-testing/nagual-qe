//! PostgreSQL configuration with TLS support.
//!
//! Provides secure connection configuration for PostgreSQL with:
//! - TLS/SSL encryption using rustls
//! - Certificate verification options
//! - Connection string builder with sslmode support
//! - Connection pooling configuration

use std::path::PathBuf;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use sqlx::postgres::{PgConnectOptions, PgPool, PgPoolOptions, PgSslMode};

use crate::error::DatabaseError;

/// TLS verification mode for PostgreSQL connections.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TlsVerifyMode {
    /// Disable TLS (not recommended for production).
    Disable,
    /// Prefer TLS but allow unencrypted connections.
    #[default]
    Prefer,
    /// Require TLS but don't verify certificates.
    Require,
    /// Require TLS and verify CA certificate.
    VerifyCa,
    /// Require TLS and verify CA certificate and hostname.
    VerifyFull,
}

impl TlsVerifyMode {
    /// Convert to sqlx PgSslMode.
    pub fn to_pg_ssl_mode(&self) -> PgSslMode {
        match self {
            TlsVerifyMode::Disable => PgSslMode::Disable,
            TlsVerifyMode::Prefer => PgSslMode::Prefer,
            TlsVerifyMode::Require => PgSslMode::Require,
            TlsVerifyMode::VerifyCa => PgSslMode::VerifyCa,
            TlsVerifyMode::VerifyFull => PgSslMode::VerifyFull,
        }
    }
}

/// TLS configuration for PostgreSQL connections.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TlsConfig {
    /// TLS verification mode.
    pub verify_mode: TlsVerifyMode,
    /// Path to CA certificate file (PEM format).
    pub ca_cert_path: Option<PathBuf>,
    /// Path to client certificate file (PEM format).
    pub client_cert_path: Option<PathBuf>,
    /// Path to client key file (PEM format).
    pub client_key_path: Option<PathBuf>,
    /// Accept invalid certificates (for development only).
    #[serde(default)]
    pub accept_invalid_certs: bool,
    /// Accept invalid hostnames (for development only).
    #[serde(default)]
    pub accept_invalid_hostnames: bool,
}

impl TlsConfig {
    /// Create a new TLS configuration with the specified verify mode.
    pub fn new(verify_mode: TlsVerifyMode) -> Self {
        Self {
            verify_mode,
            ..Default::default()
        }
    }

    /// Create a TLS configuration that disables encryption (not recommended).
    pub fn disabled() -> Self {
        Self::new(TlsVerifyMode::Disable)
    }

    /// Create a TLS configuration that requires encryption.
    pub fn required() -> Self {
        Self::new(TlsVerifyMode::Require)
    }

    /// Create a TLS configuration with full verification.
    pub fn verify_full(ca_cert_path: impl Into<PathBuf>) -> Self {
        Self {
            verify_mode: TlsVerifyMode::VerifyFull,
            ca_cert_path: Some(ca_cert_path.into()),
            ..Default::default()
        }
    }

    /// Set the CA certificate path.
    pub fn with_ca_cert(mut self, path: impl Into<PathBuf>) -> Self {
        self.ca_cert_path = Some(path.into());
        self
    }

    /// Set the client certificate and key paths.
    pub fn with_client_cert(
        mut self,
        cert_path: impl Into<PathBuf>,
        key_path: impl Into<PathBuf>,
    ) -> Self {
        self.client_cert_path = Some(cert_path.into());
        self.client_key_path = Some(key_path.into());
        self
    }

    /// Allow invalid certificates (development only).
    pub fn accept_invalid_certs(mut self) -> Self {
        self.accept_invalid_certs = true;
        self
    }

    /// Allow invalid hostnames (development only).
    pub fn accept_invalid_hostnames(mut self) -> Self {
        self.accept_invalid_hostnames = true;
        self
    }
}

/// PostgreSQL connection configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PostgresConfig {
    /// PostgreSQL host.
    pub host: String,
    /// PostgreSQL port.
    pub port: u16,
    /// Database name.
    pub database: String,
    /// Username for authentication.
    pub username: String,
    /// Password for authentication (consider using environment variables).
    #[serde(skip_serializing)]
    pub password: Option<String>,
    /// TLS configuration.
    #[serde(default)]
    pub tls: TlsConfig,
    /// Connection pool configuration.
    #[serde(default)]
    pub pool: PoolConfig,
    /// Application name for connection identification.
    pub application_name: Option<String>,
    /// Connection timeout in seconds.
    #[serde(default = "default_connect_timeout")]
    pub connect_timeout_secs: u64,
    /// Statement timeout in seconds (0 = no timeout).
    #[serde(default)]
    pub statement_timeout_secs: u64,
    /// Schema to use (defaults to "public").
    pub schema: Option<String>,
}

fn default_connect_timeout() -> u64 {
    30
}

impl Default for PostgresConfig {
    fn default() -> Self {
        Self {
            host: "localhost".to_string(),
            port: 5432,
            database: "nagual".to_string(),
            username: "nagual".to_string(),
            password: None,
            tls: TlsConfig::default(),
            pool: PoolConfig::default(),
            application_name: Some("nagual".to_string()),
            connect_timeout_secs: default_connect_timeout(),
            statement_timeout_secs: 0,
            schema: None,
        }
    }
}

impl PostgresConfig {
    /// Create a new PostgreSQL configuration.
    pub fn new(host: impl Into<String>, database: impl Into<String>) -> Self {
        Self {
            host: host.into(),
            database: database.into(),
            ..Default::default()
        }
    }

    /// Create configuration from environment variables.
    ///
    /// Reads the following environment variables:
    /// - `DATABASE_URL`: Full connection string (if set, other vars are ignored)
    /// - `POSTGRES_HOST`: Host (default: localhost)
    /// - `POSTGRES_PORT`: Port (default: 5432)
    /// - `POSTGRES_DB`: Database name (default: nagual)
    /// - `POSTGRES_USER`: Username (default: nagual)
    /// - `POSTGRES_PASSWORD`: Password
    /// - `POSTGRES_SSLMODE`: SSL mode (disable, prefer, require, verify-ca, verify-full)
    pub fn from_env() -> Self {
        let tls_mode = std::env::var("POSTGRES_SSLMODE")
            .ok()
            .map(|s| match s.to_lowercase().as_str() {
                "disable" => TlsVerifyMode::Disable,
                "prefer" => TlsVerifyMode::Prefer,
                "require" => TlsVerifyMode::Require,
                "verify-ca" => TlsVerifyMode::VerifyCa,
                "verify-full" => TlsVerifyMode::VerifyFull,
                _ => TlsVerifyMode::Prefer,
            })
            .unwrap_or_default();

        Self {
            host: std::env::var("POSTGRES_HOST").unwrap_or_else(|_| "localhost".to_string()),
            port: std::env::var("POSTGRES_PORT")
                .ok()
                .and_then(|p| p.parse().ok())
                .unwrap_or(5432),
            database: std::env::var("POSTGRES_DB").unwrap_or_else(|_| "nagual".to_string()),
            username: std::env::var("POSTGRES_USER").unwrap_or_else(|_| "nagual".to_string()),
            password: std::env::var("POSTGRES_PASSWORD").ok(),
            tls: TlsConfig::new(tls_mode),
            pool: PoolConfig::default(),
            application_name: Some("nagual".to_string()),
            connect_timeout_secs: default_connect_timeout(),
            statement_timeout_secs: 0,
            schema: std::env::var("POSTGRES_SCHEMA").ok(),
        }
    }

    /// Set the port.
    pub fn with_port(mut self, port: u16) -> Self {
        self.port = port;
        self
    }

    /// Set the username.
    pub fn with_username(mut self, username: impl Into<String>) -> Self {
        self.username = username.into();
        self
    }

    /// Set the password.
    pub fn with_password(mut self, password: impl Into<String>) -> Self {
        self.password = Some(password.into());
        self
    }

    /// Set the TLS configuration.
    pub fn with_tls(mut self, tls: TlsConfig) -> Self {
        self.tls = tls;
        self
    }

    /// Set the pool configuration.
    pub fn with_pool(mut self, pool: PoolConfig) -> Self {
        self.pool = pool;
        self
    }

    /// Set the application name.
    pub fn with_application_name(mut self, name: impl Into<String>) -> Self {
        self.application_name = Some(name.into());
        self
    }

    /// Set the schema.
    pub fn with_schema(mut self, schema: impl Into<String>) -> Self {
        self.schema = Some(schema.into());
        self
    }

    /// Build a connection string from the configuration.
    ///
    /// Format: `postgresql://user:password@host:port/database?sslmode=mode`
    pub fn connection_string(&self) -> String {
        let mut url = format!(
            "postgresql://{}",
            self.username
        );

        if let Some(ref password) = self.password {
            url.push(':');
            url.push_str(&urlencoding::encode(password));
        }

        url.push('@');
        url.push_str(&self.host);
        url.push(':');
        url.push_str(&self.port.to_string());
        url.push('/');
        url.push_str(&self.database);

        // Add query parameters
        let mut params = Vec::new();

        // SSL mode
        let sslmode = match self.tls.verify_mode {
            TlsVerifyMode::Disable => "disable",
            TlsVerifyMode::Prefer => "prefer",
            TlsVerifyMode::Require => "require",
            TlsVerifyMode::VerifyCa => "verify-ca",
            TlsVerifyMode::VerifyFull => "verify-full",
        };
        params.push(format!("sslmode={}", sslmode));

        // Application name
        if let Some(ref app_name) = self.application_name {
            params.push(format!("application_name={}", urlencoding::encode(app_name)));
        }

        // Connect timeout
        if self.connect_timeout_secs > 0 {
            params.push(format!("connect_timeout={}", self.connect_timeout_secs));
        }

        // Statement timeout
        if self.statement_timeout_secs > 0 {
            params.push(format!("statement_timeout={}000", self.statement_timeout_secs));
        }

        // Schema (search_path)
        if let Some(ref schema) = self.schema {
            params.push(format!("options=-c%20search_path%3D{}", schema));
        }

        if !params.is_empty() {
            url.push('?');
            url.push_str(&params.join("&"));
        }

        url
    }

    /// Build a connection string with the password masked (for logging).
    pub fn connection_string_masked(&self) -> String {
        let full = self.connection_string();
        if self.password.is_some() {
            // Mask password in URL
            if let Some(at_pos) = full.find('@') {
                if let Some(colon_pos) = full[..at_pos].rfind(':') {
                    let prefix = &full[..colon_pos + 1];
                    let suffix = &full[at_pos..];
                    return format!("{}****{}", prefix, suffix);
                }
            }
        }
        full
    }

    /// Build sqlx PgConnectOptions from the configuration.
    pub fn connect_options(&self) -> std::result::Result<PgConnectOptions, DatabaseError> {
        let mut options = PgConnectOptions::new()
            .host(&self.host)
            .port(self.port)
            .database(&self.database)
            .username(&self.username)
            .ssl_mode(self.tls.verify_mode.to_pg_ssl_mode());

        if let Some(ref password) = self.password {
            options = options.password(password);
        }

        if let Some(ref app_name) = self.application_name {
            options = options.application_name(app_name);
        }

        // Set SSL root cert if provided
        if let Some(ref ca_path) = self.tls.ca_cert_path {
            options = options.ssl_root_cert(ca_path);
        }

        // Set client certificate if provided
        if let Some(ref cert_path) = self.tls.client_cert_path {
            options = options.ssl_client_cert(cert_path);
        }

        // Set client key if provided
        if let Some(ref key_path) = self.tls.client_key_path {
            options = options.ssl_client_key(key_path);
        }

        // Set statement timeout if specified
        if self.statement_timeout_secs > 0 {
            options = options.options([(
                "statement_timeout",
                format!("{}s", self.statement_timeout_secs),
            )]);
        }

        Ok(options)
    }

    /// Create a connection pool from the configuration.
    pub async fn create_pool(&self) -> std::result::Result<PgPool, DatabaseError> {
        let options = self.connect_options()?;

        let pool = PgPoolOptions::new()
            .max_connections(self.pool.max_connections)
            .min_connections(self.pool.min_connections)
            .acquire_timeout(Duration::from_secs(self.pool.acquire_timeout_secs))
            .idle_timeout(self.pool.idle_timeout_secs.map(Duration::from_secs))
            .max_lifetime(self.pool.max_lifetime_secs.map(Duration::from_secs))
            .test_before_acquire(self.pool.test_before_acquire)
            .connect_with(options)
            .await
            .map_err(DatabaseError::from)?;

        Ok(pool)
    }
}

/// Connection pool configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoolConfig {
    /// Maximum number of connections in the pool.
    pub max_connections: u32,
    /// Minimum number of connections to maintain.
    pub min_connections: u32,
    /// Timeout for acquiring a connection from the pool (seconds).
    pub acquire_timeout_secs: u64,
    /// Maximum time a connection can be idle before being closed (seconds).
    pub idle_timeout_secs: Option<u64>,
    /// Maximum lifetime of a connection (seconds).
    pub max_lifetime_secs: Option<u64>,
    /// Test connections before returning them from the pool.
    pub test_before_acquire: bool,
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            max_connections: 10,
            min_connections: 1,
            acquire_timeout_secs: 30,
            idle_timeout_secs: Some(600), // 10 minutes
            max_lifetime_secs: Some(1800), // 30 minutes
            test_before_acquire: true,
        }
    }
}

impl PoolConfig {
    /// Create a pool configuration for development.
    pub fn development() -> Self {
        Self {
            max_connections: 5,
            min_connections: 1,
            acquire_timeout_secs: 10,
            idle_timeout_secs: Some(300),
            max_lifetime_secs: Some(900),
            test_before_acquire: true,
        }
    }

    /// Create a pool configuration for production.
    pub fn production() -> Self {
        Self {
            max_connections: 20,
            min_connections: 5,
            acquire_timeout_secs: 30,
            idle_timeout_secs: Some(600),
            max_lifetime_secs: Some(3600),
            test_before_acquire: true,
        }
    }
}

/// URL encoding helper module.
mod urlencoding {
    /// Percent-encode a string for use in URLs.
    pub fn encode(input: &str) -> String {
        let mut result = String::with_capacity(input.len() * 3);
        for byte in input.bytes() {
            match byte {
                b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                    result.push(byte as char);
                }
                _ => {
                    result.push('%');
                    result.push_str(&format!("{:02X}", byte));
                }
            }
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_postgres_config_default() {
        let config = PostgresConfig::default();
        assert_eq!(config.host, "localhost");
        assert_eq!(config.port, 5432);
        assert_eq!(config.database, "nagual");
    }

    #[test]
    fn test_connection_string_basic() {
        let config = PostgresConfig::new("db.example.com", "mydb")
            .with_username("user")
            .with_password("secret");

        let url = config.connection_string();
        assert!(url.starts_with("postgresql://user:secret@db.example.com:5432/mydb"));
    }

    #[test]
    fn test_connection_string_masked() {
        let config = PostgresConfig::new("localhost", "nagual")
            .with_username("user")
            .with_password("supersecret");

        let masked = config.connection_string_masked();
        assert!(!masked.contains("supersecret"));
        assert!(masked.contains("****"));
    }

    #[test]
    fn test_connection_string_special_chars() {
        let config = PostgresConfig::new("localhost", "nagual")
            .with_username("user")
            .with_password("p@ss:word/test");

        let url = config.connection_string();
        assert!(url.contains("p%40ss%3Aword%2Ftest"));
    }

    #[test]
    fn test_tls_config() {
        let tls = TlsConfig::verify_full("/path/to/ca.pem")
            .with_client_cert("/path/to/client.pem", "/path/to/client.key");

        assert_eq!(tls.verify_mode, TlsVerifyMode::VerifyFull);
        assert!(tls.ca_cert_path.is_some());
        assert!(tls.client_cert_path.is_some());
        assert!(tls.client_key_path.is_some());
    }

    #[test]
    fn test_pool_config_profiles() {
        let dev = PoolConfig::development();
        let prod = PoolConfig::production();

        assert!(prod.max_connections > dev.max_connections);
        assert!(prod.min_connections > dev.min_connections);
    }

    #[test]
    fn test_tls_verify_mode_conversion() {
        // Compare debug representations since PgSslMode doesn't implement PartialEq
        assert_eq!(
            format!("{:?}", TlsVerifyMode::Disable.to_pg_ssl_mode()),
            format!("{:?}", PgSslMode::Disable)
        );
        assert_eq!(
            format!("{:?}", TlsVerifyMode::Require.to_pg_ssl_mode()),
            format!("{:?}", PgSslMode::Require)
        );
        assert_eq!(
            format!("{:?}", TlsVerifyMode::VerifyFull.to_pg_ssl_mode()),
            format!("{:?}", PgSslMode::VerifyFull)
        );
    }
}
