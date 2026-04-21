//! Specific Health Check Implementations
//!
//! This module provides concrete health check implementations for:
//! - SQLite database connectivity and integrity
//! - PostgreSQL connection pool health
//! - Disk space monitoring
//! - Memory usage monitoring

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use rusqlite::Connection;
use serde_json::json;
use sqlx::postgres::PgPool;
use tokio::sync::Mutex;

use super::{HealthCheck, HealthCheckResult, HealthStatus};

/// Health check for SQLite database
///
/// Checks:
/// - Database file accessibility
/// - Basic query execution
/// - Database integrity (optional)
pub struct SqliteHealthCheck {
    path: PathBuf,
    check_integrity: bool,
    connection: Arc<Mutex<Option<Connection>>>,
}

impl SqliteHealthCheck {
    /// Create a new SQLite health check
    pub fn new(path: impl AsRef<Path>) -> Self {
        Self {
            path: path.as_ref().to_path_buf(),
            check_integrity: false,
            connection: Arc::new(Mutex::new(None)),
        }
    }

    /// Enable integrity check (slow, but thorough)
    pub fn with_integrity_check(mut self) -> Self {
        self.check_integrity = true;
        self
    }

    /// Set a pre-existing connection
    pub async fn with_connection(self, conn: Connection) -> Self {
        let mut guard = self.connection.lock().await;
        *guard = Some(conn);
        drop(guard);
        self
    }

    async fn perform_check(&self) -> HealthCheckResult {
        // Check if database file exists
        if !self.path.exists() {
            return HealthCheckResult::unhealthy(
                "sqlite",
                format!("Database file not found: {}", self.path.display()),
            )
            .with_metadata("path", json!(self.path.to_string_lossy()));
        }

        // Try to open the database
        let conn = match Connection::open(&self.path) {
            Ok(conn) => conn,
            Err(e) => {
                return HealthCheckResult::unhealthy(
                    "sqlite",
                    format!("Failed to open database: {}", e),
                )
                .with_metadata("path", json!(self.path.to_string_lossy()))
                .with_metadata("error", json!(e.to_string()));
            }
        };

        // Basic query test - use query_row for SELECT statements
        match conn.query_row("SELECT 1", [], |_row| Ok(())) {
            Ok(_) => {}
            Err(e) => {
                return HealthCheckResult::unhealthy(
                    "sqlite",
                    format!("Failed to execute query: {}", e),
                )
                .with_metadata("error", json!(e.to_string()));
            }
        }

        // Get database info
        let page_size: i64 = conn
            .query_row("PRAGMA page_size", [], |row| row.get(0))
            .unwrap_or(0);

        let page_count: i64 = conn
            .query_row("PRAGMA page_count", [], |row| row.get(0))
            .unwrap_or(0);

        let db_size = page_size * page_count;

        let mut result = HealthCheckResult::healthy("sqlite", "Database is operational")
            .with_metadata("path", json!(self.path.to_string_lossy()))
            .with_metadata("size_bytes", json!(db_size))
            .with_metadata("page_size", json!(page_size))
            .with_metadata("page_count", json!(page_count));

        // Optional integrity check
        if self.check_integrity {
            match conn.query_row("PRAGMA integrity_check", [], |row| {
                row.get::<_, String>(0)
            }) {
                Ok(integrity_result) if integrity_result == "ok" => {
                    result = result.with_metadata("integrity", json!("ok"));
                }
                Ok(integrity_result) => {
                    return HealthCheckResult::degraded(
                        "sqlite",
                        format!("Integrity check found issues: {}", integrity_result),
                    )
                    .with_metadata("integrity", json!(integrity_result));
                }
                Err(e) => {
                    return HealthCheckResult::degraded(
                        "sqlite",
                        format!("Integrity check failed: {}", e),
                    )
                    .with_metadata("error", json!(e.to_string()));
                }
            }
        }

        result
    }
}

#[async_trait::async_trait]
impl HealthCheck for SqliteHealthCheck {
    fn name(&self) -> &str {
        "sqlite"
    }

    async fn check(&self) -> HealthCheckResult {
        self.perform_check().await
    }

    fn timeout(&self) -> Duration {
        if self.check_integrity {
            Duration::from_secs(60) // Integrity check can be slow
        } else {
            Duration::from_secs(5)
        }
    }

    fn is_critical(&self) -> bool {
        true
    }
}

/// Health check for PostgreSQL connection pool
///
/// Checks:
/// - Pool connectivity
/// - Active/idle connection counts
/// - Query execution time
pub struct PostgresHealthCheck {
    pool: Arc<PgPool>,
    name: String,
    slow_threshold_ms: u64,
}

impl PostgresHealthCheck {
    /// Create a new PostgreSQL health check
    pub fn new(pool: Arc<PgPool>) -> Self {
        Self {
            pool,
            name: "postgres".to_string(),
            slow_threshold_ms: 100,
        }
    }

    /// Set a custom name for this check
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Set the slow query threshold in milliseconds
    pub fn with_slow_threshold(mut self, ms: u64) -> Self {
        self.slow_threshold_ms = ms;
        self
    }

    async fn perform_check(&self) -> HealthCheckResult {
        let start = std::time::Instant::now();

        // Try to execute a simple query
        let result: Result<(i32,), sqlx::Error> =
            sqlx::query_as("SELECT 1").fetch_one(self.pool.as_ref()).await;

        let query_time = start.elapsed();
        let query_time_ms = query_time.as_millis() as u64;

        match result {
            Ok(_) => {
                // Get pool statistics
                let pool_size = self.pool.size();
                let idle_count = self.pool.num_idle();

                let mut check_result = if query_time_ms > self.slow_threshold_ms {
                    HealthCheckResult::degraded(
                        &self.name,
                        format!(
                            "Database responding slowly ({}ms > {}ms threshold)",
                            query_time_ms, self.slow_threshold_ms
                        ),
                    )
                } else {
                    HealthCheckResult::healthy(&self.name, "Database is operational")
                };

                check_result = check_result
                    .with_metadata("query_time_ms", json!(query_time_ms))
                    .with_metadata("pool_size", json!(pool_size))
                    .with_metadata("idle_connections", json!(idle_count))
                    .with_metadata("active_connections", json!(pool_size - idle_count as u32));

                // Warn if pool is nearly exhausted
                if idle_count == 0 && pool_size > 0 {
                    check_result = HealthCheckResult::degraded(
                        &self.name,
                        "Connection pool exhausted (no idle connections)",
                    )
                    .with_metadata("query_time_ms", json!(query_time_ms))
                    .with_metadata("pool_size", json!(pool_size))
                    .with_metadata("idle_connections", json!(0));
                }

                check_result
            }
            Err(e) => HealthCheckResult::unhealthy(
                &self.name,
                format!("Failed to connect to database: {}", e),
            )
            .with_metadata("error", json!(e.to_string())),
        }
    }
}

#[async_trait::async_trait]
impl HealthCheck for PostgresHealthCheck {
    fn name(&self) -> &str {
        &self.name
    }

    async fn check(&self) -> HealthCheckResult {
        self.perform_check().await
    }

    fn timeout(&self) -> Duration {
        Duration::from_secs(10)
    }

    fn is_critical(&self) -> bool {
        true
    }
}

/// Health check for disk space
///
/// Checks available disk space and warns/fails based on thresholds
pub struct DiskHealthCheck {
    path: PathBuf,
    warn_threshold_percent: f64,
    fail_threshold_percent: f64,
}

impl DiskHealthCheck {
    /// Create a new disk health check for the given path
    pub fn new(path: impl AsRef<Path>) -> Self {
        Self {
            path: path.as_ref().to_path_buf(),
            warn_threshold_percent: 90.0,  // Warn at 90% usage
            fail_threshold_percent: 95.0,  // Fail at 95% usage
        }
    }

    /// Set custom thresholds (percent used)
    pub fn with_thresholds(mut self, warn_percent: f64, fail_percent: f64) -> Self {
        self.warn_threshold_percent = warn_percent;
        self.fail_threshold_percent = fail_percent;
        self
    }

    fn check_disk_space(&self) -> HealthCheckResult {
        // Use std::fs to get file system statistics
        #[cfg(unix)]
        {
            use std::ffi::CString;
            use std::mem::MaybeUninit;

            let path_cstr = match CString::new(self.path.to_string_lossy().as_bytes()) {
                Ok(s) => s,
                Err(e) => {
                    return HealthCheckResult::unknown(
                        "disk",
                        format!("Invalid path: {}", e),
                    );
                }
            };

            let mut statvfs = MaybeUninit::<libc::statvfs>::uninit();

            let result = unsafe { libc::statvfs(path_cstr.as_ptr(), statvfs.as_mut_ptr()) };

            if result != 0 {
                return HealthCheckResult::unknown(
                    "disk",
                    format!("Failed to get disk statistics for {}", self.path.display()),
                );
            }

            let statvfs = unsafe { statvfs.assume_init() };

            let block_size = statvfs.f_frsize as u64;
            let total_blocks = statvfs.f_blocks as u64;
            let available_blocks = statvfs.f_bavail as u64;
            let free_blocks = statvfs.f_bfree as u64;

            let total_bytes = total_blocks * block_size;
            let available_bytes = available_blocks * block_size;
            let free_bytes = free_blocks * block_size;
            let used_bytes = total_bytes - free_bytes;

            let usage_percent = if total_bytes > 0 {
                (used_bytes as f64 / total_bytes as f64) * 100.0
            } else {
                0.0
            };

            let status = if usage_percent >= self.fail_threshold_percent {
                HealthStatus::Unhealthy
            } else if usage_percent >= self.warn_threshold_percent {
                HealthStatus::Degraded
            } else {
                HealthStatus::Healthy
            };

            let message = match status {
                HealthStatus::Unhealthy => {
                    format!(
                        "Disk space critically low: {:.1}% used ({} available)",
                        usage_percent,
                        format_bytes(available_bytes)
                    )
                }
                HealthStatus::Degraded => {
                    format!(
                        "Disk space warning: {:.1}% used ({} available)",
                        usage_percent,
                        format_bytes(available_bytes)
                    )
                }
                _ => {
                    format!(
                        "Disk space healthy: {:.1}% used ({} available)",
                        usage_percent,
                        format_bytes(available_bytes)
                    )
                }
            };

            HealthCheckResult {
                component: "disk".to_string(),
                status,
                message,
                checked_at: chrono::Utc::now(),
                duration: Duration::ZERO,
                metadata: [
                    ("path".to_string(), json!(self.path.to_string_lossy())),
                    ("total_bytes".to_string(), json!(total_bytes)),
                    ("available_bytes".to_string(), json!(available_bytes)),
                    ("used_bytes".to_string(), json!(used_bytes)),
                    ("usage_percent".to_string(), json!(usage_percent)),
                ]
                .into_iter()
                .collect(),
            }
        }

        #[cfg(not(unix))]
        {
            HealthCheckResult::unknown(
                "disk",
                "Disk health check not supported on this platform",
            )
        }
    }
}

#[async_trait::async_trait]
impl HealthCheck for DiskHealthCheck {
    fn name(&self) -> &str {
        "disk"
    }

    async fn check(&self) -> HealthCheckResult {
        self.check_disk_space()
    }

    fn timeout(&self) -> Duration {
        Duration::from_secs(5)
    }

    fn is_critical(&self) -> bool {
        false // Non-critical by default
    }
}

/// Health check for memory usage
///
/// Checks available system memory and warns/fails based on thresholds
pub struct MemoryHealthCheck {
    warn_threshold_percent: f64,
    fail_threshold_percent: f64,
}

impl MemoryHealthCheck {
    /// Create a new memory health check with default thresholds
    pub fn new() -> Self {
        Self {
            warn_threshold_percent: 85.0,  // Warn at 85% usage
            fail_threshold_percent: 95.0,  // Fail at 95% usage
        }
    }

    /// Set custom thresholds (percent used)
    pub fn with_thresholds(mut self, warn_percent: f64, fail_percent: f64) -> Self {
        self.warn_threshold_percent = warn_percent;
        self.fail_threshold_percent = fail_percent;
        self
    }

    fn check_memory(&self) -> HealthCheckResult {
        #[cfg(unix)]
        {
            // Try to read from /proc/meminfo on Linux
            if let Ok(contents) = std::fs::read_to_string("/proc/meminfo") {
                let mut total_kb: u64 = 0;
                let mut available_kb: u64 = 0;
                let mut free_kb: u64 = 0;
                let mut buffers_kb: u64 = 0;
                let mut cached_kb: u64 = 0;

                for line in contents.lines() {
                    let parts: Vec<&str> = line.split_whitespace().collect();
                    if parts.len() >= 2 {
                        let value: u64 = parts[1].parse().unwrap_or(0);
                        match parts[0] {
                            "MemTotal:" => total_kb = value,
                            "MemAvailable:" => available_kb = value,
                            "MemFree:" => free_kb = value,
                            "Buffers:" => buffers_kb = value,
                            "Cached:" => cached_kb = value,
                            _ => {}
                        }
                    }
                }

                // If MemAvailable isn't present, estimate it
                if available_kb == 0 {
                    available_kb = free_kb + buffers_kb + cached_kb;
                }

                let total_bytes = total_kb * 1024;
                let available_bytes = available_kb * 1024;
                let used_bytes = total_bytes.saturating_sub(available_bytes);

                let usage_percent = if total_bytes > 0 {
                    (used_bytes as f64 / total_bytes as f64) * 100.0
                } else {
                    0.0
                };

                let status = if usage_percent >= self.fail_threshold_percent {
                    HealthStatus::Unhealthy
                } else if usage_percent >= self.warn_threshold_percent {
                    HealthStatus::Degraded
                } else {
                    HealthStatus::Healthy
                };

                let message = match status {
                    HealthStatus::Unhealthy => {
                        format!(
                            "Memory critically low: {:.1}% used ({} available)",
                            usage_percent,
                            format_bytes(available_bytes)
                        )
                    }
                    HealthStatus::Degraded => {
                        format!(
                            "Memory warning: {:.1}% used ({} available)",
                            usage_percent,
                            format_bytes(available_bytes)
                        )
                    }
                    _ => {
                        format!(
                            "Memory healthy: {:.1}% used ({} available)",
                            usage_percent,
                            format_bytes(available_bytes)
                        )
                    }
                };

                return HealthCheckResult {
                    component: "memory".to_string(),
                    status,
                    message,
                    checked_at: chrono::Utc::now(),
                    duration: Duration::ZERO,
                    metadata: [
                        ("total_bytes".to_string(), json!(total_bytes)),
                        ("available_bytes".to_string(), json!(available_bytes)),
                        ("used_bytes".to_string(), json!(used_bytes)),
                        ("usage_percent".to_string(), json!(usage_percent)),
                    ]
                    .into_iter()
                    .collect(),
                };
            }

            // Fallback: use sysinfo-like approach via libc
            let page_size = unsafe { libc::sysconf(libc::_SC_PAGE_SIZE) };
            let total_pages = unsafe { libc::sysconf(libc::_SC_PHYS_PAGES) };
            // _SC_AVPHYS_PAGES is Linux-only; on macOS, estimate available as 50% of total
            #[cfg(target_os = "linux")]
            let avail_pages = unsafe { libc::sysconf(libc::_SC_AVPHYS_PAGES) };
            #[cfg(not(target_os = "linux"))]
            let avail_pages = total_pages / 2;

            if page_size > 0 && total_pages > 0 {
                let total_bytes = (total_pages as u64) * (page_size as u64);
                let available_bytes = (avail_pages as u64) * (page_size as u64);
                let used_bytes = total_bytes.saturating_sub(available_bytes);

                let usage_percent = (used_bytes as f64 / total_bytes as f64) * 100.0;

                let status = if usage_percent >= self.fail_threshold_percent {
                    HealthStatus::Unhealthy
                } else if usage_percent >= self.warn_threshold_percent {
                    HealthStatus::Degraded
                } else {
                    HealthStatus::Healthy
                };

                let message = format!(
                    "Memory {}: {:.1}% used ({} available)",
                    status,
                    usage_percent,
                    format_bytes(available_bytes)
                );

                return HealthCheckResult {
                    component: "memory".to_string(),
                    status,
                    message,
                    checked_at: chrono::Utc::now(),
                    duration: Duration::ZERO,
                    metadata: [
                        ("total_bytes".to_string(), json!(total_bytes)),
                        ("available_bytes".to_string(), json!(available_bytes)),
                        ("used_bytes".to_string(), json!(used_bytes)),
                        ("usage_percent".to_string(), json!(usage_percent)),
                    ]
                    .into_iter()
                    .collect(),
                };
            }
        }

        HealthCheckResult::unknown("memory", "Memory health check not available on this platform")
    }
}

impl Default for MemoryHealthCheck {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait::async_trait]
impl HealthCheck for MemoryHealthCheck {
    fn name(&self) -> &str {
        "memory"
    }

    async fn check(&self) -> HealthCheckResult {
        self.check_memory()
    }

    fn timeout(&self) -> Duration {
        Duration::from_secs(5)
    }

    fn is_critical(&self) -> bool {
        false // Non-critical by default
    }
}

/// Format bytes into human-readable string
fn format_bytes(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;
    const TB: u64 = GB * 1024;

    if bytes >= TB {
        format!("{:.2} TB", bytes as f64 / TB as f64)
    } else if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} bytes", bytes)
    }
}

/// Composite health check that combines multiple checks
pub struct CompositeHealthCheck {
    name: String,
    checks: Vec<Arc<dyn HealthCheck>>,
    require_all_healthy: bool,
}

impl CompositeHealthCheck {
    /// Create a new composite health check
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            checks: Vec::new(),
            require_all_healthy: true,
        }
    }

    /// Add a check to this composite
    pub fn add_check(mut self, check: Arc<dyn HealthCheck>) -> Self {
        self.checks.push(check);
        self
    }

    /// Set whether all checks must be healthy (default: true)
    pub fn require_all(mut self, require: bool) -> Self {
        self.require_all_healthy = require;
        self
    }
}

#[async_trait::async_trait]
impl HealthCheck for CompositeHealthCheck {
    fn name(&self) -> &str {
        &self.name
    }

    async fn check(&self) -> HealthCheckResult {
        let mut results = Vec::new();
        let mut worst_status = HealthStatus::Healthy;

        for check in &self.checks {
            let result = check.check().await;
            worst_status = worst_status.combine(result.status);
            results.push(result);
        }

        let healthy_count = results
            .iter()
            .filter(|r| r.status == HealthStatus::Healthy)
            .count();

        let message = format!(
            "{}/{} checks healthy",
            healthy_count,
            results.len()
        );

        HealthCheckResult {
            component: self.name.clone(),
            status: worst_status,
            message,
            checked_at: chrono::Utc::now(),
            duration: Duration::ZERO,
            metadata: [
                ("check_count".to_string(), json!(results.len())),
                ("healthy_count".to_string(), json!(healthy_count)),
            ]
            .into_iter()
            .collect(),
        }
    }

    fn timeout(&self) -> Duration {
        // Sum of all check timeouts
        self.checks.iter().map(|c| c.timeout()).sum()
    }

    fn is_critical(&self) -> bool {
        // Critical if any sub-check is critical
        self.checks.iter().any(|c| c.is_critical())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[tokio::test]
    async fn test_sqlite_health_check_missing_file() {
        let check = SqliteHealthCheck::new("/nonexistent/path/database.db");
        let result = check.check().await;
        assert_eq!(result.status, HealthStatus::Unhealthy);
    }

    #[tokio::test]
    async fn test_sqlite_health_check_valid_db() {
        // Use a persistent temp file that won't be auto-deleted
        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join(format!("nagual_test_{}.db", std::process::id()));

        // Create a valid SQLite database
        {
            let conn = Connection::open(&path).unwrap();
            conn.execute("CREATE TABLE test (id INTEGER)", []).unwrap();
            // Ensure write is flushed
            conn.execute("PRAGMA wal_checkpoint(FULL)", []).ok();
        } // conn is dropped here, closing the database

        let check = SqliteHealthCheck::new(&path);
        let result = check.check().await;

        // Clean up
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(format!("{}-wal", path.display()));
        let _ = std::fs::remove_file(format!("{}-shm", path.display()));

        assert_eq!(result.status, HealthStatus::Healthy, "Expected Healthy but got {:?}: {}", result.status, result.message);
    }

    #[tokio::test]
    async fn test_disk_health_check() {
        let check = DiskHealthCheck::new("/");
        let result = check.check().await;
        // Should be healthy or degraded, not unknown on Unix
        #[cfg(unix)]
        assert_ne!(result.status, HealthStatus::Unknown);
    }

    #[tokio::test]
    async fn test_memory_health_check() {
        let check = MemoryHealthCheck::new();
        let result = check.check().await;
        // Should have some result on most platforms
        assert!(result.metadata.contains_key("total_bytes") || result.status == HealthStatus::Unknown);
    }

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(500), "500 bytes");
        assert_eq!(format_bytes(1024), "1.00 KB");
        assert_eq!(format_bytes(1024 * 1024), "1.00 MB");
        assert_eq!(format_bytes(1024 * 1024 * 1024), "1.00 GB");
    }

    #[tokio::test]
    async fn test_composite_health_check() {
        struct AlwaysHealthy;
        struct AlwaysDegraded;

        #[async_trait::async_trait]
        impl HealthCheck for AlwaysHealthy {
            fn name(&self) -> &str { "healthy" }
            async fn check(&self) -> HealthCheckResult {
                HealthCheckResult::healthy("healthy", "OK")
            }
        }

        #[async_trait::async_trait]
        impl HealthCheck for AlwaysDegraded {
            fn name(&self) -> &str { "degraded" }
            async fn check(&self) -> HealthCheckResult {
                HealthCheckResult::degraded("degraded", "Slow")
            }
        }

        let composite = CompositeHealthCheck::new("composite")
            .add_check(Arc::new(AlwaysHealthy))
            .add_check(Arc::new(AlwaysDegraded));

        let result = composite.check().await;
        assert_eq!(result.status, HealthStatus::Degraded);
    }
}
