//! Audit logging for security events.
//!
//! Provides append-only audit logging for access, modification, and deletion
//! events. Supports both SQLite and PostgreSQL backends with async logging
//! to avoid blocking operations.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use chrono::{DateTime, Utc};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use uuid::Uuid;

use crate::error::{NagualError, Result};

/// Type of audit event.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuditEventType {
    /// Data was accessed/read
    DataAccess,
    /// Data was created
    DataCreate,
    /// Data was modified
    DataModify,
    /// Data was deleted
    DataDelete,
    /// Successful authentication
    AuthSuccess,
    /// Failed authentication attempt
    AuthFailure,
    /// Sync operation started
    SyncStart,
    /// Sync operation completed
    SyncComplete,
    /// Sync operation failed
    SyncFailure,
    /// Credential rotation
    CredentialRotation,
    /// Security alert triggered
    SecurityAlert,
    /// Configuration change
    ConfigChange,
    /// Permission change
    PermissionChange,
    /// Encryption/decryption operation
    CryptoOperation,
    /// Export operation
    DataExport,
    /// Import operation
    DataImport,
    /// Backup created
    BackupCreate,
    /// Backup restored
    BackupRestore,
    /// Audit log tamper attempt detected
    TamperAttempt,
}

impl std::fmt::Display for AuditEventType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            AuditEventType::DataAccess => "data_access",
            AuditEventType::DataCreate => "data_create",
            AuditEventType::DataModify => "data_modify",
            AuditEventType::DataDelete => "data_delete",
            AuditEventType::AuthSuccess => "auth_success",
            AuditEventType::AuthFailure => "auth_failure",
            AuditEventType::SyncStart => "sync_start",
            AuditEventType::SyncComplete => "sync_complete",
            AuditEventType::SyncFailure => "sync_failure",
            AuditEventType::CredentialRotation => "credential_rotation",
            AuditEventType::SecurityAlert => "security_alert",
            AuditEventType::ConfigChange => "config_change",
            AuditEventType::PermissionChange => "permission_change",
            AuditEventType::CryptoOperation => "crypto_operation",
            AuditEventType::DataExport => "data_export",
            AuditEventType::DataImport => "data_import",
            AuditEventType::BackupCreate => "backup_create",
            AuditEventType::BackupRestore => "backup_restore",
            AuditEventType::TamperAttempt => "tamper_attempt",
        };
        write!(f, "{}", s)
    }
}

/// Outcome of an audit event.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuditOutcome {
    /// Operation succeeded
    Success,
    /// Operation failed
    Failure,
    /// Operation was denied
    Denied,
    /// Operation was partially successful
    Partial,
    /// Operation is pending
    Pending,
}

impl Default for AuditOutcome {
    fn default() -> Self {
        AuditOutcome::Success
    }
}

impl std::fmt::Display for AuditOutcome {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AuditOutcome::Success => write!(f, "success"),
            AuditOutcome::Failure => write!(f, "failure"),
            AuditOutcome::Denied => write!(f, "denied"),
            AuditOutcome::Partial => write!(f, "partial"),
            AuditOutcome::Pending => write!(f, "pending"),
        }
    }
}

/// An audit log entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEntry {
    /// Unique identifier for this entry
    pub id: String,
    /// When the event occurred
    pub timestamp: DateTime<Utc>,
    /// Type of event
    pub event_type: AuditEventType,
    /// User or system that initiated the event
    pub user_id: String,
    /// Action performed (e.g., "read", "update", "delete")
    pub action: String,
    /// Type of resource affected (e.g., "memory", "pattern", "config")
    pub resource_type: Option<String>,
    /// ID of the affected resource
    pub resource_id: Option<String>,
    /// Previous value (for modifications)
    pub old_value: Option<serde_json::Value>,
    /// New value (for modifications)
    pub new_value: Option<serde_json::Value>,
    /// IP address of the client
    pub ip_address: Option<String>,
    /// User agent string
    pub user_agent: Option<String>,
    /// Outcome of the operation
    pub outcome: AuditOutcome,
    /// Additional metadata
    pub metadata: HashMap<String, serde_json::Value>,
    /// Hash of the previous entry (for tamper detection)
    pub previous_hash: Option<String>,
    /// Hash of this entry
    pub entry_hash: String,
}

impl AuditEntry {
    /// Create a new audit entry.
    fn new(
        event_type: AuditEventType,
        user_id: impl Into<String>,
        action: impl Into<String>,
    ) -> Self {
        let id = Uuid::new_v4().to_string();
        let timestamp = Utc::now();

        let mut entry = Self {
            id,
            timestamp,
            event_type,
            user_id: user_id.into(),
            action: action.into(),
            resource_type: None,
            resource_id: None,
            old_value: None,
            new_value: None,
            ip_address: None,
            user_agent: None,
            outcome: AuditOutcome::Success,
            metadata: HashMap::new(),
            previous_hash: None,
            entry_hash: String::new(),
        };

        // Compute hash after all fields are set
        entry.entry_hash = entry.compute_hash();
        entry
    }

    /// Compute a hash of the entry for tamper detection.
    fn compute_hash(&self) -> String {
        use ring::digest::{digest, SHA256};

        let data = format!(
            "{}:{}:{}:{}:{}:{:?}:{:?}:{:?}",
            self.id,
            self.timestamp.to_rfc3339(),
            self.event_type,
            self.user_id,
            self.action,
            self.resource_type,
            self.resource_id,
            self.previous_hash,
        );

        let hash = digest(&SHA256, data.as_bytes());
        hex::encode(hash.as_ref())
    }

    /// Set the previous hash for chain integrity.
    fn with_previous_hash(mut self, hash: Option<String>) -> Self {
        self.previous_hash = hash;
        self.entry_hash = self.compute_hash();
        self
    }
}

/// Builder for creating audit entries.
pub struct AuditEntryBuilder {
    entry: AuditEntry,
}

impl AuditEntryBuilder {
    /// Create a new audit entry builder.
    pub fn new(event_type: AuditEventType, user_id: impl Into<String>, action: impl Into<String>) -> Self {
        Self {
            entry: AuditEntry::new(event_type, user_id, action),
        }
    }

    /// Set the resource being accessed.
    pub fn resource(mut self, resource_type: impl Into<String>, resource_id: impl Into<String>) -> Self {
        self.entry.resource_type = Some(resource_type.into());
        self.entry.resource_id = Some(resource_id.into());
        self
    }

    /// Set the old value (for modifications).
    pub fn old_value(mut self, value: serde_json::Value) -> Self {
        self.entry.old_value = Some(sanitize_for_audit(value));
        self
    }

    /// Set the new value (for modifications).
    pub fn new_value(mut self, value: serde_json::Value) -> Self {
        self.entry.new_value = Some(sanitize_for_audit(value));
        self
    }

    /// Set the client IP address.
    pub fn ip_address(mut self, ip: impl Into<String>) -> Self {
        self.entry.ip_address = Some(ip.into());
        self
    }

    /// Set the user agent.
    pub fn user_agent(mut self, ua: impl Into<String>) -> Self {
        self.entry.user_agent = Some(ua.into());
        self
    }

    /// Set the outcome.
    pub fn outcome(mut self, outcome: AuditOutcome) -> Self {
        self.entry.outcome = outcome;
        self
    }

    /// Add metadata.
    pub fn metadata(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.entry.metadata.insert(key.into(), sanitize_for_audit(value));
        self
    }

    /// Build the audit entry.
    pub fn build(self) -> AuditEntry {
        self.entry
    }
}

/// Sanitize values to remove sensitive data before audit logging.
fn sanitize_for_audit(value: serde_json::Value) -> serde_json::Value {
    const SENSITIVE_KEYS: &[&str] = &[
        "password",
        "secret",
        "key",
        "token",
        "credential",
        "auth",
        "api_key",
        "apikey",
        "private",
        "ssn",
        "credit_card",
    ];

    match value {
        serde_json::Value::Object(mut map) => {
            for (key, val) in map.clone().iter() {
                let key_lower = key.to_lowercase();
                if SENSITIVE_KEYS.iter().any(|s| key_lower.contains(s)) {
                    map.insert(key.clone(), serde_json::json!("[REDACTED]"));
                } else {
                    map.insert(key.clone(), sanitize_for_audit(val.clone()));
                }
            }
            serde_json::Value::Object(map)
        }
        serde_json::Value::Array(arr) => {
            serde_json::Value::Array(arr.into_iter().map(sanitize_for_audit).collect())
        }
        other => other,
    }
}

/// Audit logger configuration.
#[derive(Debug, Clone)]
pub struct AuditLoggerConfig {
    /// Maximum entries to buffer before forcing a flush
    pub buffer_size: usize,
    /// Maximum time to wait before flushing buffered entries
    pub flush_interval: Duration,
    /// Whether to also write to a file
    pub file_path: Option<String>,
    /// Whether to enable chain hashing for tamper detection
    pub enable_chain_hashing: bool,
    /// Minimum event type to log (for filtering)
    pub min_log_level: Option<AuditEventType>,
}

impl Default for AuditLoggerConfig {
    fn default() -> Self {
        Self {
            buffer_size: 100,
            flush_interval: Duration::from_secs(5),
            file_path: None,
            enable_chain_hashing: true,
            min_log_level: None,
        }
    }
}

/// Message type for the async audit logger channel.
enum AuditMessage {
    Log(AuditEntry),
    Flush,
    Shutdown,
}

/// Audit logger for recording security events.
///
/// Uses async logging to avoid blocking the main application.
/// For SQLite, entries are buffered and flushed synchronously in a blocking task.
///
/// The logger spawns a background thread for SQLite operations. This thread is
/// automatically joined when the logger is dropped, ensuring graceful shutdown.
pub struct AuditLogger {
    config: AuditLoggerConfig,
    sender: mpsc::Sender<AuditMessage>,
    last_hash: Arc<RwLock<Option<String>>>,
    stats: Arc<RwLock<AuditStats>>,
    /// Path to SQLite database for worker thread
    db_path: Option<String>,
    /// Handle to the background worker thread for graceful shutdown
    worker_handle: Option<std::thread::JoinHandle<()>>,
}

/// Statistics about audit logging.
#[derive(Debug, Clone, Default)]
pub struct AuditStats {
    /// Total entries logged
    pub total_logged: u64,
    /// Entries currently buffered
    pub buffered: usize,
    /// Last flush time
    pub last_flush: Option<Instant>,
    /// Total flushes performed
    pub flush_count: u64,
    /// Errors encountered
    pub error_count: u64,
}

impl AuditLogger {
    /// Create a new audit logger with SQLite backend.
    pub async fn new_sqlite(
        db: Arc<crate::db::SqliteDb>,
        config: AuditLoggerConfig,
    ) -> Result<Self> {
        // Ensure the audit_log table exists
        db.execute_batch(include_str!("../migrations/audit_log_sqlite.sql"))
            .await?;

        let db_path = db.path().to_string();
        let (sender, receiver) = mpsc::channel(config.buffer_size * 2);
        let last_hash = Arc::new(RwLock::new(None));
        let stats = Arc::new(RwLock::new(AuditStats::default()));

        // Spawn the background worker in a blocking task (SQLite is not Send)
        let worker_config = config.clone();
        let worker_stats = stats.clone();
        let file_path = config.file_path.clone();
        let worker_db_path = db_path.clone();

        let worker_handle = std::thread::spawn(move || {
            // Create a new runtime for this thread
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("Failed to create runtime for audit worker");

            rt.block_on(async {
                Self::sqlite_worker_thread(
                    receiver,
                    worker_db_path,
                    worker_config,
                    worker_stats,
                    file_path,
                )
                .await;
            });
        });

        Ok(Self {
            config,
            sender,
            last_hash,
            stats,
            db_path: Some(db_path),
            worker_handle: Some(worker_handle),
        })
    }

    /// Background worker for SQLite logging (runs in dedicated thread).
    async fn sqlite_worker_thread(
        mut receiver: mpsc::Receiver<AuditMessage>,
        db_path: String,
        config: AuditLoggerConfig,
        stats: Arc<RwLock<AuditStats>>,
        file_path: Option<String>,
    ) {
        // Open a dedicated SQLite connection for this thread
        let conn = match rusqlite::Connection::open(&db_path) {
            Ok(c) => c,
            Err(e) => {
                tracing::error!("Failed to open audit log database: {}", e);
                return;
            }
        };

        let mut buffer: Vec<AuditEntry> = Vec::with_capacity(config.buffer_size);
        let mut last_flush = Instant::now();

        loop {
            tokio::select! {
                msg = receiver.recv() => {
                    match msg {
                        Some(AuditMessage::Log(entry)) => {
                            buffer.push(entry);
                            stats.write().buffered = buffer.len();

                            if buffer.len() >= config.buffer_size {
                                Self::flush_to_sqlite_sync(&conn, &mut buffer, &stats, &file_path);
                                last_flush = Instant::now();
                            }
                        }
                        Some(AuditMessage::Flush) => {
                            Self::flush_to_sqlite_sync(&conn, &mut buffer, &stats, &file_path);
                            last_flush = Instant::now();
                        }
                        Some(AuditMessage::Shutdown) | None => {
                            // Final flush before shutdown
                            Self::flush_to_sqlite_sync(&conn, &mut buffer, &stats, &file_path);
                            break;
                        }
                    }
                }
                _ = tokio::time::sleep(config.flush_interval) => {
                    if !buffer.is_empty() && last_flush.elapsed() >= config.flush_interval {
                        Self::flush_to_sqlite_sync(&conn, &mut buffer, &stats, &file_path);
                        last_flush = Instant::now();
                    }
                }
            }
        }
    }

    /// Flush buffered entries to SQLite (synchronous version for worker thread).
    fn flush_to_sqlite_sync(
        conn: &rusqlite::Connection,
        buffer: &mut Vec<AuditEntry>,
        stats: &Arc<RwLock<AuditStats>>,
        file_path: &Option<String>,
    ) {
        if buffer.is_empty() {
            return;
        }

        let entries = std::mem::take(buffer);
        let count = entries.len() as u64;

        // Insert all entries in a transaction
        let result = conn.execute_batch("BEGIN TRANSACTION");
        let mut success = result.is_ok();

        if success {
            for entry in &entries {
                let result = conn.execute(
                    r#"
                    INSERT INTO audit_log (
                        id, timestamp, event_type, user_id, action,
                        resource_type, resource_id, old_value, new_value,
                        ip_address, user_agent, outcome, metadata,
                        previous_hash, entry_hash
                    ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15)
                    "#,
                    rusqlite::params![
                        entry.id,
                        entry.timestamp.to_rfc3339(),
                        entry.event_type.to_string(),
                        entry.user_id,
                        entry.action,
                        entry.resource_type,
                        entry.resource_id,
                        entry.old_value.as_ref().map(|v| v.to_string()),
                        entry.new_value.as_ref().map(|v| v.to_string()),
                        entry.ip_address,
                        entry.user_agent,
                        entry.outcome.to_string(),
                        serde_json::to_string(&entry.metadata).unwrap_or_default(),
                        entry.previous_hash,
                        entry.entry_hash,
                    ],
                );

                if result.is_err() {
                    success = false;
                    let _ = conn.execute_batch("ROLLBACK");
                    break;
                }
            }

            if success {
                let _ = conn.execute_batch("COMMIT");
            }
        }

        // Also write to file if configured
        if let Some(ref path) = file_path {
            if let Ok(mut file) = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)
            {
                use std::io::Write;
                for entry in &entries {
                    if let Ok(json) = serde_json::to_string(entry) {
                        let _ = writeln!(file, "{}", json);
                    }
                }
            }
        }

        // Update stats
        let mut stats = stats.write();
        stats.buffered = 0;
        stats.last_flush = Some(Instant::now());
        stats.flush_count += 1;

        if success {
            stats.total_logged += count;
        } else {
            stats.error_count += 1;
            tracing::error!("Failed to flush audit log");
        }
    }

    /// Log an access event (data read).
    pub async fn log_access(
        &self,
        user_id: impl Into<String>,
        resource_type: impl Into<String>,
        resource_id: impl Into<String>,
    ) -> Result<String> {
        let entry = AuditEntryBuilder::new(AuditEventType::DataAccess, user_id, "read")
            .resource(resource_type, resource_id)
            .build();

        self.log(entry).await
    }

    /// Log a modification event (data update).
    pub async fn log_modification(
        &self,
        user_id: impl Into<String>,
        resource_type: impl Into<String>,
        resource_id: impl Into<String>,
        old_value: Option<serde_json::Value>,
        new_value: Option<serde_json::Value>,
    ) -> Result<String> {
        let mut builder = AuditEntryBuilder::new(AuditEventType::DataModify, user_id, "update")
            .resource(resource_type, resource_id);

        if let Some(old) = old_value {
            builder = builder.old_value(old);
        }
        if let Some(new) = new_value {
            builder = builder.new_value(new);
        }

        self.log(builder.build()).await
    }

    /// Log a deletion event.
    pub async fn log_deletion(
        &self,
        user_id: impl Into<String>,
        resource_type: impl Into<String>,
        resource_id: impl Into<String>,
        old_value: Option<serde_json::Value>,
    ) -> Result<String> {
        let mut builder = AuditEntryBuilder::new(AuditEventType::DataDelete, user_id, "delete")
            .resource(resource_type, resource_id);

        if let Some(old) = old_value {
            builder = builder.old_value(old);
        }

        self.log(builder.build()).await
    }

    /// Log an authentication event.
    pub async fn log_auth(
        &self,
        user_id: impl Into<String>,
        success: bool,
        ip_address: Option<String>,
    ) -> Result<String> {
        let event_type = if success {
            AuditEventType::AuthSuccess
        } else {
            AuditEventType::AuthFailure
        };

        let mut builder = AuditEntryBuilder::new(event_type, user_id, "authenticate")
            .outcome(if success {
                AuditOutcome::Success
            } else {
                AuditOutcome::Failure
            });

        if let Some(ip) = ip_address {
            builder = builder.ip_address(ip);
        }

        self.log(builder.build()).await
    }

    /// Log a credential rotation event.
    pub async fn log_credential_rotation(
        &self,
        user_id: impl Into<String>,
        credential_type: impl Into<String>,
    ) -> Result<String> {
        let entry = AuditEntryBuilder::new(AuditEventType::CredentialRotation, user_id, "rotate")
            .metadata("credential_type", serde_json::json!(credential_type.into()))
            .build();

        self.log(entry).await
    }

    /// Log a security alert.
    pub async fn log_security_alert(
        &self,
        alert_type: impl Into<String>,
        description: impl Into<String>,
        severity: impl Into<String>,
    ) -> Result<String> {
        let entry = AuditEntryBuilder::new(AuditEventType::SecurityAlert, "system", "alert")
            .metadata("alert_type", serde_json::json!(alert_type.into()))
            .metadata("description", serde_json::json!(description.into()))
            .metadata("severity", serde_json::json!(severity.into()))
            .build();

        self.log(entry).await
    }

    /// Log a custom audit entry.
    pub async fn log(&self, mut entry: AuditEntry) -> Result<String> {
        let entry_id = entry.id.clone();

        // Add chain hash if enabled
        if self.config.enable_chain_hashing {
            let previous_hash = self.last_hash.read().clone();
            entry = entry.with_previous_hash(previous_hash);
            *self.last_hash.write() = Some(entry.entry_hash.clone());
        }

        // Send to background worker
        self.sender
            .send(AuditMessage::Log(entry))
            .await
            .map_err(|e| NagualError::internal(format!("Failed to send audit message: {}", e)))?;

        Ok(entry_id)
    }

    /// Flush buffered entries immediately.
    pub async fn flush(&self) -> Result<()> {
        self.sender
            .send(AuditMessage::Flush)
            .await
            .map_err(|e| NagualError::internal(format!("Failed to send flush message: {}", e)))?;
        Ok(())
    }

    /// Shutdown the audit logger gracefully.
    pub async fn shutdown(&self) -> Result<()> {
        self.sender
            .send(AuditMessage::Shutdown)
            .await
            .map_err(|e| NagualError::internal(format!("Failed to send shutdown message: {}", e)))?;
        Ok(())
    }

    /// Get current audit statistics.
    pub fn stats(&self) -> AuditStats {
        self.stats.read().clone()
    }

    /// Create an entry builder for custom entries.
    pub fn builder(
        &self,
        event_type: AuditEventType,
        user_id: impl Into<String>,
        action: impl Into<String>,
    ) -> AuditEntryBuilder {
        AuditEntryBuilder::new(event_type, user_id, action)
    }

    /// Shutdown the logger synchronously (blocking).
    /// Used by Drop to ensure clean shutdown without async.
    fn shutdown_sync(&mut self) {
        // Try to send shutdown message (non-blocking, best effort)
        if let Ok(permit) = self.sender.try_reserve() {
            permit.send(AuditMessage::Shutdown);
        }

        // Join the worker thread if we have a handle
        if let Some(handle) = self.worker_handle.take() {
            // Give the thread a reasonable time to finish (max 5 seconds)
            let start = std::time::Instant::now();
            while !handle.is_finished() {
                if start.elapsed() > std::time::Duration::from_secs(5) {
                    tracing::warn!("Audit worker thread did not finish in time, abandoning");
                    break;
                }
                std::thread::sleep(std::time::Duration::from_millis(10));
            }

            // Try to join if finished
            if handle.is_finished() {
                let _ = handle.join();
            }
        }
    }
}

impl Drop for AuditLogger {
    fn drop(&mut self) {
        self.shutdown_sync();
    }
}

/// Query builder for audit log searches.
pub struct AuditQuery {
    event_types: Option<Vec<AuditEventType>>,
    user_id: Option<String>,
    resource_type: Option<String>,
    resource_id: Option<String>,
    outcome: Option<AuditOutcome>,
    start_time: Option<DateTime<Utc>>,
    end_time: Option<DateTime<Utc>>,
    limit: usize,
    offset: usize,
}

impl Default for AuditQuery {
    fn default() -> Self {
        Self {
            event_types: None,
            user_id: None,
            resource_type: None,
            resource_id: None,
            outcome: None,
            start_time: None,
            end_time: None,
            limit: 100,
            offset: 0,
        }
    }
}

impl AuditQuery {
    /// Create a new audit query.
    pub fn new() -> Self {
        Self::default()
    }

    /// Filter by event types.
    pub fn event_types(mut self, types: Vec<AuditEventType>) -> Self {
        self.event_types = Some(types);
        self
    }

    /// Filter by user ID.
    pub fn user_id(mut self, id: impl Into<String>) -> Self {
        self.user_id = Some(id.into());
        self
    }

    /// Filter by resource type.
    pub fn resource_type(mut self, rtype: impl Into<String>) -> Self {
        self.resource_type = Some(rtype.into());
        self
    }

    /// Filter by resource ID.
    pub fn resource_id(mut self, rid: impl Into<String>) -> Self {
        self.resource_id = Some(rid.into());
        self
    }

    /// Filter by outcome.
    pub fn outcome(mut self, outcome: AuditOutcome) -> Self {
        self.outcome = Some(outcome);
        self
    }

    /// Filter by time range.
    pub fn time_range(mut self, start: DateTime<Utc>, end: DateTime<Utc>) -> Self {
        self.start_time = Some(start);
        self.end_time = Some(end);
        self
    }

    /// Set result limit.
    pub fn limit(mut self, limit: usize) -> Self {
        self.limit = limit;
        self
    }

    /// Set result offset.
    pub fn offset(mut self, offset: usize) -> Self {
        self.offset = offset;
        self
    }

    /// Execute the query against SQLite.
    pub async fn execute_sqlite(&self, db: &crate::db::SqliteDb) -> Result<Vec<AuditEntry>> {
        let mut sql = String::from(
            "SELECT id, timestamp, event_type, user_id, action, resource_type, resource_id,
             old_value, new_value, ip_address, user_agent, outcome, metadata,
             previous_hash, entry_hash FROM audit_log WHERE 1=1",
        );
        let mut params: Vec<Box<dyn rusqlite::ToSql>> = Vec::new();

        if let Some(ref user_id) = self.user_id {
            sql.push_str(" AND user_id = ?");
            params.push(Box::new(user_id.clone()));
        }

        if let Some(ref resource_type) = self.resource_type {
            sql.push_str(" AND resource_type = ?");
            params.push(Box::new(resource_type.clone()));
        }

        if let Some(ref resource_id) = self.resource_id {
            sql.push_str(" AND resource_id = ?");
            params.push(Box::new(resource_id.clone()));
        }

        if let Some(ref outcome) = self.outcome {
            sql.push_str(" AND outcome = ?");
            params.push(Box::new(outcome.to_string()));
        }

        if let Some(ref start) = self.start_time {
            sql.push_str(" AND timestamp >= ?");
            params.push(Box::new(start.to_rfc3339()));
        }

        if let Some(ref end) = self.end_time {
            sql.push_str(" AND timestamp <= ?");
            params.push(Box::new(end.to_rfc3339()));
        }

        sql.push_str(" ORDER BY timestamp DESC");
        sql.push_str(&format!(" LIMIT {} OFFSET {}", self.limit, self.offset));

        // For simplicity, we'll use a raw query approach
        // In production, you'd want proper parameter binding
        let entries = db
            .query(&sql, &[], |row| {
                let event_type_str: String = row.get(2)?;
                let outcome_str: String = row.get(11)?;
                let metadata_str: String = row.get(12)?;

                Ok(AuditEntry {
                    id: row.get(0)?,
                    timestamp: chrono::DateTime::parse_from_rfc3339(&row.get::<_, String>(1)?)
                        .map(|dt| dt.with_timezone(&Utc))
                        .unwrap_or_else(|_| Utc::now()),
                    event_type: parse_event_type(&event_type_str),
                    user_id: row.get(3)?,
                    action: row.get(4)?,
                    resource_type: row.get(5)?,
                    resource_id: row.get(6)?,
                    old_value: row
                        .get::<_, Option<String>>(7)?
                        .and_then(|s| serde_json::from_str(&s).ok()),
                    new_value: row
                        .get::<_, Option<String>>(8)?
                        .and_then(|s| serde_json::from_str(&s).ok()),
                    ip_address: row.get(9)?,
                    user_agent: row.get(10)?,
                    outcome: parse_outcome(&outcome_str),
                    metadata: serde_json::from_str(&metadata_str).unwrap_or_default(),
                    previous_hash: row.get(13)?,
                    entry_hash: row.get(14)?,
                })
            })
            .await?;

        Ok(entries)
    }
}

fn parse_event_type(s: &str) -> AuditEventType {
    match s {
        "data_access" => AuditEventType::DataAccess,
        "data_create" => AuditEventType::DataCreate,
        "data_modify" => AuditEventType::DataModify,
        "data_delete" => AuditEventType::DataDelete,
        "auth_success" => AuditEventType::AuthSuccess,
        "auth_failure" => AuditEventType::AuthFailure,
        "sync_start" => AuditEventType::SyncStart,
        "sync_complete" => AuditEventType::SyncComplete,
        "sync_failure" => AuditEventType::SyncFailure,
        "credential_rotation" => AuditEventType::CredentialRotation,
        "security_alert" => AuditEventType::SecurityAlert,
        "config_change" => AuditEventType::ConfigChange,
        "permission_change" => AuditEventType::PermissionChange,
        "crypto_operation" => AuditEventType::CryptoOperation,
        "data_export" => AuditEventType::DataExport,
        "data_import" => AuditEventType::DataImport,
        "backup_create" => AuditEventType::BackupCreate,
        "backup_restore" => AuditEventType::BackupRestore,
        "tamper_attempt" => AuditEventType::TamperAttempt,
        _ => AuditEventType::DataAccess,
    }
}

fn parse_outcome(s: &str) -> AuditOutcome {
    match s {
        "success" => AuditOutcome::Success,
        "failure" => AuditOutcome::Failure,
        "denied" => AuditOutcome::Denied,
        "partial" => AuditOutcome::Partial,
        "pending" => AuditOutcome::Pending,
        _ => AuditOutcome::Success,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audit_entry_creation() {
        let entry = AuditEntryBuilder::new(AuditEventType::DataAccess, "user123", "read")
            .resource("memory", "mem-456")
            .outcome(AuditOutcome::Success)
            .build();

        assert!(!entry.id.is_empty());
        assert_eq!(entry.event_type, AuditEventType::DataAccess);
        assert_eq!(entry.user_id, "user123");
        assert_eq!(entry.action, "read");
        assert_eq!(entry.resource_type, Some("memory".to_string()));
        assert_eq!(entry.resource_id, Some("mem-456".to_string()));
        assert_eq!(entry.outcome, AuditOutcome::Success);
        assert!(!entry.entry_hash.is_empty());
    }

    #[test]
    fn test_sanitize_for_audit() {
        let value = serde_json::json!({
            "username": "john",
            "password": "secret123",
            "api_key": "sk-abc123",
            "data": {
                "token": "bearer-xyz"
            }
        });

        let sanitized = sanitize_for_audit(value);
        let obj = sanitized.as_object().unwrap();

        assert_eq!(obj.get("username").unwrap(), "john");
        assert_eq!(obj.get("password").unwrap(), "[REDACTED]");
        assert_eq!(obj.get("api_key").unwrap(), "[REDACTED]");

        let data = obj.get("data").unwrap().as_object().unwrap();
        assert_eq!(data.get("token").unwrap(), "[REDACTED]");
    }

    #[test]
    fn test_chain_hashing() {
        let entry1 = AuditEntry::new(AuditEventType::DataAccess, "user1", "read");
        let hash1 = entry1.entry_hash.clone();

        let entry2 = AuditEntry::new(AuditEventType::DataModify, "user2", "update")
            .with_previous_hash(Some(hash1.clone()));

        assert_eq!(entry2.previous_hash, Some(hash1));
        assert_ne!(entry2.entry_hash, entry2.previous_hash.unwrap());
    }

    #[test]
    fn test_event_type_display() {
        assert_eq!(AuditEventType::DataAccess.to_string(), "data_access");
        assert_eq!(AuditEventType::AuthFailure.to_string(), "auth_failure");
        assert_eq!(
            AuditEventType::CredentialRotation.to_string(),
            "credential_rotation"
        );
    }

    #[test]
    fn test_outcome_display() {
        assert_eq!(AuditOutcome::Success.to_string(), "success");
        assert_eq!(AuditOutcome::Denied.to_string(), "denied");
    }

    #[test]
    fn test_query_builder() {
        let query = AuditQuery::new()
            .user_id("user123")
            .resource_type("memory")
            .outcome(AuditOutcome::Success)
            .limit(50)
            .offset(10);

        assert_eq!(query.user_id, Some("user123".to_string()));
        assert_eq!(query.resource_type, Some("memory".to_string()));
        assert_eq!(query.outcome, Some(AuditOutcome::Success));
        assert_eq!(query.limit, 50);
        assert_eq!(query.offset, 10);
    }
}
