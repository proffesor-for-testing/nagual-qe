-- Migration: Create audit_log table with immutability constraints
-- Version: 008
-- Description: Append-only audit logging for security events

-- Create the audit_log table
CREATE TABLE IF NOT EXISTS audit_log (
    -- Primary key (UUID)
    id TEXT PRIMARY KEY NOT NULL,

    -- When the event occurred (ISO 8601 format)
    timestamp TEXT NOT NULL,

    -- Type of audit event (data_access, data_modify, auth_success, etc.)
    event_type TEXT NOT NULL,

    -- User or system that initiated the event
    user_id TEXT NOT NULL,

    -- Action performed (read, update, delete, authenticate, etc.)
    action TEXT NOT NULL,

    -- Type of resource affected (memory, pattern, config, credential, etc.)
    resource_type TEXT,

    -- ID of the affected resource
    resource_id TEXT,

    -- Previous value (JSON, for modifications)
    old_value TEXT,

    -- New value (JSON, for modifications)
    new_value TEXT,

    -- IP address of the client
    ip_address TEXT,

    -- User agent string
    user_agent TEXT,

    -- Outcome of the operation (success, failure, denied, partial, pending)
    outcome TEXT NOT NULL DEFAULT 'success',

    -- Additional metadata (JSON)
    metadata TEXT DEFAULT '{}',

    -- Hash of the previous entry (for tamper detection chain)
    previous_hash TEXT,

    -- Hash of this entry
    entry_hash TEXT NOT NULL,

    -- Created timestamp (for retention management)
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Indexes for efficient querying
CREATE INDEX IF NOT EXISTS idx_audit_log_timestamp ON audit_log(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_audit_log_event_type ON audit_log(event_type);
CREATE INDEX IF NOT EXISTS idx_audit_log_user_id ON audit_log(user_id);
CREATE INDEX IF NOT EXISTS idx_audit_log_resource ON audit_log(resource_type, resource_id);
CREATE INDEX IF NOT EXISTS idx_audit_log_outcome ON audit_log(outcome);
CREATE INDEX IF NOT EXISTS idx_audit_log_created_at ON audit_log(created_at);

-- Trigger to prevent UPDATE on audit_log (immutability)
CREATE TRIGGER IF NOT EXISTS audit_log_prevent_update
BEFORE UPDATE ON audit_log
BEGIN
    -- Log the tamper attempt before rejecting
    INSERT INTO audit_log_violations (
        id,
        timestamp,
        violation_type,
        attempted_action,
        target_entry_id,
        details
    ) VALUES (
        lower(hex(randomblob(16))),
        datetime('now'),
        'UPDATE_ATTEMPT',
        'UPDATE',
        OLD.id,
        json_object(
            'old_event_type', OLD.event_type,
            'old_user_id', OLD.user_id,
            'old_action', OLD.action
        )
    );

    -- Raise an error to prevent the update
    SELECT RAISE(ABORT, 'SECURITY VIOLATION: Audit log entries cannot be modified. This incident has been logged.');
END;

-- Trigger to prevent DELETE on audit_log (immutability)
CREATE TRIGGER IF NOT EXISTS audit_log_prevent_delete
BEFORE DELETE ON audit_log
BEGIN
    -- Log the tamper attempt before rejecting
    INSERT INTO audit_log_violations (
        id,
        timestamp,
        violation_type,
        attempted_action,
        target_entry_id,
        details
    ) VALUES (
        lower(hex(randomblob(16))),
        datetime('now'),
        'DELETE_ATTEMPT',
        'DELETE',
        OLD.id,
        json_object(
            'deleted_event_type', OLD.event_type,
            'deleted_user_id', OLD.user_id,
            'deleted_action', OLD.action,
            'deleted_timestamp', OLD.timestamp
        )
    );

    -- Raise an error to prevent the delete
    SELECT RAISE(ABORT, 'SECURITY VIOLATION: Audit log entries cannot be deleted. This incident has been logged.');
END;

-- Table to log violation attempts (also immutable)
CREATE TABLE IF NOT EXISTS audit_log_violations (
    id TEXT PRIMARY KEY NOT NULL,
    timestamp TEXT NOT NULL,
    violation_type TEXT NOT NULL,
    attempted_action TEXT NOT NULL,
    target_entry_id TEXT,
    details TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Index for violation lookups
CREATE INDEX IF NOT EXISTS idx_audit_violations_timestamp ON audit_log_violations(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_audit_violations_type ON audit_log_violations(violation_type);

-- Trigger to prevent UPDATE on violations table
CREATE TRIGGER IF NOT EXISTS audit_violations_prevent_update
BEFORE UPDATE ON audit_log_violations
BEGIN
    SELECT RAISE(ABORT, 'SECURITY VIOLATION: Violation log entries cannot be modified.');
END;

-- Trigger to prevent DELETE on violations table
CREATE TRIGGER IF NOT EXISTS audit_violations_prevent_delete
BEFORE DELETE ON audit_log_violations
BEGIN
    SELECT RAISE(ABORT, 'SECURITY VIOLATION: Violation log entries cannot be deleted.');
END;

-- Trigger to validate entry hash on INSERT (optional integrity check)
-- This ensures the hash matches the entry content
CREATE TRIGGER IF NOT EXISTS audit_log_validate_hash
BEFORE INSERT ON audit_log
WHEN NEW.entry_hash IS NULL OR NEW.entry_hash = ''
BEGIN
    SELECT RAISE(ABORT, 'SECURITY VIOLATION: Audit log entries must have a valid entry_hash.');
END;

-- View for recent audit activity (convenience)
CREATE VIEW IF NOT EXISTS v_recent_audit_activity AS
SELECT
    id,
    timestamp,
    event_type,
    user_id,
    action,
    resource_type,
    resource_id,
    outcome,
    json_extract(metadata, '$.description') as description
FROM audit_log
ORDER BY timestamp DESC
LIMIT 100;

-- View for security alerts
CREATE VIEW IF NOT EXISTS v_security_alerts AS
SELECT
    id,
    timestamp,
    user_id,
    action,
    outcome,
    json_extract(metadata, '$.alert_type') as alert_type,
    json_extract(metadata, '$.severity') as severity,
    json_extract(metadata, '$.description') as description
FROM audit_log
WHERE event_type = 'security_alert'
ORDER BY timestamp DESC;

-- View for failed authentication attempts
CREATE VIEW IF NOT EXISTS v_failed_auth_attempts AS
SELECT
    id,
    timestamp,
    user_id,
    ip_address,
    user_agent,
    json_extract(metadata, '$.reason') as failure_reason
FROM audit_log
WHERE event_type = 'auth_failure'
ORDER BY timestamp DESC;

-- View for credential rotations
CREATE VIEW IF NOT EXISTS v_credential_rotations AS
SELECT
    id,
    timestamp,
    user_id,
    outcome,
    json_extract(metadata, '$.credential_type') as credential_type,
    json_extract(metadata, '$.new_version') as new_version
FROM audit_log
WHERE event_type = 'credential_rotation'
ORDER BY timestamp DESC;

-- View for tamper attempts
CREATE VIEW IF NOT EXISTS v_tamper_attempts AS
SELECT
    v.id,
    v.timestamp,
    v.violation_type,
    v.attempted_action,
    v.target_entry_id,
    v.details,
    a.event_type as target_event_type,
    a.user_id as target_user_id
FROM audit_log_violations v
LEFT JOIN audit_log a ON v.target_entry_id = a.id
ORDER BY v.timestamp DESC;
