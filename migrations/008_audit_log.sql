-- Migration: 008_audit_log
-- Description: Create audit_log table with immutability constraints for PostgreSQL
-- Version: 008

-- Create the audit_log table
CREATE TABLE IF NOT EXISTS audit_log (
    -- Primary key (UUID)
    id TEXT PRIMARY KEY NOT NULL,

    -- When the event occurred
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Type of audit event
    event_type TEXT NOT NULL,

    -- User or system that initiated the event
    user_id TEXT NOT NULL,

    -- Action performed
    action TEXT NOT NULL,

    -- Type of resource affected
    resource_type TEXT,

    -- ID of the affected resource
    resource_id TEXT,

    -- Previous value (JSONB, for modifications)
    old_value JSONB,

    -- New value (JSONB, for modifications)
    new_value JSONB,

    -- IP address of the client
    ip_address TEXT,

    -- User agent string
    user_agent TEXT,

    -- Outcome of the operation
    outcome TEXT NOT NULL DEFAULT 'success',

    -- Additional metadata (JSONB)
    metadata JSONB DEFAULT '{}',

    -- Hash of the previous entry (for tamper detection chain)
    previous_hash TEXT,

    -- Hash of this entry
    entry_hash TEXT NOT NULL,

    -- Created timestamp (for retention management)
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes for efficient querying
CREATE INDEX IF NOT EXISTS idx_audit_log_timestamp ON audit_log(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_audit_log_event_type ON audit_log(event_type);
CREATE INDEX IF NOT EXISTS idx_audit_log_user_id ON audit_log(user_id);
CREATE INDEX IF NOT EXISTS idx_audit_log_resource ON audit_log(resource_type, resource_id);
CREATE INDEX IF NOT EXISTS idx_audit_log_outcome ON audit_log(outcome);
CREATE INDEX IF NOT EXISTS idx_audit_log_created_at ON audit_log(created_at);

-- GIN index for JSONB metadata queries
CREATE INDEX IF NOT EXISTS idx_audit_log_metadata ON audit_log USING GIN (metadata);

-- Table to log violation attempts
CREATE TABLE IF NOT EXISTS audit_log_violations (
    id TEXT PRIMARY KEY NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    violation_type TEXT NOT NULL,
    attempted_action TEXT NOT NULL,
    target_entry_id TEXT REFERENCES audit_log(id),
    details JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Index for violation lookups
CREATE INDEX IF NOT EXISTS idx_audit_violations_timestamp ON audit_log_violations(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_audit_violations_type ON audit_log_violations(violation_type);

-- Function to log tamper attempts and prevent modification
CREATE OR REPLACE FUNCTION prevent_audit_modification()
RETURNS TRIGGER AS $$
BEGIN
    -- Log the tamper attempt
    INSERT INTO audit_log_violations (
        id,
        timestamp,
        violation_type,
        attempted_action,
        target_entry_id,
        details
    ) VALUES (
        gen_random_uuid()::text,
        NOW(),
        TG_OP || '_ATTEMPT',
        TG_OP,
        OLD.id,
        jsonb_build_object(
            'event_type', OLD.event_type,
            'user_id', OLD.user_id,
            'action', OLD.action,
            'timestamp', OLD.timestamp
        )
    );

    -- Raise an exception to prevent the modification
    RAISE EXCEPTION 'SECURITY VIOLATION: Audit log entries cannot be modified or deleted. This incident has been logged.';
END;
$$ LANGUAGE plpgsql;

-- Trigger to prevent UPDATE on audit_log
DROP TRIGGER IF EXISTS audit_log_prevent_update ON audit_log;
CREATE TRIGGER audit_log_prevent_update
BEFORE UPDATE ON audit_log
FOR EACH ROW
EXECUTE FUNCTION prevent_audit_modification();

-- Trigger to prevent DELETE on audit_log
DROP TRIGGER IF EXISTS audit_log_prevent_delete ON audit_log;
CREATE TRIGGER audit_log_prevent_delete
BEFORE DELETE ON audit_log
FOR EACH ROW
EXECUTE FUNCTION prevent_audit_modification();

-- Function to prevent modification of violation log
CREATE OR REPLACE FUNCTION prevent_violation_modification()
RETURNS TRIGGER AS $$
BEGIN
    RAISE EXCEPTION 'SECURITY VIOLATION: Violation log entries cannot be modified or deleted.';
END;
$$ LANGUAGE plpgsql;

-- Triggers for violations table
DROP TRIGGER IF EXISTS audit_violations_prevent_update ON audit_log_violations;
CREATE TRIGGER audit_violations_prevent_update
BEFORE UPDATE ON audit_log_violations
FOR EACH ROW
EXECUTE FUNCTION prevent_violation_modification();

DROP TRIGGER IF EXISTS audit_violations_prevent_delete ON audit_log_violations;
CREATE TRIGGER audit_violations_prevent_delete
BEFORE DELETE ON audit_log_violations
FOR EACH ROW
EXECUTE FUNCTION prevent_violation_modification();

-- Function to validate entry hash on INSERT
CREATE OR REPLACE FUNCTION validate_audit_entry_hash()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.entry_hash IS NULL OR NEW.entry_hash = '' THEN
        RAISE EXCEPTION 'SECURITY VIOLATION: Audit log entries must have a valid entry_hash.';
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger to validate entry hash
DROP TRIGGER IF EXISTS audit_log_validate_hash ON audit_log;
CREATE TRIGGER audit_log_validate_hash
BEFORE INSERT ON audit_log
FOR EACH ROW
EXECUTE FUNCTION validate_audit_entry_hash();

-- View for recent audit activity
CREATE OR REPLACE VIEW v_recent_audit_activity AS
SELECT
    id,
    timestamp,
    event_type,
    user_id,
    action,
    resource_type,
    resource_id,
    outcome,
    metadata->>'description' as description
FROM audit_log
ORDER BY timestamp DESC
LIMIT 100;

-- View for security alerts
CREATE OR REPLACE VIEW v_security_alerts AS
SELECT
    id,
    timestamp,
    user_id,
    action,
    outcome,
    metadata->>'alert_type' as alert_type,
    metadata->>'severity' as severity,
    metadata->>'description' as description
FROM audit_log
WHERE event_type = 'security_alert'
ORDER BY timestamp DESC;

-- View for failed authentication attempts
CREATE OR REPLACE VIEW v_failed_auth_attempts AS
SELECT
    id,
    timestamp,
    user_id,
    ip_address,
    user_agent,
    metadata->>'reason' as failure_reason
FROM audit_log
WHERE event_type = 'auth_failure'
ORDER BY timestamp DESC;

-- View for credential rotations
CREATE OR REPLACE VIEW v_credential_rotations AS
SELECT
    id,
    timestamp,
    user_id,
    outcome,
    metadata->>'credential_type' as credential_type,
    metadata->>'new_version' as new_version
FROM audit_log
WHERE event_type = 'credential_rotation'
ORDER BY timestamp DESC;

-- View for tamper attempts
CREATE OR REPLACE VIEW v_tamper_attempts AS
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

-- Function to get audit statistics
CREATE OR REPLACE FUNCTION get_audit_statistics(
    start_time TIMESTAMPTZ DEFAULT NOW() - INTERVAL '24 hours',
    end_time TIMESTAMPTZ DEFAULT NOW()
)
RETURNS TABLE (
    event_type TEXT,
    total_count BIGINT,
    success_count BIGINT,
    failure_count BIGINT,
    unique_users BIGINT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        al.event_type,
        COUNT(*)::BIGINT as total_count,
        COUNT(*) FILTER (WHERE al.outcome = 'success')::BIGINT as success_count,
        COUNT(*) FILTER (WHERE al.outcome IN ('failure', 'denied'))::BIGINT as failure_count,
        COUNT(DISTINCT al.user_id)::BIGINT as unique_users
    FROM audit_log al
    WHERE al.timestamp BETWEEN start_time AND end_time
    GROUP BY al.event_type
    ORDER BY total_count DESC;
END;
$$ LANGUAGE plpgsql;

-- Comment on table
COMMENT ON TABLE audit_log IS 'Immutable audit log for security events. Entries cannot be modified or deleted.';
COMMENT ON TABLE audit_log_violations IS 'Log of attempted modifications to the audit log.';
