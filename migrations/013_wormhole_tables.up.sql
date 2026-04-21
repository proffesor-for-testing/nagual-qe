-- Migration: 013_wormhole_tables
-- Description: Create tables for wormhole neural shortcuts
-- Created: 2026-02-02
--
-- Wormholes are neural shortcuts that create direct pathways between
-- frequently co-accessed patterns, bypassing normal graph traversal.
--
-- Key features:
-- - Auto-creation when patterns co-accessed >3 times in trajectories
-- - Strength-based decay for unused wormholes (30 day threshold)
-- - Audit logging for creation/deletion events
-- - Integration with ProfDAG edge system

-- ============================================================================
-- TABLE: wormholes
-- ============================================================================
-- Main wormhole storage with strength tracking and lifecycle management

CREATE TABLE IF NOT EXISTS wormholes (
    -- Primary key: UUID formatted as text
    id TEXT PRIMARY KEY,

    -- Source pattern/node ID
    source_id TEXT NOT NULL,

    -- Target pattern/node ID
    target_id TEXT NOT NULL,

    -- Current strength (0.0 - 1.0)
    -- Calculated as: base_strength + (usage * increment) - decay
    strength REAL NOT NULL DEFAULT 0.5
        CHECK (strength >= 0.0 AND strength <= 1.0),

    -- Why this wormhole was created (JSON)
    -- Contains: type (co_access, semantic_similarity, manual, learned)
    -- and additional context like count, similarity score, etc.
    creation_reason TEXT NOT NULL DEFAULT '{"type": "manual", "reason": "unknown"}',

    -- When the wormhole was created
    created_at TEXT NOT NULL,

    -- When the wormhole was last used/traversed
    last_used TEXT NOT NULL,

    -- Total number of times this wormhole has been traversed
    usage_count INTEGER NOT NULL DEFAULT 0,

    -- Number of graph edges this wormhole bypasses
    -- Used to calculate traversal savings
    path_distance_saved INTEGER,

    -- Whether this wormhole is currently active
    -- Inactive wormholes are kept for historical analysis
    is_active INTEGER NOT NULL DEFAULT 1,

    -- Additional metadata (JSON)
    metadata TEXT DEFAULT '{}',

    -- Prevent duplicate wormholes
    UNIQUE (source_id, target_id),

    -- Prevent self-loops
    CHECK (source_id != target_id)
);

-- Add table comment
-- Note: SQLite doesn't support COMMENT ON, but we document here for clarity

-- ============================================================================
-- TABLE: wormhole_co_access
-- ============================================================================
-- Tracks co-access patterns between nodes to identify wormhole candidates

CREATE TABLE IF NOT EXISTS wormhole_co_access (
    -- Pattern pair (lexicographically ordered)
    pattern_a TEXT NOT NULL,
    pattern_b TEXT NOT NULL,

    -- Number of times these patterns were co-accessed
    count INTEGER NOT NULL DEFAULT 1,

    -- First co-access timestamp
    first_accessed TEXT NOT NULL,

    -- Most recent co-access timestamp
    last_accessed TEXT NOT NULL,

    -- Last session where co-access occurred
    last_session_id TEXT,

    -- Last trajectory where co-access occurred
    last_trajectory_id TEXT,

    -- Primary key is the pattern pair
    PRIMARY KEY (pattern_a, pattern_b),

    -- Ensure consistent ordering (pattern_a < pattern_b)
    CHECK (pattern_a < pattern_b)
);

-- ============================================================================
-- TABLE: wormhole_usage_log
-- ============================================================================
-- Audit log for wormhole lifecycle events

CREATE TABLE IF NOT EXISTS wormhole_usage_log (
    -- Auto-incrementing ID
    id INTEGER PRIMARY KEY AUTOINCREMENT,

    -- Reference to wormhole
    wormhole_id TEXT NOT NULL,

    -- Event type: traversed, created, deactivated, deleted, strength_updated
    event_type TEXT NOT NULL CHECK (event_type IN (
        'traversed', 'created', 'deactivated', 'reactivated',
        'deleted', 'strength_updated', 'decay_applied'
    )),

    -- When the event occurred
    timestamp TEXT NOT NULL,

    -- Old strength (for strength_updated events)
    old_strength REAL,

    -- New strength (for strength_updated events)
    new_strength REAL,

    -- Additional context (JSON)
    context TEXT DEFAULT '{}',

    -- Foreign key constraint (soft - SQLite doesn't enforce by default)
    -- FOREIGN KEY (wormhole_id) REFERENCES wormholes(id) ON DELETE CASCADE

    -- Note: We don't enforce FK here because wormholes may be deleted
    -- while we want to retain the audit log
    CHECK (wormhole_id != '')
);

-- ============================================================================
-- INDEXES FOR wormholes
-- ============================================================================

-- Index for source lookups (get all wormholes from a pattern)
CREATE INDEX IF NOT EXISTS idx_wormholes_source
    ON wormholes (source_id);

-- Index for target lookups (get all wormholes to a pattern)
CREATE INDEX IF NOT EXISTS idx_wormholes_target
    ON wormholes (target_id);

-- Index for active wormholes by strength
CREATE INDEX IF NOT EXISTS idx_wormholes_active_strength
    ON wormholes (is_active, strength DESC);

-- Index for finding unused wormholes (for decay/cleanup)
CREATE INDEX IF NOT EXISTS idx_wormholes_last_used
    ON wormholes (last_used);

-- Index for high-usage wormholes
CREATE INDEX IF NOT EXISTS idx_wormholes_usage
    ON wormholes (usage_count DESC);

-- ============================================================================
-- INDEXES FOR wormhole_co_access
-- ============================================================================

-- Index for finding candidates above threshold
CREATE INDEX IF NOT EXISTS idx_co_access_count
    ON wormhole_co_access (count DESC);

-- Index for pattern-based lookups
CREATE INDEX IF NOT EXISTS idx_co_access_pattern_a
    ON wormhole_co_access (pattern_a);

CREATE INDEX IF NOT EXISTS idx_co_access_pattern_b
    ON wormhole_co_access (pattern_b);

-- Index for recent co-accesses
CREATE INDEX IF NOT EXISTS idx_co_access_last_accessed
    ON wormhole_co_access (last_accessed DESC);

-- ============================================================================
-- INDEXES FOR wormhole_usage_log
-- ============================================================================

-- Index for wormhole-specific queries
CREATE INDEX IF NOT EXISTS idx_usage_log_wormhole
    ON wormhole_usage_log (wormhole_id, timestamp DESC);

-- Index for event type queries
CREATE INDEX IF NOT EXISTS idx_usage_log_event_type
    ON wormhole_usage_log (event_type, timestamp DESC);

-- Index for time-based queries
CREATE INDEX IF NOT EXISTS idx_usage_log_timestamp
    ON wormhole_usage_log (timestamp DESC);

-- ============================================================================
-- TRIGGERS
-- ============================================================================

-- Trigger to log wormhole creation
CREATE TRIGGER IF NOT EXISTS trg_wormhole_created
    AFTER INSERT ON wormholes
BEGIN
    INSERT INTO wormhole_usage_log (wormhole_id, event_type, timestamp, new_strength, context)
    VALUES (
        NEW.id,
        'created',
        NEW.created_at,
        NEW.strength,
        json_object(
            'source_id', NEW.source_id,
            'target_id', NEW.target_id,
            'creation_reason', NEW.creation_reason
        )
    );
END;

-- Trigger to log wormhole deactivation
CREATE TRIGGER IF NOT EXISTS trg_wormhole_deactivated
    AFTER UPDATE OF is_active ON wormholes
    WHEN OLD.is_active = 1 AND NEW.is_active = 0
BEGIN
    INSERT INTO wormhole_usage_log (wormhole_id, event_type, timestamp, old_strength, new_strength, context)
    VALUES (
        NEW.id,
        'deactivated',
        datetime('now'),
        OLD.strength,
        NEW.strength,
        json_object('reason', 'strength below threshold')
    );
END;

-- Trigger to log wormhole reactivation
CREATE TRIGGER IF NOT EXISTS trg_wormhole_reactivated
    AFTER UPDATE OF is_active ON wormholes
    WHEN OLD.is_active = 0 AND NEW.is_active = 1
BEGIN
    INSERT INTO wormhole_usage_log (wormhole_id, event_type, timestamp, old_strength, new_strength)
    VALUES (
        NEW.id,
        'reactivated',
        datetime('now'),
        OLD.strength,
        NEW.strength
    );
END;

-- ============================================================================
-- VIEWS
-- ============================================================================

-- View for active wormholes with full details
CREATE VIEW IF NOT EXISTS v_active_wormholes AS
SELECT
    w.id,
    w.source_id,
    w.target_id,
    w.strength,
    w.creation_reason,
    w.created_at,
    w.last_used,
    w.usage_count,
    w.path_distance_saved,
    -- Calculate days since last use
    CAST((julianday('now') - julianday(w.last_used)) AS INTEGER) AS days_since_use,
    -- Calculate traversal savings percentage
    CASE
        WHEN w.path_distance_saved IS NOT NULL AND w.path_distance_saved > 0
        THEN ROUND((w.path_distance_saved - 1.0) / w.path_distance_saved * 100, 1)
        ELSE NULL
    END AS savings_percent
FROM wormholes w
WHERE w.is_active = 1;

-- View for wormhole candidates (co-access records not yet promoted)
CREATE VIEW IF NOT EXISTS v_wormhole_candidates AS
SELECT
    ca.pattern_a,
    ca.pattern_b,
    ca.count AS co_access_count,
    ca.first_accessed,
    ca.last_accessed,
    -- Calculate days between first and last access
    CAST((julianday(ca.last_accessed) - julianday(ca.first_accessed)) AS INTEGER) AS active_days
FROM wormhole_co_access ca
WHERE ca.count >= 3  -- Default activation threshold
AND NOT EXISTS (
    SELECT 1 FROM wormholes w
    WHERE (w.source_id = ca.pattern_a AND w.target_id = ca.pattern_b)
       OR (w.source_id = ca.pattern_b AND w.target_id = ca.pattern_a)
)
ORDER BY ca.count DESC;

-- View for wormhole statistics
CREATE VIEW IF NOT EXISTS v_wormhole_stats AS
SELECT
    (SELECT COUNT(*) FROM wormholes WHERE is_active = 1) AS active_count,
    (SELECT COUNT(*) FROM wormholes WHERE is_active = 0) AS inactive_count,
    (SELECT COALESCE(AVG(strength), 0) FROM wormholes WHERE is_active = 1) AS avg_strength,
    (SELECT COALESCE(AVG(usage_count), 0) FROM wormholes WHERE is_active = 1) AS avg_usage,
    (SELECT COALESCE(SUM(COALESCE(path_distance_saved, 0) * usage_count), 0)
     FROM wormholes WHERE is_active = 1) AS total_traversals_saved,
    (SELECT COUNT(*) FROM wormhole_co_access WHERE count >= 3) AS pending_candidates,
    (SELECT COUNT(*) FROM wormhole_usage_log WHERE event_type = 'traversed'
     AND timestamp >= datetime('now', '-7 days')) AS weekly_traversals;

-- ============================================================================
-- INITIAL DATA
-- ============================================================================

-- No initial data needed - wormholes are created dynamically based on usage patterns
