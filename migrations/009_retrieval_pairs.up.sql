-- Migration: 009_retrieval_pairs
-- Description: Co-retrieval tracking and pattern edges for auto edge creation
-- Created: 2026-01-31
--
-- This migration creates:
-- - retrieval_pairs: Tracks patterns retrieved together for co-retrieval edges
-- - pattern_edges: Stores edges between patterns (similar_to, co_retrieved)
-- - edge_audit_log: Audit trail for edge pruning operations

-- ============================================================================
-- TABLE: retrieval_pairs
-- ============================================================================
-- Tracks co-retrieval frequency between pattern pairs.
-- Used to automatically create CoRetrieved edges when patterns are
-- frequently retrieved together (threshold: > 3 times).

CREATE TABLE IF NOT EXISTS retrieval_pairs (
    -- Pattern IDs forming the pair (ordered: pattern_a < pattern_b for uniqueness)
    pattern_a TEXT NOT NULL,
    pattern_b TEXT NOT NULL,

    -- Co-retrieval statistics
    count INTEGER NOT NULL DEFAULT 1,

    -- Timestamps
    first_retrieved TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_retrieved TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Session tracking
    last_session_id TEXT,
    last_agent_id TEXT,

    -- Metadata (can include query contexts, etc.)
    metadata JSONB DEFAULT '{}',

    -- Composite primary key ensures uniqueness
    PRIMARY KEY (pattern_a, pattern_b),

    -- Ensure pattern_a < pattern_b for consistent ordering
    CHECK (pattern_a < pattern_b),

    -- Foreign keys to patterns table
    FOREIGN KEY (pattern_a) REFERENCES patterns(id) ON DELETE CASCADE,
    FOREIGN KEY (pattern_b) REFERENCES patterns(id) ON DELETE CASCADE
);

-- Add table comment
COMMENT ON TABLE retrieval_pairs IS 'Tracks co-retrieval frequency for automatic edge creation';

-- Column comments
COMMENT ON COLUMN retrieval_pairs.count IS 'Number of times patterns were retrieved together';
COMMENT ON COLUMN retrieval_pairs.first_retrieved IS 'First time patterns were retrieved together';
COMMENT ON COLUMN retrieval_pairs.last_retrieved IS 'Most recent co-retrieval';

-- ============================================================================
-- TABLE: pattern_edges
-- ============================================================================
-- Stores directed edges between patterns with different relationship types.
-- Supports: similar_to, co_retrieved, derived_from, related_to

-- Edge type enum
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'edge_type') THEN
        CREATE TYPE edge_type AS ENUM (
            'similar_to',      -- Similarity-based edge (cosine > 0.85)
            'co_retrieved',    -- Co-retrieval based edge
            'derived_from',    -- Pattern derived from another
            'related_to'       -- Manual or inferred relation
        );
    END IF;
END$$;

CREATE TABLE IF NOT EXISTS pattern_edges (
    -- Unique edge identifier
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Edge endpoints
    source_pattern TEXT NOT NULL,
    target_pattern TEXT NOT NULL,

    -- Edge properties
    edge_type edge_type NOT NULL,
    strength FLOAT NOT NULL DEFAULT 0.5 CHECK (strength BETWEEN 0 AND 1),

    -- Metadata
    confidence FLOAT DEFAULT 0.5 CHECK (confidence BETWEEN 0 AND 1),
    bidirectional BOOLEAN DEFAULT FALSE,

    -- Auto-creation metadata
    auto_created BOOLEAN DEFAULT FALSE,
    creation_reason TEXT,

    -- Versioning for updates
    version INTEGER DEFAULT 1,

    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_accessed TIMESTAMPTZ,

    -- Metadata (can include similarity scores, co-retrieval counts, etc.)
    metadata JSONB DEFAULT '{}',

    -- Prevent duplicate edges of same type
    UNIQUE (source_pattern, target_pattern, edge_type),

    -- Foreign keys
    FOREIGN KEY (source_pattern) REFERENCES patterns(id) ON DELETE CASCADE,
    FOREIGN KEY (target_pattern) REFERENCES patterns(id) ON DELETE CASCADE
);

-- Add table comment
COMMENT ON TABLE pattern_edges IS 'Stores edges between patterns with relationship types and strengths';

-- Column comments
COMMENT ON COLUMN pattern_edges.strength IS 'Edge strength (0.0-1.0), used for pruning weak edges';
COMMENT ON COLUMN pattern_edges.auto_created IS 'Whether edge was automatically created vs manual';
COMMENT ON COLUMN pattern_edges.creation_reason IS 'Explanation for auto-created edges';

-- ============================================================================
-- TABLE: edge_audit_log
-- ============================================================================
-- Audit trail for edge operations (creation, updates, pruning).

-- Edge operation type enum
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'edge_operation') THEN
        CREATE TYPE edge_operation AS ENUM (
            'created',         -- Edge was created
            'updated',         -- Edge was updated
            'pruned',          -- Edge was pruned (weak/old)
            'deleted',         -- Edge was manually deleted
            'strengthened',    -- Edge strength increased
            'weakened'         -- Edge strength decreased
        );
    END IF;
END$$;

CREATE TABLE IF NOT EXISTS edge_audit_log (
    -- Unique log entry identifier
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Edge identification
    edge_id UUID NOT NULL,
    source_pattern TEXT NOT NULL,
    target_pattern TEXT NOT NULL,
    edge_type edge_type NOT NULL,

    -- Operation details
    operation edge_operation NOT NULL,

    -- Edge state at operation time
    old_strength FLOAT,
    new_strength FLOAT,

    -- Operation context
    reason TEXT,
    job_id TEXT,                -- Maintenance job that triggered this

    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Metadata
    metadata JSONB DEFAULT '{}'
);

-- Add table comment
COMMENT ON TABLE edge_audit_log IS 'Audit trail for edge operations (creation, pruning, updates)';

-- ============================================================================
-- INDEXES FOR retrieval_pairs
-- ============================================================================

-- Index for looking up pairs by either pattern
CREATE INDEX retrieval_pairs_pattern_a_idx ON retrieval_pairs (pattern_a);
CREATE INDEX retrieval_pairs_pattern_b_idx ON retrieval_pairs (pattern_b);

-- Index for finding pairs above co-retrieval threshold
CREATE INDEX retrieval_pairs_count_idx ON retrieval_pairs (count DESC)
    WHERE count >= 3;

-- Index for recent co-retrievals
CREATE INDEX retrieval_pairs_last_retrieved_idx ON retrieval_pairs (last_retrieved DESC);

-- ============================================================================
-- INDEXES FOR pattern_edges
-- ============================================================================

-- Index for traversing from source
CREATE INDEX pattern_edges_source_idx ON pattern_edges (source_pattern);

-- Index for traversing to target
CREATE INDEX pattern_edges_target_idx ON pattern_edges (target_pattern);

-- Index for edge type queries
CREATE INDEX pattern_edges_type_idx ON pattern_edges (edge_type);

-- Index for finding weak edges (for pruning)
CREATE INDEX pattern_edges_weak_idx ON pattern_edges (strength, created_at)
    WHERE strength < 0.1;

-- Index for auto-created edges
CREATE INDEX pattern_edges_auto_idx ON pattern_edges (auto_created)
    WHERE auto_created = TRUE;

-- Index for edge age calculations
CREATE INDEX pattern_edges_created_at_idx ON pattern_edges (created_at);

-- Composite index for similarity edges lookup
CREATE INDEX pattern_edges_similar_idx ON pattern_edges (source_pattern, strength DESC)
    WHERE edge_type = 'similar_to';

-- ============================================================================
-- INDEXES FOR edge_audit_log
-- ============================================================================

-- Index for edge history lookup
CREATE INDEX edge_audit_log_edge_id_idx ON edge_audit_log (edge_id);

-- Index for operation type queries
CREATE INDEX edge_audit_log_operation_idx ON edge_audit_log (operation);

-- Index for time-range queries
CREATE INDEX edge_audit_log_created_at_idx ON edge_audit_log (created_at DESC);

-- Index for job audits
CREATE INDEX edge_audit_log_job_id_idx ON edge_audit_log (job_id)
    WHERE job_id IS NOT NULL;

-- ============================================================================
-- TRIGGERS FOR updated_at
-- ============================================================================

-- Pattern edges trigger
CREATE TRIGGER pattern_edges_updated_at
    BEFORE UPDATE ON pattern_edges
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- ============================================================================
-- FUNCTIONS FOR CO-RETRIEVAL
-- ============================================================================

-- Function to record a co-retrieval event
CREATE OR REPLACE FUNCTION record_co_retrieval(
    p_pattern_a TEXT,
    p_pattern_b TEXT,
    p_session_id TEXT DEFAULT NULL,
    p_agent_id TEXT DEFAULT NULL
)
RETURNS VOID AS $$
DECLARE
    v_ordered_a TEXT;
    v_ordered_b TEXT;
BEGIN
    -- Ensure consistent ordering (pattern_a < pattern_b)
    IF p_pattern_a < p_pattern_b THEN
        v_ordered_a := p_pattern_a;
        v_ordered_b := p_pattern_b;
    ELSE
        v_ordered_a := p_pattern_b;
        v_ordered_b := p_pattern_a;
    END IF;

    -- Upsert the co-retrieval pair
    INSERT INTO retrieval_pairs (
        pattern_a, pattern_b, count, first_retrieved, last_retrieved,
        last_session_id, last_agent_id
    ) VALUES (
        v_ordered_a, v_ordered_b, 1, NOW(), NOW(),
        p_session_id, p_agent_id
    )
    ON CONFLICT (pattern_a, pattern_b) DO UPDATE SET
        count = retrieval_pairs.count + 1,
        last_retrieved = NOW(),
        last_session_id = COALESCE(p_session_id, retrieval_pairs.last_session_id),
        last_agent_id = COALESCE(p_agent_id, retrieval_pairs.last_agent_id);
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION record_co_retrieval IS 'Records a co-retrieval event between two patterns';

-- Function to get co-retrieval count for a pair
CREATE OR REPLACE FUNCTION get_co_retrieval_count(
    p_pattern_a TEXT,
    p_pattern_b TEXT
)
RETURNS INTEGER AS $$
DECLARE
    v_ordered_a TEXT;
    v_ordered_b TEXT;
    v_count INTEGER;
BEGIN
    -- Ensure consistent ordering
    IF p_pattern_a < p_pattern_b THEN
        v_ordered_a := p_pattern_a;
        v_ordered_b := p_pattern_b;
    ELSE
        v_ordered_a := p_pattern_b;
        v_ordered_b := p_pattern_a;
    END IF;

    SELECT count INTO v_count
    FROM retrieval_pairs
    WHERE pattern_a = v_ordered_a AND pattern_b = v_ordered_b;

    RETURN COALESCE(v_count, 0);
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION get_co_retrieval_count IS 'Returns the co-retrieval count for a pattern pair';

-- Function to find pairs exceeding threshold
CREATE OR REPLACE FUNCTION get_coretrieval_candidates(
    p_threshold INTEGER DEFAULT 3,
    p_limit INTEGER DEFAULT 100
)
RETURNS TABLE (
    pattern_a TEXT,
    pattern_b TEXT,
    count INTEGER,
    last_retrieved TIMESTAMPTZ
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        rp.pattern_a,
        rp.pattern_b,
        rp.count,
        rp.last_retrieved
    FROM retrieval_pairs rp
    WHERE rp.count >= p_threshold
    AND NOT EXISTS (
        -- Exclude pairs that already have a co_retrieved edge
        SELECT 1 FROM pattern_edges pe
        WHERE pe.edge_type = 'co_retrieved'
        AND (
            (pe.source_pattern = rp.pattern_a AND pe.target_pattern = rp.pattern_b)
            OR (pe.source_pattern = rp.pattern_b AND pe.target_pattern = rp.pattern_a)
        )
    )
    ORDER BY rp.count DESC, rp.last_retrieved DESC
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION get_coretrieval_candidates IS 'Finds co-retrieval pairs that should become edges';
