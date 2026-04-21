-- Migration: 001_initial_schema
-- Description: Initial PostgreSQL schema for Nagual with pgvector support
-- Created: 2026-01-31
--
-- This migration creates the core tables for the Nagual system:
-- - learnings: Knowledge items with vector embeddings
-- - patterns: ReasoningBank pattern storage (15+ fields per ADR-005)
-- - predictions: Prediction engine with Brier score calibration (ADR-006)
-- - sync_log: Synchronization tracking for dual-write pattern

-- ============================================================================
-- EXTENSION SETUP
-- ============================================================================

-- Enable pgvector extension for vector similarity search
CREATE EXTENSION IF NOT EXISTS ruvector VERSION '0.1.0';

-- Enable pg_trgm for trigram-based text search (fuzzy matching)
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- ============================================================================
-- CUSTOM TYPES
-- ============================================================================

-- Prediction status enum
CREATE TYPE prediction_status AS ENUM ('pending', 'resolved_true', 'resolved_false');

-- Sync operation type enum
CREATE TYPE sync_operation AS ENUM ('insert', 'update', 'delete', 'upsert');

-- Sync status enum
CREATE TYPE sync_status AS ENUM ('pending', 'syncing', 'completed', 'failed', 'retrying');

-- ============================================================================
-- TABLE: learnings
-- ============================================================================
-- Stores knowledge items with vector embeddings for semantic search.
-- Each learning represents a piece of knowledge that can be retrieved
-- based on semantic similarity.

CREATE TABLE learnings (
    -- Primary key: UUID for distributed generation
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Content fields
    content TEXT NOT NULL,
    title TEXT,
    summary TEXT,

    -- Categorization
    category TEXT NOT NULL DEFAULT 'general',
    tags TEXT[] DEFAULT '{}',

    -- Source tracking
    source TEXT,
    source_url TEXT,
    agent_id TEXT DEFAULT 'default',
    session_id TEXT,

    -- Vector embedding (128-dimensional per ADR-002)
    embedding ruvector(128),

    -- Quality metrics
    relevance_score FLOAT DEFAULT 0.5 CHECK (relevance_score BETWEEN 0 AND 1),
    confidence FLOAT DEFAULT 0.5 CHECK (confidence BETWEEN 0 AND 1),
    usage_count INTEGER DEFAULT 0,

    -- Metadata
    metadata JSONB DEFAULT '{}',

    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Add table comment
COMMENT ON TABLE learnings IS 'Knowledge items with vector embeddings for semantic search';

-- ============================================================================
-- TABLE: patterns
-- ============================================================================
-- ReasoningBank pattern storage implementing closed-loop memory pattern
-- from arXiv:2509.25140. Stores problem-solution pairs with context and
-- learning signals for SONA integration.
--
-- Schema follows ADR-005 with 15+ fields for comprehensive pattern tracking.

CREATE TABLE patterns (
    -- Primary key: Unique identifier (format: pat_{timestamp}_{uuid})
    id TEXT PRIMARY KEY,

    -- Timestamp when pattern was created
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Domain/category for pattern classification
    category TEXT NOT NULL,

    -- Problem description (indexed for FTS)
    problem TEXT NOT NULL,

    -- Solution description (indexed for FTS)
    solution TEXT,

    -- Additional context information
    context TEXT,

    -- Effectiveness rating (0-10 scale)
    effectiveness INTEGER DEFAULT 0 CHECK (effectiveness BETWEEN 0 AND 10),

    -- Number of times retrieved and used
    reuse_count INTEGER DEFAULT 0,

    -- SONA reward signal (0.0-1.0)
    reward FLOAT DEFAULT 0.5 CHECK (reward BETWEEN 0 AND 1),

    -- Whether pattern led to successful outcome
    success BOOLEAN DEFAULT FALSE,

    -- Specific improvement suggestions
    critique TEXT,

    -- Agent that created this pattern
    agent_id TEXT DEFAULT 'default',

    -- Session context for traceability
    session_id TEXT,

    -- Confidence in the solution (0.0-1.0)
    confidence FLOAT DEFAULT 0.5 CHECK (confidence BETWEEN 0 AND 1),

    -- Semantic embedding for similarity search (128-dim)
    embedding ruvector(128),

    -- Categorical tags for filtering
    tags TEXT[] DEFAULT '{}',

    -- Links to related pattern IDs
    related_patterns TEXT[] DEFAULT '{}',

    -- Additional metadata (JSON)
    metadata JSONB DEFAULT '{}',

    -- Creation timestamp
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Last update timestamp
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Add table comment
COMMENT ON TABLE patterns IS 'ReasoningBank pattern storage for closed-loop memory (ADR-005)';

-- Column comments
COMMENT ON COLUMN patterns.id IS 'Unique identifier (format: pat_{timestamp}_{uuid})';
COMMENT ON COLUMN patterns.category IS 'Domain/category (e.g., quality-engineering, testing)';
COMMENT ON COLUMN patterns.problem IS 'Problem description - indexed for full-text search';
COMMENT ON COLUMN patterns.solution IS 'Solution description - indexed for full-text search';
COMMENT ON COLUMN patterns.effectiveness IS 'Effectiveness rating on 0-10 scale';
COMMENT ON COLUMN patterns.reward IS 'SONA reward signal for reinforcement learning';
COMMENT ON COLUMN patterns.embedding IS '128-dimensional vector embedding for semantic search';

-- ============================================================================
-- TABLE: predictions
-- ============================================================================
-- Prediction engine with Brier score calibration per ADR-006.
-- Supports probabilistic predictions with evidence linking and
-- self-calibration based on historical accuracy.

CREATE TABLE predictions (
    -- Primary key: Unique identifier (format: pred_{uuid})
    id TEXT PRIMARY KEY,

    -- Category for prediction grouping
    category TEXT NOT NULL,

    -- Description of what is being predicted
    description TEXT NOT NULL,

    -- Predicted probability (0.0-1.0)
    probability FLOAT NOT NULL CHECK (probability BETWEEN 0 AND 1),

    -- Timeline estimates (optional)
    timeline_min_days INTEGER,
    timeline_max_days INTEGER,

    -- Evidence from patterns supporting this prediction
    -- Format: [{pattern_id, weight, summary}]
    evidence JSONB DEFAULT '[]',

    -- Current prediction status
    status prediction_status DEFAULT 'pending',

    -- When the prediction should be resolved
    resolution_date TIMESTAMPTZ,

    -- Actual outcome once resolved
    actual_outcome BOOLEAN,

    -- Brier score = (probability - outcome)^2
    -- Perfect: 0.0, Random: 0.25, Worst: 1.0
    brier_score FLOAT CHECK (brier_score IS NULL OR brier_score BETWEEN 0 AND 1),

    -- Calibration data
    calibrated_probability FLOAT CHECK (calibrated_probability IS NULL OR calibrated_probability BETWEEN 0 AND 1),
    calibration_bucket TEXT, -- e.g., "0.7-0.8"

    -- Reasoning trace for prediction generation
    reasoning TEXT,

    -- Source patterns used for prediction
    source_patterns TEXT[] DEFAULT '{}',

    -- Agent that created this prediction
    agent_id TEXT DEFAULT 'default',

    -- Metadata
    metadata JSONB DEFAULT '{}',

    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    resolved_at TIMESTAMPTZ,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Add table comment
COMMENT ON TABLE predictions IS 'Prediction engine with Brier score calibration (ADR-006)';

-- Column comments
COMMENT ON COLUMN predictions.probability IS 'Predicted probability (0.0-1.0)';
COMMENT ON COLUMN predictions.brier_score IS 'Brier score: (probability - outcome)^2. Perfect=0, Random=0.25, Worst=1.0';
COMMENT ON COLUMN predictions.evidence IS 'JSON array of evidence from patterns: [{pattern_id, weight, summary}]';
COMMENT ON COLUMN predictions.calibration_bucket IS 'Probability bucket for calibration tracking (e.g., "0.7-0.8")';

-- ============================================================================
-- TABLE: sync_log
-- ============================================================================
-- Tracks synchronization operations between SQLite and PostgreSQL
-- for the dual-write pattern (ADR-001).

CREATE TABLE sync_log (
    -- Primary key
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Entity identification
    entity_type TEXT NOT NULL, -- 'learning', 'pattern', 'prediction'
    entity_id TEXT NOT NULL,

    -- Operation details
    operation sync_operation NOT NULL,
    status sync_status DEFAULT 'pending',

    -- Source tracking
    source_db TEXT NOT NULL, -- 'sqlite' or 'postgres'
    target_db TEXT NOT NULL, -- 'sqlite' or 'postgres'

    -- Version tracking for conflict resolution
    source_version BIGINT NOT NULL DEFAULT 1,
    target_version BIGINT,

    -- Payload (the data being synced)
    payload JSONB NOT NULL,

    -- Error tracking
    error_message TEXT,
    error_count INTEGER DEFAULT 0,
    last_error_at TIMESTAMPTZ,

    -- Retry configuration
    max_retries INTEGER DEFAULT 3,
    next_retry_at TIMESTAMPTZ,

    -- Conflict resolution
    conflict_detected BOOLEAN DEFAULT FALSE,
    conflict_resolution TEXT, -- 'source_wins', 'target_wins', 'merge', 'manual'
    conflict_data JSONB,

    -- Timing
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,

    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Add table comment
COMMENT ON TABLE sync_log IS 'Synchronization tracking for dual-write pattern (ADR-001)';

-- Column comments
COMMENT ON COLUMN sync_log.entity_type IS 'Type of entity being synced: learning, pattern, prediction';
COMMENT ON COLUMN sync_log.source_db IS 'Source database: sqlite or postgres';
COMMENT ON COLUMN sync_log.conflict_resolution IS 'How conflicts were resolved: source_wins, target_wins, merge, manual';

-- ============================================================================
-- TABLE: calibration_buckets
-- ============================================================================
-- Tracks calibration statistics for prediction accuracy by probability bucket.
-- Used for self-calibrating predictions per ADR-006.

CREATE TABLE calibration_buckets (
    -- Composite primary key
    category TEXT NOT NULL,
    bucket_min FLOAT NOT NULL CHECK (bucket_min >= 0 AND bucket_min < 1),
    bucket_max FLOAT NOT NULL CHECK (bucket_max > 0 AND bucket_max <= 1),

    -- Statistics
    total_predictions INTEGER DEFAULT 0,
    true_outcomes INTEGER DEFAULT 0,
    false_outcomes INTEGER DEFAULT 0,

    -- Calibration metrics
    expected_rate FLOAT NOT NULL, -- Expected truth rate for this bucket
    actual_rate FLOAT, -- Observed truth rate
    calibration_error FLOAT, -- actual_rate - expected_rate

    -- Brier score statistics
    total_brier_score FLOAT DEFAULT 0,
    avg_brier_score FLOAT,

    -- Adjustment factor for future predictions
    adjustment_factor FLOAT DEFAULT 0,

    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    PRIMARY KEY (category, bucket_min, bucket_max),
    CHECK (bucket_max > bucket_min)
);

-- Add table comment
COMMENT ON TABLE calibration_buckets IS 'Calibration statistics for prediction accuracy by probability bucket';

-- ============================================================================
-- BASIC INDEXES (HNSW indexes are in separate migration)
-- ============================================================================

-- Learnings indexes
CREATE INDEX learnings_category_idx ON learnings (category);
CREATE INDEX learnings_tags_idx ON learnings USING GIN (tags);
CREATE INDEX learnings_agent_id_idx ON learnings (agent_id);
CREATE INDEX learnings_created_at_idx ON learnings (created_at DESC);

-- Patterns indexes
CREATE INDEX patterns_category_idx ON patterns (category);
CREATE INDEX patterns_reward_idx ON patterns (reward DESC);
CREATE INDEX patterns_tags_idx ON patterns USING GIN (tags);
CREATE INDEX patterns_agent_id_idx ON patterns (agent_id);
CREATE INDEX patterns_created_at_idx ON patterns (created_at DESC);
CREATE INDEX patterns_success_idx ON patterns (success) WHERE success = TRUE;

-- Predictions indexes
CREATE INDEX predictions_status_idx ON predictions (status);
CREATE INDEX predictions_category_idx ON predictions (category);
CREATE INDEX predictions_created_at_idx ON predictions (created_at DESC);
CREATE INDEX predictions_resolution_date_idx ON predictions (resolution_date) WHERE status = 'pending';
CREATE INDEX predictions_brier_score_idx ON predictions (brier_score) WHERE brier_score IS NOT NULL;

-- Sync log indexes
CREATE INDEX sync_log_entity_idx ON sync_log (entity_type, entity_id);
CREATE INDEX sync_log_status_idx ON sync_log (status) WHERE status != 'completed';
CREATE INDEX sync_log_next_retry_idx ON sync_log (next_retry_at) WHERE status = 'retrying';
CREATE INDEX sync_log_created_at_idx ON sync_log (created_at DESC);

-- ============================================================================
-- TRIGGERS FOR updated_at
-- ============================================================================

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Learnings trigger
CREATE TRIGGER learnings_updated_at
    BEFORE UPDATE ON learnings
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Patterns trigger
CREATE TRIGGER patterns_updated_at
    BEFORE UPDATE ON patterns
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Predictions trigger
CREATE TRIGGER predictions_updated_at
    BEFORE UPDATE ON predictions
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Sync log trigger
CREATE TRIGGER sync_log_updated_at
    BEFORE UPDATE ON sync_log
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Calibration buckets trigger
CREATE TRIGGER calibration_buckets_updated_at
    BEFORE UPDATE ON calibration_buckets
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- ============================================================================
-- INITIAL CALIBRATION BUCKET DATA
-- ============================================================================

-- Insert default calibration buckets for common categories
DO $$
DECLARE
    categories TEXT[] := ARRAY['general', 'quality-engineering', 'testing', 'sprint-planning'];
    cat TEXT;
    i INTEGER;
BEGIN
    FOREACH cat IN ARRAY categories
    LOOP
        FOR i IN 0..9
        LOOP
            INSERT INTO calibration_buckets (
                category, bucket_min, bucket_max, expected_rate
            ) VALUES (
                cat,
                i::FLOAT / 10.0,
                (i + 1)::FLOAT / 10.0,
                (i::FLOAT + 0.5) / 10.0  -- Midpoint of bucket
            ) ON CONFLICT DO NOTHING;
        END LOOP;
    END LOOP;
END $$;
