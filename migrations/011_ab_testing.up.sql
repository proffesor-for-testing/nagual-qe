-- Migration: 011_ab_testing
-- Description: A/B Testing Infrastructure for SONA Learning
-- Created: 2026-01-31
--
-- This migration creates:
-- - ab_test_assignments: Tracks variant assignments for sessions
-- - ab_test_metrics: Records metrics for each variant (retrieval_time, reward, relevance)
-- - ab_test_experiments: Stores experiment configuration
-- - ab_test_baselines: Weekly baseline snapshots for regression detection
-- - improvement_targets: Quarterly improvement goals

-- ============================================================================
-- ENUM TYPES
-- ============================================================================

-- Variant type (Control vs Treatment)
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'ab_variant') THEN
        CREATE TYPE ab_variant AS ENUM ('control', 'treatment');
    END IF;
END$$;

-- Metric types
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'ab_metric_type') THEN
        CREATE TYPE ab_metric_type AS ENUM (
            'retrieval_time',
            'reward_achieved',
            'pattern_relevance',
            'pattern_count',
            'user_satisfaction'
        );
    END IF;
END$$;

-- Aggregation methods
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'metric_aggregation') THEN
        CREATE TYPE metric_aggregation AS ENUM (
            'mean',
            'median',
            'p95',
            'sum',
            'count'
        );
    END IF;
END$$;

-- Regression severity
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'regression_severity') THEN
        CREATE TYPE regression_severity AS ENUM ('warning', 'critical');
    END IF;
END$$;

-- ============================================================================
-- TABLE: ab_test_experiments
-- ============================================================================
-- Stores configuration for A/B test experiments.

CREATE TABLE IF NOT EXISTS ab_test_experiments (
    -- Unique experiment identifier
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Experiment name (unique)
    name TEXT NOT NULL UNIQUE,

    -- Configuration
    split_ratio FLOAT NOT NULL DEFAULT 0.2 CHECK (split_ratio BETWEEN 0 AND 1),
    deterministic_assignment BOOLEAN NOT NULL DEFAULT TRUE,
    min_samples_for_analysis INTEGER NOT NULL DEFAULT 100,
    rolling_window_days INTEGER NOT NULL DEFAULT 7,

    -- State
    is_active BOOLEAN NOT NULL DEFAULT TRUE,

    -- Timestamps
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ends_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Description and notes
    description TEXT,
    metadata JSONB DEFAULT '{}'
);

-- Add table comment
COMMENT ON TABLE ab_test_experiments IS 'Configuration for A/B test experiments';
COMMENT ON COLUMN ab_test_experiments.split_ratio IS 'Ratio of traffic to treatment (SONA-optimized), default 0.2 = 20%';
COMMENT ON COLUMN ab_test_experiments.deterministic_assignment IS 'Whether assignment is deterministic based on session_id';

-- ============================================================================
-- TABLE: ab_test_assignments
-- ============================================================================
-- Tracks variant assignments for sessions to ensure consistency.

CREATE TABLE IF NOT EXISTS ab_test_assignments (
    -- Unique assignment identifier
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Experiment reference
    experiment_id UUID NOT NULL REFERENCES ab_test_experiments(id) ON DELETE CASCADE,

    -- Session and agent identifiers
    session_id TEXT NOT NULL,
    agent_id TEXT,

    -- Assigned variant
    variant ab_variant NOT NULL,

    -- Assignment metadata
    hash_value BIGINT, -- Hash used for deterministic assignment

    -- Timestamps
    assigned_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Additional context
    metadata JSONB DEFAULT '{}',

    -- Ensure unique assignment per session per experiment
    UNIQUE (experiment_id, session_id)
);

-- Add table comment
COMMENT ON TABLE ab_test_assignments IS 'Tracks variant assignments for sessions';
COMMENT ON COLUMN ab_test_assignments.hash_value IS 'Hash value used for deterministic assignment';

-- Indexes for ab_test_assignments
CREATE INDEX ab_test_assignments_experiment_idx ON ab_test_assignments (experiment_id);
CREATE INDEX ab_test_assignments_session_idx ON ab_test_assignments (session_id);
CREATE INDEX ab_test_assignments_variant_idx ON ab_test_assignments (variant);
CREATE INDEX ab_test_assignments_assigned_at_idx ON ab_test_assignments (assigned_at DESC);

-- ============================================================================
-- TABLE: ab_test_metrics
-- ============================================================================
-- Records metrics for A/B test analysis.

CREATE TABLE IF NOT EXISTS ab_test_metrics (
    -- Unique metric identifier
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Experiment reference
    experiment_id UUID NOT NULL REFERENCES ab_test_experiments(id) ON DELETE CASCADE,

    -- Assignment reference (optional, for linking to specific session)
    assignment_id UUID REFERENCES ab_test_assignments(id) ON DELETE SET NULL,

    -- Variant and metric type
    variant ab_variant NOT NULL,
    metric_type ab_metric_type NOT NULL,

    -- Metric value
    value FLOAT NOT NULL,

    -- Session context
    session_id TEXT,
    agent_id TEXT,

    -- Request context
    pattern_ids TEXT[], -- Patterns involved in this metric
    query_text TEXT,    -- Optional: the query that triggered retrieval

    -- Timestamps
    recorded_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Additional context
    metadata JSONB DEFAULT '{}'
);

-- Add table comment
COMMENT ON TABLE ab_test_metrics IS 'Records metrics for A/B test analysis';
COMMENT ON COLUMN ab_test_metrics.value IS 'Metric value (time in ms, scores 0-1, counts, etc.)';

-- Indexes for ab_test_metrics
CREATE INDEX ab_test_metrics_experiment_idx ON ab_test_metrics (experiment_id);
CREATE INDEX ab_test_metrics_variant_idx ON ab_test_metrics (variant);
CREATE INDEX ab_test_metrics_metric_type_idx ON ab_test_metrics (metric_type);
CREATE INDEX ab_test_metrics_recorded_at_idx ON ab_test_metrics (recorded_at DESC);

-- Composite index for analysis queries
CREATE INDEX ab_test_metrics_analysis_idx ON ab_test_metrics (
    experiment_id, variant, metric_type, recorded_at DESC
);

-- Index for session-based queries
CREATE INDEX ab_test_metrics_session_idx ON ab_test_metrics (session_id)
    WHERE session_id IS NOT NULL;

-- ============================================================================
-- TABLE: ab_test_baselines
-- ============================================================================
-- Weekly baseline snapshots for regression detection.

CREATE TABLE IF NOT EXISTS ab_test_baselines (
    -- Unique baseline identifier
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Experiment reference
    experiment_id UUID NOT NULL REFERENCES ab_test_experiments(id) ON DELETE CASCADE,

    -- Baseline metadata
    metric_type ab_metric_type NOT NULL,
    aggregation metric_aggregation NOT NULL,
    variant ab_variant, -- NULL for overall baseline

    -- Baseline values
    value FLOAT NOT NULL,
    sample_count INTEGER NOT NULL,
    std_dev FLOAT,
    min_value FLOAT,
    max_value FLOAT,
    p25 FLOAT,
    p50 FLOAT,
    p75 FLOAT,
    p95 FLOAT,

    -- Time period
    period_start TIMESTAMPTZ NOT NULL,
    period_end TIMESTAMPTZ NOT NULL,
    week_number INTEGER, -- Week of year
    year INTEGER,

    -- Timestamps
    computed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Additional context
    metadata JSONB DEFAULT '{}',

    -- Unique constraint for period + metric + variant combination
    UNIQUE (experiment_id, metric_type, aggregation, variant, period_start, period_end)
);

-- Add table comment
COMMENT ON TABLE ab_test_baselines IS 'Weekly baseline snapshots for regression detection';
COMMENT ON COLUMN ab_test_baselines.period_start IS 'Start of the baseline period';
COMMENT ON COLUMN ab_test_baselines.week_number IS 'ISO week number (1-53)';

-- Indexes for ab_test_baselines
CREATE INDEX ab_test_baselines_experiment_idx ON ab_test_baselines (experiment_id);
CREATE INDEX ab_test_baselines_metric_idx ON ab_test_baselines (metric_type);
CREATE INDEX ab_test_baselines_period_idx ON ab_test_baselines (period_start DESC, period_end DESC);
CREATE INDEX ab_test_baselines_week_idx ON ab_test_baselines (year DESC, week_number DESC);

-- ============================================================================
-- TABLE: regression_alerts
-- ============================================================================
-- Records detected regression alerts.

CREATE TABLE IF NOT EXISTS regression_alerts (
    -- Unique alert identifier
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Experiment reference
    experiment_id UUID NOT NULL REFERENCES ab_test_experiments(id) ON DELETE CASCADE,

    -- Alert details
    metric_type ab_metric_type NOT NULL,
    severity regression_severity NOT NULL,

    -- Values
    baseline_value FLOAT NOT NULL,
    current_value FLOAT NOT NULL,
    percentage_change FLOAT NOT NULL,

    -- Context
    baseline_id UUID REFERENCES ab_test_baselines(id) ON DELETE SET NULL,
    consecutive_days INTEGER DEFAULT 1,

    -- Alert state
    is_acknowledged BOOLEAN NOT NULL DEFAULT FALSE,
    acknowledged_by TEXT,
    acknowledged_at TIMESTAMPTZ,
    is_resolved BOOLEAN NOT NULL DEFAULT FALSE,
    resolved_at TIMESTAMPTZ,
    resolution_notes TEXT,

    -- Timestamps
    detected_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Description and metadata
    description TEXT,
    metadata JSONB DEFAULT '{}'
);

-- Add table comment
COMMENT ON TABLE regression_alerts IS 'Records detected regression alerts';
COMMENT ON COLUMN regression_alerts.percentage_change IS 'Negative = degradation for reward/relevance, positive = degradation for retrieval_time';

-- Indexes for regression_alerts
CREATE INDEX regression_alerts_experiment_idx ON regression_alerts (experiment_id);
CREATE INDEX regression_alerts_severity_idx ON regression_alerts (severity);
CREATE INDEX regression_alerts_detected_at_idx ON regression_alerts (detected_at DESC);
CREATE INDEX regression_alerts_unresolved_idx ON regression_alerts (is_resolved, detected_at DESC)
    WHERE is_resolved = FALSE;

-- ============================================================================
-- TABLE: improvement_targets
-- ============================================================================
-- Quarterly improvement goals and progress tracking.

CREATE TABLE IF NOT EXISTS improvement_targets (
    -- Unique target identifier
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Experiment reference (optional, NULL for system-wide targets)
    experiment_id UUID REFERENCES ab_test_experiments(id) ON DELETE CASCADE,

    -- Target definition
    metric_type ab_metric_type NOT NULL,
    baseline_value FLOAT NOT NULL,
    target_value FLOAT NOT NULL,
    description TEXT NOT NULL,

    -- Time period
    quarter SMALLINT NOT NULL CHECK (quarter BETWEEN 1 AND 4),
    year INTEGER NOT NULL,

    -- Current progress
    current_value FLOAT,
    progress FLOAT, -- 0.0 to 1.0+ (can exceed 1.0 if exceeded target)
    is_achieved BOOLEAN NOT NULL DEFAULT FALSE,
    achieved_at TIMESTAMPTZ,

    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Additional context
    metadata JSONB DEFAULT '{}',

    -- Unique constraint for metric + quarter + year
    UNIQUE (experiment_id, metric_type, quarter, year)
);

-- Add table comment
COMMENT ON TABLE improvement_targets IS 'Quarterly improvement goals and progress tracking';
COMMENT ON COLUMN improvement_targets.progress IS 'Progress towards target (0.0 = no progress, 1.0 = achieved, >1.0 = exceeded)';

-- Indexes for improvement_targets
CREATE INDEX improvement_targets_quarter_idx ON improvement_targets (year DESC, quarter DESC);
CREATE INDEX improvement_targets_metric_idx ON improvement_targets (metric_type);
CREATE INDEX improvement_targets_achieved_idx ON improvement_targets (is_achieved);

-- ============================================================================
-- TRIGGERS
-- ============================================================================

-- Update timestamps trigger for experiments
CREATE TRIGGER ab_test_experiments_updated_at
    BEFORE UPDATE ON ab_test_experiments
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Update timestamps trigger for improvement_targets
CREATE TRIGGER improvement_targets_updated_at
    BEFORE UPDATE ON improvement_targets
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- ============================================================================
-- FUNCTIONS
-- ============================================================================

-- Function to assign a variant deterministically
CREATE OR REPLACE FUNCTION assign_ab_variant(
    p_experiment_id UUID,
    p_session_id TEXT,
    p_agent_id TEXT DEFAULT NULL
)
RETURNS ab_variant AS $$
DECLARE
    v_experiment ab_test_experiments%ROWTYPE;
    v_existing_variant ab_variant;
    v_hash_value BIGINT;
    v_normalized FLOAT;
    v_new_variant ab_variant;
BEGIN
    -- Get experiment configuration
    SELECT * INTO v_experiment
    FROM ab_test_experiments
    WHERE id = p_experiment_id AND is_active = TRUE;

    IF NOT FOUND THEN
        -- Return control if experiment not found or inactive
        RETURN 'control';
    END IF;

    -- Check for existing assignment
    SELECT variant INTO v_existing_variant
    FROM ab_test_assignments
    WHERE experiment_id = p_experiment_id AND session_id = p_session_id;

    IF FOUND THEN
        RETURN v_existing_variant;
    END IF;

    -- Calculate hash for deterministic assignment
    v_hash_value := abs(('x' || substr(md5(p_session_id || v_experiment.name), 1, 16))::bit(64)::bigint);
    v_normalized := v_hash_value::float / 9223372036854775807::float;

    -- Determine variant
    IF v_normalized < v_experiment.split_ratio THEN
        v_new_variant := 'treatment';
    ELSE
        v_new_variant := 'control';
    END IF;

    -- Store assignment
    INSERT INTO ab_test_assignments (
        experiment_id, session_id, agent_id, variant, hash_value
    ) VALUES (
        p_experiment_id, p_session_id, p_agent_id, v_new_variant, v_hash_value
    )
    ON CONFLICT (experiment_id, session_id) DO NOTHING;

    RETURN v_new_variant;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION assign_ab_variant IS 'Assigns a variant to a session deterministically based on hash';

-- Function to record a metric
CREATE OR REPLACE FUNCTION record_ab_metric(
    p_experiment_id UUID,
    p_variant ab_variant,
    p_metric_type ab_metric_type,
    p_value FLOAT,
    p_session_id TEXT DEFAULT NULL,
    p_agent_id TEXT DEFAULT NULL,
    p_pattern_ids TEXT[] DEFAULT NULL,
    p_query_text TEXT DEFAULT NULL
)
RETURNS UUID AS $$
DECLARE
    v_metric_id UUID;
    v_assignment_id UUID;
BEGIN
    -- Get assignment ID if session provided
    IF p_session_id IS NOT NULL THEN
        SELECT id INTO v_assignment_id
        FROM ab_test_assignments
        WHERE experiment_id = p_experiment_id AND session_id = p_session_id;
    END IF;

    -- Insert metric
    INSERT INTO ab_test_metrics (
        experiment_id, assignment_id, variant, metric_type, value,
        session_id, agent_id, pattern_ids, query_text
    ) VALUES (
        p_experiment_id, v_assignment_id, p_variant, p_metric_type, p_value,
        p_session_id, p_agent_id, p_pattern_ids, p_query_text
    )
    RETURNING id INTO v_metric_id;

    RETURN v_metric_id;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION record_ab_metric IS 'Records a metric for A/B test analysis';

-- Function to compute weekly baseline
CREATE OR REPLACE FUNCTION compute_ab_baseline(
    p_experiment_id UUID,
    p_metric_type ab_metric_type,
    p_aggregation metric_aggregation,
    p_variant ab_variant DEFAULT NULL,
    p_start_date DATE DEFAULT CURRENT_DATE - INTERVAL '7 days',
    p_end_date DATE DEFAULT CURRENT_DATE
)
RETURNS UUID AS $$
DECLARE
    v_baseline_id UUID;
    v_stats RECORD;
    v_week_number INTEGER;
    v_year INTEGER;
BEGIN
    -- Calculate statistics
    SELECT
        COUNT(*) as sample_count,
        AVG(value) as mean_value,
        STDDEV_POP(value) as std_dev,
        MIN(value) as min_value,
        MAX(value) as max_value,
        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY value) as p25,
        PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY value) as p50,
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY value) as p75,
        PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY value) as p95
    INTO v_stats
    FROM ab_test_metrics
    WHERE experiment_id = p_experiment_id
    AND metric_type = p_metric_type
    AND recorded_at >= p_start_date
    AND recorded_at < p_end_date + INTERVAL '1 day'
    AND (p_variant IS NULL OR variant = p_variant);

    -- Get week number and year
    v_week_number := EXTRACT(WEEK FROM p_start_date);
    v_year := EXTRACT(YEAR FROM p_start_date);

    -- Calculate aggregated value
    DECLARE
        v_agg_value FLOAT;
    BEGIN
        CASE p_aggregation
            WHEN 'mean' THEN v_agg_value := v_stats.mean_value;
            WHEN 'median' THEN v_agg_value := v_stats.p50;
            WHEN 'p95' THEN v_agg_value := v_stats.p95;
            WHEN 'sum' THEN
                SELECT SUM(value) INTO v_agg_value
                FROM ab_test_metrics
                WHERE experiment_id = p_experiment_id
                AND metric_type = p_metric_type
                AND recorded_at >= p_start_date
                AND recorded_at < p_end_date + INTERVAL '1 day'
                AND (p_variant IS NULL OR variant = p_variant);
            WHEN 'count' THEN v_agg_value := v_stats.sample_count;
        END CASE;

        -- Insert baseline
        INSERT INTO ab_test_baselines (
            experiment_id, metric_type, aggregation, variant,
            value, sample_count, std_dev, min_value, max_value,
            p25, p50, p75, p95,
            period_start, period_end, week_number, year
        ) VALUES (
            p_experiment_id, p_metric_type, p_aggregation, p_variant,
            COALESCE(v_agg_value, 0), COALESCE(v_stats.sample_count, 0),
            v_stats.std_dev, v_stats.min_value, v_stats.max_value,
            v_stats.p25, v_stats.p50, v_stats.p75, v_stats.p95,
            p_start_date, p_end_date, v_week_number, v_year
        )
        ON CONFLICT (experiment_id, metric_type, aggregation, variant, period_start, period_end)
        DO UPDATE SET
            value = EXCLUDED.value,
            sample_count = EXCLUDED.sample_count,
            std_dev = EXCLUDED.std_dev,
            min_value = EXCLUDED.min_value,
            max_value = EXCLUDED.max_value,
            p25 = EXCLUDED.p25,
            p50 = EXCLUDED.p50,
            p75 = EXCLUDED.p75,
            p95 = EXCLUDED.p95,
            computed_at = NOW()
        RETURNING id INTO v_baseline_id;
    END;

    RETURN v_baseline_id;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION compute_ab_baseline IS 'Computes weekly baseline statistics for regression detection';

-- Function to detect regression
CREATE OR REPLACE FUNCTION detect_ab_regression(
    p_experiment_id UUID,
    p_warning_threshold FLOAT DEFAULT 5.0,  -- 5% drop
    p_critical_threshold FLOAT DEFAULT 10.0 -- 10% drop
)
RETURNS TABLE (
    metric_type ab_metric_type,
    severity regression_severity,
    baseline_value FLOAT,
    current_value FLOAT,
    percentage_change FLOAT,
    consecutive_days INTEGER
) AS $$
BEGIN
    RETURN QUERY
    WITH current_week AS (
        SELECT
            b.metric_type,
            b.aggregation,
            b.variant,
            b.value as current_value,
            b.sample_count,
            b.period_start
        FROM ab_test_baselines b
        WHERE b.experiment_id = p_experiment_id
        AND b.period_end >= CURRENT_DATE - INTERVAL '7 days'
        ORDER BY b.period_end DESC
        LIMIT 1
    ),
    previous_week AS (
        SELECT
            b.metric_type,
            b.aggregation,
            b.variant,
            b.value as baseline_value,
            b.sample_count,
            b.id as baseline_id
        FROM ab_test_baselines b
        WHERE b.experiment_id = p_experiment_id
        AND b.period_end < CURRENT_DATE - INTERVAL '7 days'
        AND b.period_end >= CURRENT_DATE - INTERVAL '14 days'
        ORDER BY b.period_end DESC
        LIMIT 1
    )
    SELECT
        cw.metric_type,
        CASE
            WHEN ABS((cw.current_value - pw.baseline_value) / NULLIF(pw.baseline_value, 0) * 100) >= p_critical_threshold
                THEN 'critical'::regression_severity
            ELSE 'warning'::regression_severity
        END as severity,
        pw.baseline_value,
        cw.current_value,
        (cw.current_value - pw.baseline_value) / NULLIF(pw.baseline_value, 0) * 100 as percentage_change,
        1 as consecutive_days -- Would need more logic for actual consecutive days
    FROM current_week cw
    JOIN previous_week pw ON cw.metric_type = pw.metric_type
        AND cw.aggregation = pw.aggregation
        AND (cw.variant IS NOT DISTINCT FROM pw.variant)
    WHERE
        -- For reward/relevance, negative change is regression
        (cw.metric_type IN ('reward_achieved', 'pattern_relevance', 'user_satisfaction')
            AND (cw.current_value - pw.baseline_value) / NULLIF(pw.baseline_value, 0) * 100 <= -p_warning_threshold)
        OR
        -- For retrieval_time, positive change (slower) is regression
        (cw.metric_type = 'retrieval_time'
            AND (cw.current_value - pw.baseline_value) / NULLIF(pw.baseline_value, 0) * 100 >= p_warning_threshold);
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION detect_ab_regression IS 'Detects week-over-week regression in A/B test metrics';

-- Function to update improvement target progress
CREATE OR REPLACE FUNCTION update_improvement_progress(
    p_target_id UUID,
    p_current_value FLOAT
)
RETURNS VOID AS $$
DECLARE
    v_target improvement_targets%ROWTYPE;
    v_progress FLOAT;
    v_is_achieved BOOLEAN;
BEGIN
    SELECT * INTO v_target FROM improvement_targets WHERE id = p_target_id;

    IF NOT FOUND THEN
        RETURN;
    END IF;

    -- Calculate progress
    IF v_target.target_value = v_target.baseline_value THEN
        v_progress := 1.0;
    ELSE
        v_progress := (p_current_value - v_target.baseline_value) /
                      (v_target.target_value - v_target.baseline_value);
    END IF;

    -- Check if achieved
    IF v_target.metric_type = 'retrieval_time' THEN
        -- Lower is better
        v_is_achieved := p_current_value <= v_target.target_value;
    ELSE
        -- Higher is better
        v_is_achieved := p_current_value >= v_target.target_value;
    END IF;

    -- Update target
    UPDATE improvement_targets SET
        current_value = p_current_value,
        progress = v_progress,
        is_achieved = v_is_achieved,
        achieved_at = CASE WHEN v_is_achieved AND NOT is_achieved THEN NOW() ELSE achieved_at END,
        updated_at = NOW()
    WHERE id = p_target_id;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION update_improvement_progress IS 'Updates the progress for an improvement target';

-- ============================================================================
-- VIEWS
-- ============================================================================

-- View for experiment summary statistics
CREATE OR REPLACE VIEW ab_test_summary AS
SELECT
    e.id as experiment_id,
    e.name as experiment_name,
    e.is_active,
    COUNT(DISTINCT a.session_id) as total_sessions,
    COUNT(DISTINCT a.session_id) FILTER (WHERE a.variant = 'control') as control_sessions,
    COUNT(DISTINCT a.session_id) FILTER (WHERE a.variant = 'treatment') as treatment_sessions,
    COUNT(m.id) as total_metrics,
    e.started_at,
    e.ends_at
FROM ab_test_experiments e
LEFT JOIN ab_test_assignments a ON a.experiment_id = e.id
LEFT JOIN ab_test_metrics m ON m.experiment_id = e.id
GROUP BY e.id, e.name, e.is_active, e.started_at, e.ends_at;

COMMENT ON VIEW ab_test_summary IS 'Summary statistics for A/B test experiments';

-- View for variant comparison
CREATE OR REPLACE VIEW ab_test_variant_comparison AS
SELECT
    e.id as experiment_id,
    e.name as experiment_name,
    m.metric_type,
    m.variant,
    COUNT(*) as sample_count,
    AVG(m.value) as mean_value,
    STDDEV_POP(m.value) as std_dev,
    MIN(m.value) as min_value,
    MAX(m.value) as max_value,
    PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY m.value) as median_value
FROM ab_test_experiments e
JOIN ab_test_metrics m ON m.experiment_id = e.id
GROUP BY e.id, e.name, m.metric_type, m.variant;

COMMENT ON VIEW ab_test_variant_comparison IS 'Comparison of metrics between Control and Treatment variants';

-- ============================================================================
-- DEFAULT EXPERIMENT
-- ============================================================================

-- Insert default SONA optimization experiment
INSERT INTO ab_test_experiments (
    name, split_ratio, description
) VALUES (
    'sona_retrieval_optimization',
    0.2,
    'A/B test comparing SONA-optimized retrieval (treatment) vs baseline (control). Default 20% treatment split.'
) ON CONFLICT (name) DO NOTHING;
