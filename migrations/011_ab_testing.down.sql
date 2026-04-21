-- Migration: 011_ab_testing (DOWN)
-- Description: Remove A/B Testing Infrastructure
-- Created: 2026-01-31

-- ============================================================================
-- DROP VIEWS
-- ============================================================================

DROP VIEW IF EXISTS ab_test_variant_comparison;
DROP VIEW IF EXISTS ab_test_summary;

-- ============================================================================
-- DROP FUNCTIONS
-- ============================================================================

DROP FUNCTION IF EXISTS update_improvement_progress(UUID, FLOAT);
DROP FUNCTION IF EXISTS detect_ab_regression(UUID, FLOAT, FLOAT);
DROP FUNCTION IF EXISTS compute_ab_baseline(UUID, ab_metric_type, metric_aggregation, ab_variant, DATE, DATE);
DROP FUNCTION IF EXISTS record_ab_metric(UUID, ab_variant, ab_metric_type, FLOAT, TEXT, TEXT, TEXT[], TEXT);
DROP FUNCTION IF EXISTS assign_ab_variant(UUID, TEXT, TEXT);

-- ============================================================================
-- DROP TRIGGERS
-- ============================================================================

DROP TRIGGER IF EXISTS improvement_targets_updated_at ON improvement_targets;
DROP TRIGGER IF EXISTS ab_test_experiments_updated_at ON ab_test_experiments;

-- ============================================================================
-- DROP TABLES
-- ============================================================================

DROP TABLE IF EXISTS improvement_targets;
DROP TABLE IF EXISTS regression_alerts;
DROP TABLE IF EXISTS ab_test_baselines;
DROP TABLE IF EXISTS ab_test_metrics;
DROP TABLE IF EXISTS ab_test_assignments;
DROP TABLE IF EXISTS ab_test_experiments;

-- ============================================================================
-- DROP TYPES
-- ============================================================================

DROP TYPE IF EXISTS regression_severity;
DROP TYPE IF EXISTS metric_aggregation;
DROP TYPE IF EXISTS ab_metric_type;
DROP TYPE IF EXISTS ab_variant;
