-- Migration: 001_initial_schema (rollback)
-- Description: Rollback initial PostgreSQL schema
-- Created: 2026-01-31
--
-- WARNING: This will delete all data in the affected tables!

-- ============================================================================
-- DROP TRIGGERS
-- ============================================================================

DROP TRIGGER IF EXISTS calibration_buckets_updated_at ON calibration_buckets;
DROP TRIGGER IF EXISTS sync_log_updated_at ON sync_log;
DROP TRIGGER IF EXISTS predictions_updated_at ON predictions;
DROP TRIGGER IF EXISTS patterns_updated_at ON patterns;
DROP TRIGGER IF EXISTS learnings_updated_at ON learnings;

-- Drop the trigger function
DROP FUNCTION IF EXISTS update_updated_at_column();

-- ============================================================================
-- DROP INDEXES (in case they weren't dropped with tables)
-- ============================================================================

-- Sync log indexes
DROP INDEX IF EXISTS sync_log_created_at_idx;
DROP INDEX IF EXISTS sync_log_next_retry_idx;
DROP INDEX IF EXISTS sync_log_status_idx;
DROP INDEX IF EXISTS sync_log_entity_idx;

-- Predictions indexes
DROP INDEX IF EXISTS predictions_brier_score_idx;
DROP INDEX IF EXISTS predictions_resolution_date_idx;
DROP INDEX IF EXISTS predictions_created_at_idx;
DROP INDEX IF EXISTS predictions_category_idx;
DROP INDEX IF EXISTS predictions_status_idx;

-- Patterns indexes
DROP INDEX IF EXISTS patterns_success_idx;
DROP INDEX IF EXISTS patterns_created_at_idx;
DROP INDEX IF EXISTS patterns_agent_id_idx;
DROP INDEX IF EXISTS patterns_tags_idx;
DROP INDEX IF EXISTS patterns_reward_idx;
DROP INDEX IF EXISTS patterns_category_idx;

-- Learnings indexes
DROP INDEX IF EXISTS learnings_created_at_idx;
DROP INDEX IF EXISTS learnings_agent_id_idx;
DROP INDEX IF EXISTS learnings_tags_idx;
DROP INDEX IF EXISTS learnings_category_idx;

-- ============================================================================
-- DROP TABLES
-- ============================================================================

-- Drop in reverse dependency order
DROP TABLE IF EXISTS calibration_buckets;
DROP TABLE IF EXISTS sync_log;
DROP TABLE IF EXISTS predictions;
DROP TABLE IF EXISTS patterns;
DROP TABLE IF EXISTS learnings;

-- ============================================================================
-- DROP CUSTOM TYPES
-- ============================================================================

DROP TYPE IF EXISTS sync_status;
DROP TYPE IF EXISTS sync_operation;
DROP TYPE IF EXISTS prediction_status;

-- ============================================================================
-- DROP EXTENSIONS (optional - may affect other schemas)
-- ============================================================================

-- Note: These extensions might be used by other applications/schemas.
-- Uncomment only if you're sure you want to remove them.

-- DROP EXTENSION IF EXISTS pg_trgm;
-- DROP EXTENSION IF EXISTS vector;
