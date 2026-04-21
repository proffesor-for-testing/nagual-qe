-- Migration: 013_wormhole_tables (DOWN)
-- Description: Remove wormhole neural shortcuts tables
-- Created: 2026-02-02

-- Drop views first (they depend on tables)
DROP VIEW IF EXISTS v_wormhole_stats;
DROP VIEW IF EXISTS v_wormhole_candidates;
DROP VIEW IF EXISTS v_active_wormholes;

-- Drop triggers
DROP TRIGGER IF EXISTS trg_wormhole_created;
DROP TRIGGER IF EXISTS trg_wormhole_deactivated;
DROP TRIGGER IF EXISTS trg_wormhole_reactivated;

-- Drop indexes
DROP INDEX IF EXISTS idx_wormholes_source;
DROP INDEX IF EXISTS idx_wormholes_target;
DROP INDEX IF EXISTS idx_wormholes_active_strength;
DROP INDEX IF EXISTS idx_wormholes_last_used;
DROP INDEX IF EXISTS idx_wormholes_usage;

DROP INDEX IF EXISTS idx_co_access_count;
DROP INDEX IF EXISTS idx_co_access_pattern_a;
DROP INDEX IF EXISTS idx_co_access_pattern_b;
DROP INDEX IF EXISTS idx_co_access_last_accessed;

DROP INDEX IF EXISTS idx_usage_log_wormhole;
DROP INDEX IF EXISTS idx_usage_log_event_type;
DROP INDEX IF EXISTS idx_usage_log_timestamp;

-- Drop tables (order matters due to potential foreign key relationships)
DROP TABLE IF EXISTS wormhole_usage_log;
DROP TABLE IF EXISTS wormhole_co_access;
DROP TABLE IF EXISTS wormholes;
