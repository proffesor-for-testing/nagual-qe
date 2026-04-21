-- Migration: 002_hnsw_indexes (rollback)
-- Description: Remove HNSW and supplementary indexes
-- Created: 2026-01-31
--
-- Note: Dropping these indexes does not affect data, only search performance.

-- ============================================================================
-- DROP TRIGRAM INDEXES
-- ============================================================================

DROP INDEX IF EXISTS learnings_content_trgm_idx;
DROP INDEX IF EXISTS patterns_solution_trgm_idx;
DROP INDEX IF EXISTS patterns_problem_trgm_idx;

-- ============================================================================
-- DROP B-TREE INDEXES
-- ============================================================================

DROP INDEX IF EXISTS learnings_updated_at_idx;
DROP INDEX IF EXISTS patterns_updated_at_idx;
DROP INDEX IF EXISTS patterns_category_reward_idx;
DROP INDEX IF EXISTS patterns_category_created_idx;

-- ============================================================================
-- DROP HNSW INDEXES
-- ============================================================================

DROP INDEX IF EXISTS learnings_embedding_hnsw_idx;
DROP INDEX IF EXISTS patterns_embedding_hnsw_idx;
