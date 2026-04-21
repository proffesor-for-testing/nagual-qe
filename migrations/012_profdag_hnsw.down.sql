-- Migration: 012_profdag_hnsw (rollback)
-- Description: Remove ProfDAG tables and HNSW indexes
-- Created: 2026-02-01
--
-- Note: This drops all ProfDAG data. Use with caution in production.

-- ============================================================================
-- DROP TRIGGERS
-- ============================================================================

DROP TRIGGER IF EXISTS profdag_edges_updated_at ON profdag_edges;
DROP TRIGGER IF EXISTS profdag_nodes_updated_at ON profdag_nodes;

-- ============================================================================
-- DROP INDEXES
-- ============================================================================

-- Edge indexes
DROP INDEX IF EXISTS profdag_edges_type_idx;
DROP INDEX IF EXISTS profdag_edges_target_idx;
DROP INDEX IF EXISTS profdag_edges_source_idx;

-- Node indexes
DROP INDEX IF EXISTS profdag_nodes_tags_idx;
DROP INDEX IF EXISTS profdag_nodes_type_proficiency_idx;
DROP INDEX IF EXISTS profdag_nodes_depth_idx;
DROP INDEX IF EXISTS profdag_nodes_parent_idx;
DROP INDEX IF EXISTS profdag_nodes_updated_at_idx;
DROP INDEX IF EXISTS profdag_nodes_proficiency_idx;
DROP INDEX IF EXISTS profdag_nodes_type_idx;

-- HNSW/IVFFlat indexes
DROP INDEX IF EXISTS profdag_nodes_embedding_ivfflat_idx;
DROP INDEX IF EXISTS profdag_nodes_embedding_hnsw_idx;

-- ============================================================================
-- DROP TABLES
-- ============================================================================

DROP TABLE IF EXISTS profdag_edges;
DROP TABLE IF EXISTS profdag_nodes;
