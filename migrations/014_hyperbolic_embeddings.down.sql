-- Migration: 014_hyperbolic_embeddings (DOWN)
-- Description: Rollback hyperbolic embedding support
-- Created: 2026-02-06

-- Drop view first (depends on columns)
DROP VIEW IF EXISTS v_hierarchy_tree;

-- Drop indexes
DROP INDEX IF EXISTS idx_profdag_nodes_type_depth;
DROP INDEX IF EXISTS idx_profdag_nodes_hierarchy_depth;

-- Drop columns from profdag_nodes
ALTER TABLE profdag_nodes
    DROP COLUMN IF EXISTS hierarchy_depth;

ALTER TABLE profdag_nodes
    DROP COLUMN IF EXISTS hyperbolic_embedding;
