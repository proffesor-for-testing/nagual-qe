-- Migration: 010_context_graph (rollback)
-- Description: Drop context_graph table and related objects
-- Created: 2026-01-31

-- Drop trigger
DROP TRIGGER IF EXISTS context_graph_updated_at ON context_graph;

-- Drop indexes (will be dropped automatically with table, but explicit for clarity)
DROP INDEX IF EXISTS idx_context_graph_source_type;
DROP INDEX IF EXISTS idx_context_graph_target_type;
DROP INDEX IF EXISTS idx_context_graph_strength;
DROP INDEX IF EXISTS idx_context_graph_edge_type;
DROP INDEX IF EXISTS idx_context_graph_source;
DROP INDEX IF EXISTS idx_context_graph_target;

-- Drop table
DROP TABLE IF EXISTS context_graph;
