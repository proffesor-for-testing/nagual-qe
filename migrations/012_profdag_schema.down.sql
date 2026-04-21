-- Migration: 012_profdag_schema (DOWN)
-- Description: Rollback ProfDAG schema
-- Created: 2026-02-01

-- Drop triggers first
DROP TRIGGER IF EXISTS profdag_edges_updated_at ON profdag_edges;
DROP TRIGGER IF EXISTS profdag_nodes_updated_at ON profdag_nodes;

-- Drop indexes for profdag_edges
DROP INDEX IF EXISTS idx_profdag_edges_metadata;
DROP INDEX IF EXISTS idx_profdag_edges_wormhole;
DROP INDEX IF EXISTS idx_profdag_edges_similarity;
DROP INDEX IF EXISTS idx_profdag_edges_weight;
DROP INDEX IF EXISTS idx_profdag_edges_edge_type;
DROP INDEX IF EXISTS idx_profdag_edges_target_type;
DROP INDEX IF EXISTS idx_profdag_edges_source_type;
DROP INDEX IF EXISTS idx_profdag_edges_target;
DROP INDEX IF EXISTS idx_profdag_edges_source;

-- Drop indexes for profdag_nodes
DROP INDEX IF EXISTS idx_profdag_nodes_embedding_hnsw;
DROP INDEX IF EXISTS idx_profdag_nodes_metadata;
DROP INDEX IF EXISTS idx_profdag_nodes_created_at;
DROP INDEX IF EXISTS idx_profdag_nodes_confidence;
DROP INDEX IF EXISTS idx_profdag_nodes_importance;
DROP INDEX IF EXISTS idx_profdag_nodes_agent_session;
DROP INDEX IF EXISTS idx_profdag_nodes_source;
DROP INDEX IF EXISTS idx_profdag_nodes_type;

-- Drop tables (edges first due to foreign key constraints)
DROP TABLE IF EXISTS profdag_edges;
DROP TABLE IF EXISTS profdag_nodes;
