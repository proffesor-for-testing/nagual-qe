-- Migration: 010_context_graph
-- Description: Create context_graph table for Phase 2.C
-- Created: 2026-01-31
--
-- This migration creates the context_graph table for storing edges
-- between patterns, learnings, and predictions in the knowledge graph.
--
-- Supports 8 edge types:
-- - related_to: Generic relationship
-- - derived_from: Derivation relationship
-- - supersedes: Supersession relationship
-- - conflicts_with: Conflict between entities
-- - supports: Support relationship
-- - contradicts: Contradiction relationship
-- - co_retrieved: Co-retrieval pattern
-- - similar_to: Semantic similarity

-- ============================================================================
-- TABLE: context_graph
-- ============================================================================

CREATE TABLE IF NOT EXISTS context_graph (
    -- Primary key: Unique edge identifier
    id TEXT PRIMARY KEY,

    -- Source node ID (references patterns.id, learnings.id, or predictions.id)
    source_id TEXT NOT NULL,

    -- Target node ID (references patterns.id, learnings.id, or predictions.id)
    target_id TEXT NOT NULL,

    -- Edge type: one of the 8 relationship types
    edge_type TEXT NOT NULL CHECK (edge_type IN (
        'related_to', 'derived_from', 'supersedes', 'conflicts_with',
        'supports', 'contradicts', 'co_retrieved', 'similar_to'
    )),

    -- Relationship strength (0.0 to 1.0)
    -- Higher values indicate stronger relationships
    strength FLOAT NOT NULL CHECK (strength >= 0.0 AND strength <= 1.0),

    -- Optional metadata as JSONB
    metadata JSONB,

    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ,

    -- Prevent duplicate edges between same nodes with same type
    UNIQUE (source_id, target_id, edge_type)
);

-- Add table comment
COMMENT ON TABLE context_graph IS 'Context graph for tracking relationships between patterns, learnings, and predictions (Phase 2.C)';

-- Column comments
COMMENT ON COLUMN context_graph.id IS 'Unique edge identifier (format: edge_{source}_{target}_{type}_{timestamp})';
COMMENT ON COLUMN context_graph.source_id IS 'Source node ID - references patterns.id, learnings.id, or predictions.id';
COMMENT ON COLUMN context_graph.target_id IS 'Target node ID - references patterns.id, learnings.id, or predictions.id';
COMMENT ON COLUMN context_graph.edge_type IS 'Relationship type: related_to, derived_from, supersedes, conflicts_with, supports, contradicts, co_retrieved, similar_to';
COMMENT ON COLUMN context_graph.strength IS 'Relationship strength: 0.0-0.3 (weak), 0.3-0.7 (moderate), 0.7-1.0 (strong)';
COMMENT ON COLUMN context_graph.metadata IS 'Optional JSON metadata for the edge';

-- ============================================================================
-- INDEXES (Task 2.C.5)
-- ============================================================================

-- Index for outgoing edges from a node with edge type filter
-- Optimizes queries: SELECT * FROM context_graph WHERE source_id = ? AND edge_type = ?
CREATE INDEX IF NOT EXISTS idx_context_graph_source_type
    ON context_graph (source_id, edge_type);

COMMENT ON INDEX idx_context_graph_source_type IS 'Composite index for outgoing edge lookups by source and type';

-- Index for incoming edges to a node with edge type filter
-- Optimizes queries: SELECT * FROM context_graph WHERE target_id = ? AND edge_type = ?
CREATE INDEX IF NOT EXISTS idx_context_graph_target_type
    ON context_graph (target_id, edge_type);

COMMENT ON INDEX idx_context_graph_target_type IS 'Composite index for incoming edge lookups by target and type';

-- Index for strength-based pruning queries
-- Optimizes queries: DELETE FROM context_graph WHERE strength < ?
CREATE INDEX IF NOT EXISTS idx_context_graph_strength
    ON context_graph (strength DESC);

COMMENT ON INDEX idx_context_graph_strength IS 'Index for strength-based pruning and filtering queries';

-- Index for finding all edges of a specific type
-- Optimizes queries: SELECT * FROM context_graph WHERE edge_type = ?
CREATE INDEX IF NOT EXISTS idx_context_graph_edge_type
    ON context_graph (edge_type);

COMMENT ON INDEX idx_context_graph_edge_type IS 'Index for edge type lookups';

-- Index for finding all edges from a specific source
-- Optimizes queries: SELECT * FROM context_graph WHERE source_id = ?
CREATE INDEX IF NOT EXISTS idx_context_graph_source
    ON context_graph (source_id);

COMMENT ON INDEX idx_context_graph_source IS 'Index for all outgoing edges from a node';

-- Index for finding all edges to a specific target
-- Optimizes queries: SELECT * FROM context_graph WHERE target_id = ?
CREATE INDEX IF NOT EXISTS idx_context_graph_target
    ON context_graph (target_id);

COMMENT ON INDEX idx_context_graph_target IS 'Index for all incoming edges to a node';

-- ============================================================================
-- TRIGGER FOR updated_at
-- ============================================================================

-- Reuse the existing update_updated_at_column function from migration 001
CREATE TRIGGER context_graph_updated_at
    BEFORE UPDATE ON context_graph
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- ============================================================================
-- STATISTICS
-- ============================================================================

-- Update table statistics for query optimizer
ANALYZE context_graph;
