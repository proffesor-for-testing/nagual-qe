-- Migration: 012_profdag_schema
-- Description: Create ProfDAG tables for Probabilistic Forecasting Directed Acyclic Graph
-- Created: 2026-02-01
--
-- ProfDAG (Probabilistic Forecasting DAG) is a graph-based knowledge representation
-- for tracking patterns, trajectories, predictions, and decisions with temporal
-- and semantic relationships. The graph enables:
--
-- - Pattern evolution tracking over time
-- - Trajectory-based reasoning across agent sessions
-- - Decision support through wormhole connections
-- - Temporal relationships for forecasting
-- - Semantic similarity edges for knowledge retrieval
--
-- Node Types:
-- - pattern: A learned problem-solution pair
-- - trajectory: A sequence of actions/decisions in a session
-- - prediction: A probabilistic forecast
-- - decision: A choice point with outcomes
--
-- Edge Types:
-- - leads_to: Causal/temporal progression
-- - similar_to: Semantic similarity between nodes
-- - derived_from: One node was derived from another
-- - wormhole: Non-local connection bridging distant parts of the graph
-- - temporal_link: Time-based relationship

-- ============================================================================
-- TABLE: profdag_nodes
-- ============================================================================

CREATE TABLE IF NOT EXISTS profdag_nodes (
    -- Primary key: UUID for distributed generation
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Node type classification
    node_type TEXT NOT NULL CHECK (node_type IN (
        'pattern', 'trajectory', 'prediction', 'decision'
    )),

    -- Content: The actual data/description for this node
    content TEXT NOT NULL,

    -- Vector embedding for semantic search (128-dimensional per project standard)
    embedding ruvector(128),

    -- Flexible metadata storage
    metadata JSONB DEFAULT '{}',

    -- Reference to source entity (pattern_id, prediction_id, etc.)
    source_id TEXT,
    source_type TEXT,

    -- Quality metrics
    confidence FLOAT DEFAULT 0.5 CHECK (confidence >= 0.0 AND confidence <= 1.0),
    importance FLOAT DEFAULT 0.5 CHECK (importance >= 0.0 AND importance <= 1.0),

    -- Session tracking
    agent_id TEXT,
    session_id TEXT,

    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ
);

-- Add table comment
COMMENT ON TABLE profdag_nodes IS 'ProfDAG nodes representing patterns, trajectories, predictions, and decisions';

-- Column comments
COMMENT ON COLUMN profdag_nodes.id IS 'Unique node identifier (UUID)';
COMMENT ON COLUMN profdag_nodes.node_type IS 'Node classification: pattern, trajectory, prediction, decision';
COMMENT ON COLUMN profdag_nodes.content IS 'The actual content/description for this node';
COMMENT ON COLUMN profdag_nodes.embedding IS '128-dimensional vector embedding for semantic search';
COMMENT ON COLUMN profdag_nodes.metadata IS 'Flexible JSON metadata for node-specific attributes';
COMMENT ON COLUMN profdag_nodes.source_id IS 'Reference to source entity in another table';
COMMENT ON COLUMN profdag_nodes.source_type IS 'Type of source entity (reasoning_patterns, predictions, etc.)';
COMMENT ON COLUMN profdag_nodes.confidence IS 'Confidence score for this node (0.0-1.0)';
COMMENT ON COLUMN profdag_nodes.importance IS 'Importance/weight of this node in the graph (0.0-1.0)';

-- ============================================================================
-- TABLE: profdag_edges
-- ============================================================================

CREATE TABLE IF NOT EXISTS profdag_edges (
    -- Composite primary key using source, target, and type ensures no duplicate edges
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Source node reference
    source_id UUID NOT NULL REFERENCES profdag_nodes(id) ON DELETE CASCADE,

    -- Target node reference
    target_id UUID NOT NULL REFERENCES profdag_nodes(id) ON DELETE CASCADE,

    -- Edge type classification
    edge_type TEXT NOT NULL CHECK (edge_type IN (
        'leads_to', 'similar_to', 'derived_from', 'wormhole', 'temporal_link'
    )),

    -- Edge weight/strength (0.0 to 1.0)
    -- Higher values indicate stronger relationships
    weight FLOAT NOT NULL DEFAULT 0.5 CHECK (weight >= 0.0 AND weight <= 1.0),

    -- Flexible metadata storage
    metadata JSONB DEFAULT '{}',

    -- Temporal metadata for temporal_link edges
    temporal_distance_hours INTEGER,
    temporal_direction TEXT CHECK (temporal_direction IS NULL OR temporal_direction IN ('forward', 'backward', 'concurrent')),

    -- Similarity metadata for similar_to edges
    similarity_score FLOAT CHECK (similarity_score IS NULL OR (similarity_score >= 0.0 AND similarity_score <= 1.0)),

    -- Wormhole-specific metadata
    wormhole_strength FLOAT CHECK (wormhole_strength IS NULL OR (wormhole_strength >= 0.0 AND wormhole_strength <= 1.0)),
    wormhole_reason TEXT,

    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ,

    -- Prevent duplicate edges between same nodes with same type
    UNIQUE (source_id, target_id, edge_type),

    -- Prevent self-loops (a node cannot connect to itself)
    CHECK (source_id != target_id)
);

-- Add table comment
COMMENT ON TABLE profdag_edges IS 'ProfDAG edges representing relationships between nodes';

-- Column comments
COMMENT ON COLUMN profdag_edges.source_id IS 'Source node UUID reference';
COMMENT ON COLUMN profdag_edges.target_id IS 'Target node UUID reference';
COMMENT ON COLUMN profdag_edges.edge_type IS 'Relationship type: leads_to, similar_to, derived_from, wormhole, temporal_link';
COMMENT ON COLUMN profdag_edges.weight IS 'Edge strength: 0.0-0.3 (weak), 0.3-0.7 (moderate), 0.7-1.0 (strong)';
COMMENT ON COLUMN profdag_edges.temporal_distance_hours IS 'For temporal_link: time difference in hours between nodes';
COMMENT ON COLUMN profdag_edges.temporal_direction IS 'For temporal_link: direction of time flow';
COMMENT ON COLUMN profdag_edges.similarity_score IS 'For similar_to: cosine similarity between embeddings';
COMMENT ON COLUMN profdag_edges.wormhole_strength IS 'For wormhole: strength of the non-local connection';
COMMENT ON COLUMN profdag_edges.wormhole_reason IS 'For wormhole: explanation of why nodes are connected';

-- ============================================================================
-- INDEXES FOR profdag_nodes
-- ============================================================================

-- Index for node type lookups
CREATE INDEX IF NOT EXISTS idx_profdag_nodes_type
    ON profdag_nodes (node_type);

COMMENT ON INDEX idx_profdag_nodes_type IS 'Index for filtering nodes by type';

-- Index for source lookups (finding nodes by their source entity)
CREATE INDEX IF NOT EXISTS idx_profdag_nodes_source
    ON profdag_nodes (source_type, source_id);

COMMENT ON INDEX idx_profdag_nodes_source IS 'Index for finding nodes by source entity';

-- Index for agent/session lookups
CREATE INDEX IF NOT EXISTS idx_profdag_nodes_agent_session
    ON profdag_nodes (agent_id, session_id);

COMMENT ON INDEX idx_profdag_nodes_agent_session IS 'Index for finding nodes by agent and session';

-- Index for importance-based queries
CREATE INDEX IF NOT EXISTS idx_profdag_nodes_importance
    ON profdag_nodes (importance DESC);

COMMENT ON INDEX idx_profdag_nodes_importance IS 'Index for importance-based node retrieval';

-- Index for confidence-based queries
CREATE INDEX IF NOT EXISTS idx_profdag_nodes_confidence
    ON profdag_nodes (confidence DESC);

COMMENT ON INDEX idx_profdag_nodes_confidence IS 'Index for confidence-based node retrieval';

-- Index for timestamp-based queries
CREATE INDEX IF NOT EXISTS idx_profdag_nodes_created_at
    ON profdag_nodes (created_at DESC);

COMMENT ON INDEX idx_profdag_nodes_created_at IS 'Index for time-based node queries';

-- GIN index on metadata for JSONB queries
CREATE INDEX IF NOT EXISTS idx_profdag_nodes_metadata
    ON profdag_nodes USING GIN (metadata);

COMMENT ON INDEX idx_profdag_nodes_metadata IS 'GIN index for JSONB metadata queries';

-- ============================================================================
-- INDEXES FOR profdag_edges
-- ============================================================================

-- Index for outgoing edges from a node
CREATE INDEX IF NOT EXISTS idx_profdag_edges_source
    ON profdag_edges (source_id);

COMMENT ON INDEX idx_profdag_edges_source IS 'Index for finding all outgoing edges from a node';

-- Index for incoming edges to a node
CREATE INDEX IF NOT EXISTS idx_profdag_edges_target
    ON profdag_edges (target_id);

COMMENT ON INDEX idx_profdag_edges_target IS 'Index for finding all incoming edges to a node';

-- Composite index for source with edge type
CREATE INDEX IF NOT EXISTS idx_profdag_edges_source_type
    ON profdag_edges (source_id, edge_type);

COMMENT ON INDEX idx_profdag_edges_source_type IS 'Composite index for outgoing edges by type';

-- Composite index for target with edge type
CREATE INDEX IF NOT EXISTS idx_profdag_edges_target_type
    ON profdag_edges (target_id, edge_type);

COMMENT ON INDEX idx_profdag_edges_target_type IS 'Composite index for incoming edges by type';

-- Index for edge type lookups
CREATE INDEX IF NOT EXISTS idx_profdag_edges_edge_type
    ON profdag_edges (edge_type);

COMMENT ON INDEX idx_profdag_edges_edge_type IS 'Index for finding all edges of a specific type';

-- Index for weight-based pruning queries
CREATE INDEX IF NOT EXISTS idx_profdag_edges_weight
    ON profdag_edges (weight DESC);

COMMENT ON INDEX idx_profdag_edges_weight IS 'Index for weight-based edge filtering and pruning';

-- Index for similar_to edges by similarity score
CREATE INDEX IF NOT EXISTS idx_profdag_edges_similarity
    ON profdag_edges (similarity_score DESC)
    WHERE edge_type = 'similar_to' AND similarity_score IS NOT NULL;

COMMENT ON INDEX idx_profdag_edges_similarity IS 'Partial index for similar_to edges by similarity score';

-- Index for wormhole edges
CREATE INDEX IF NOT EXISTS idx_profdag_edges_wormhole
    ON profdag_edges (wormhole_strength DESC)
    WHERE edge_type = 'wormhole' AND wormhole_strength IS NOT NULL;

COMMENT ON INDEX idx_profdag_edges_wormhole IS 'Partial index for wormhole edges by strength';

-- GIN index on edge metadata
CREATE INDEX IF NOT EXISTS idx_profdag_edges_metadata
    ON profdag_edges USING GIN (metadata);

COMMENT ON INDEX idx_profdag_edges_metadata IS 'GIN index for JSONB edge metadata queries';

-- ============================================================================
-- HNSW VECTOR INDEX FOR SEMANTIC SEARCH
-- ============================================================================

-- HNSW index on node embeddings for fast approximate nearest neighbor search
-- Using cosine distance (1 - cosine_similarity) for semantic similarity
CREATE INDEX IF NOT EXISTS idx_profdag_nodes_embedding_hnsw
    ON profdag_nodes USING hnsw (embedding ruvector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

COMMENT ON INDEX idx_profdag_nodes_embedding_hnsw IS 'HNSW index for fast semantic similarity search on node embeddings';

-- ============================================================================
-- TRIGGERS FOR updated_at
-- ============================================================================

-- Reuse the existing update_updated_at_column function from migration 001
CREATE TRIGGER profdag_nodes_updated_at
    BEFORE UPDATE ON profdag_nodes
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER profdag_edges_updated_at
    BEFORE UPDATE ON profdag_edges
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- ============================================================================
-- STATISTICS
-- ============================================================================

-- Update table statistics for query optimizer
ANALYZE profdag_nodes;
ANALYZE profdag_edges;
