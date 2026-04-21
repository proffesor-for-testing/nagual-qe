-- Migration: 014_hyperbolic_embeddings
-- Description: Add hyperbolic embedding support to ProfDAG nodes
-- Created: 2026-02-06
--
-- PROFDAG-009: Hyperbolic Embeddings for Hierarchical Knowledge
--
-- Hyperbolic geometry (Poincare ball model) naturally represents hierarchical
-- structures because the volume of a hyperbolic ball grows exponentially with
-- radius, matching the exponential growth of tree structures.
--
-- Points near the origin represent general/root concepts.
-- Points near the boundary represent specific/leaf concepts.
-- The Poincare distance respects hierarchical relationships.

-- ============================================================================
-- ALTER TABLE: profdag_nodes - Add hyperbolic embedding columns
-- ============================================================================

-- Hyperbolic embedding stored as JSON array of f64 coordinates in the Poincare ball.
-- Unlike the Euclidean `embedding` column (pgvector), hyperbolic embeddings require
-- custom distance functions, so we store them as JSON and compute distances in Rust.
ALTER TABLE profdag_nodes
    ADD COLUMN IF NOT EXISTS hyperbolic_embedding TEXT;

COMMENT ON COLUMN profdag_nodes.hyperbolic_embedding IS
    'Poincare ball embedding as JSON array of f64 coordinates. Points near origin are general, near boundary are specific.';

-- Hierarchy depth: 0.0 = root/general concept, 1.0 = leaf/specific concept.
-- Derived from the norm of the hyperbolic embedding relative to the ball radius.
ALTER TABLE profdag_nodes
    ADD COLUMN IF NOT EXISTS hierarchy_depth REAL DEFAULT NULL
    CHECK (hierarchy_depth IS NULL OR (hierarchy_depth >= 0.0 AND hierarchy_depth <= 1.0));

COMMENT ON COLUMN profdag_nodes.hierarchy_depth IS
    'Hierarchy depth: 0.0 = root/general concept, 1.0 = leaf/specific concept. Derived from Poincare ball norm.';

-- ============================================================================
-- INDEXES
-- ============================================================================

-- Index on hierarchy_depth for efficient depth-based queries
-- (e.g., find all root concepts, find all leaf concepts, range queries)
CREATE INDEX IF NOT EXISTS idx_profdag_nodes_hierarchy_depth
    ON profdag_nodes (hierarchy_depth ASC)
    WHERE hierarchy_depth IS NOT NULL;

COMMENT ON INDEX idx_profdag_nodes_hierarchy_depth IS
    'Index for hierarchy depth queries (find roots, leaves, depth ranges)';

-- Composite index for node_type + hierarchy_depth (common query pattern:
-- "find all pattern nodes at a certain hierarchy level")
CREATE INDEX IF NOT EXISTS idx_profdag_nodes_type_depth
    ON profdag_nodes (node_type, hierarchy_depth ASC)
    WHERE hierarchy_depth IS NOT NULL;

COMMENT ON INDEX idx_profdag_nodes_type_depth IS
    'Composite index for filtering nodes by type and hierarchy depth';

-- ============================================================================
-- VIEW: v_hierarchy_tree
-- ============================================================================

-- View that presents the ProfDAG nodes ordered by hierarchy depth,
-- making it easy to traverse the knowledge hierarchy from general to specific.
CREATE OR REPLACE VIEW v_hierarchy_tree AS
SELECT
    id,
    node_type,
    content,
    hierarchy_depth,
    confidence,
    importance,
    -- Classify into hierarchy tiers for grouping
    CASE
        WHEN hierarchy_depth IS NULL THEN 'unclassified'
        WHEN hierarchy_depth < 0.2 THEN 'root'
        WHEN hierarchy_depth < 0.4 THEN 'high_level'
        WHEN hierarchy_depth < 0.6 THEN 'mid_level'
        WHEN hierarchy_depth < 0.8 THEN 'detailed'
        ELSE 'leaf'
    END AS hierarchy_tier,
    -- Flag indicating if hyperbolic embedding is available
    (hyperbolic_embedding IS NOT NULL) AS has_hyperbolic_embedding,
    created_at,
    updated_at
FROM profdag_nodes
ORDER BY
    COALESCE(hierarchy_depth, 999.0) ASC,
    importance DESC,
    created_at DESC;

COMMENT ON VIEW v_hierarchy_tree IS
    'Hierarchical view of ProfDAG nodes ordered by depth (general to specific)';

-- ============================================================================
-- STATISTICS
-- ============================================================================

ANALYZE profdag_nodes;
