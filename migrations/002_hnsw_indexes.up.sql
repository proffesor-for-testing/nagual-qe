-- Migration: 002_hnsw_indexes
-- Description: Create HNSW indexes for vector similarity search
-- Created: 2026-01-31
--
-- HNSW (Hierarchical Navigable Small World) indexes enable fast
-- approximate nearest neighbor search for vector embeddings.
--
-- Configuration parameters per ADR-002:
--   m = 16         : Maximum number of connections per layer
--   ef_construction = 64 : Size of dynamic candidate list during build
--
-- Higher m values increase recall but use more memory.
-- Higher ef_construction improves index quality but slows build time.
--
-- For queries, set probes/ef_search based on accuracy needs:
--   SET hnsw.ef_search = 100;  -- Default is 40

-- ============================================================================
-- HNSW INDEXES FOR SEMANTIC SEARCH
-- ============================================================================
-- NOTE: HNSW index AM in ruvector-postgres v0.1.0 has incomplete
-- connect_node_to_neighbors (stub), causing INSERT hangs on arm64.
-- Disabled until ruvector-postgres matures. Sequential scan with
-- cosine distance works correctly for datasets < 10k vectors.
--
-- CREATE INDEX IF NOT EXISTS patterns_embedding_hnsw_idx
--     ON patterns
--     USING hnsw (embedding ruvector_cosine_ops)
--     WITH (m = 16, ef_construction = 64);
-- COMMENT ON INDEX patterns_embedding_hnsw_idx IS 'HNSW index for semantic pattern search using cosine similarity';
--
-- CREATE INDEX IF NOT EXISTS learnings_embedding_hnsw_idx
--     ON learnings
--     USING hnsw (embedding ruvector_cosine_ops)
--     WITH (m = 16, ef_construction = 64);
-- COMMENT ON INDEX learnings_embedding_hnsw_idx IS 'HNSW index for semantic learning search using cosine similarity';

-- ============================================================================
-- ADDITIONAL B-TREE INDEXES FOR COMMON QUERY PATTERNS
-- ============================================================================

-- Composite index for patterns filtered by domain and time
-- Useful for queries like: "Get recent patterns in category X"
CREATE INDEX IF NOT EXISTS patterns_category_created_idx
    ON patterns (category, created_at DESC);

COMMENT ON INDEX patterns_category_created_idx IS 'Composite index for category-filtered time-ordered queries';

-- Composite index for patterns filtered by domain and reward
-- Useful for queries like: "Get highest-reward patterns in category X"
CREATE INDEX IF NOT EXISTS patterns_category_reward_idx
    ON patterns (category, reward DESC);

COMMENT ON INDEX patterns_category_reward_idx IS 'Composite index for category-filtered reward-ordered queries';

-- Index for recently updated patterns
-- Useful for sync operations and change tracking
CREATE INDEX IF NOT EXISTS patterns_updated_at_idx
    ON patterns (updated_at DESC);

COMMENT ON INDEX patterns_updated_at_idx IS 'Index for tracking recently modified patterns';

-- Index for recently updated learnings
CREATE INDEX IF NOT EXISTS learnings_updated_at_idx
    ON learnings (updated_at DESC);

COMMENT ON INDEX learnings_updated_at_idx IS 'Index for tracking recently modified learnings';

-- ============================================================================
-- TRIGRAM INDEXES FOR FUZZY TEXT SEARCH
-- ============================================================================

-- Trigram index on patterns.problem for fuzzy matching
CREATE INDEX IF NOT EXISTS patterns_problem_trgm_idx
    ON patterns
    USING gin (problem gin_trgm_ops);

COMMENT ON INDEX patterns_problem_trgm_idx IS 'Trigram index for fuzzy problem text search';

-- Trigram index on patterns.solution for fuzzy matching
CREATE INDEX IF NOT EXISTS patterns_solution_trgm_idx
    ON patterns
    USING gin (solution gin_trgm_ops)
    WHERE solution IS NOT NULL;

COMMENT ON INDEX patterns_solution_trgm_idx IS 'Trigram index for fuzzy solution text search';

-- Trigram index on learnings.content for fuzzy matching
CREATE INDEX IF NOT EXISTS learnings_content_trgm_idx
    ON learnings
    USING gin (content gin_trgm_ops);

COMMENT ON INDEX learnings_content_trgm_idx IS 'Trigram index for fuzzy content search';

-- ============================================================================
-- CONFIGURATION RECOMMENDATIONS
-- ============================================================================

-- To improve search accuracy at query time, you can set ef_search:
--
--   -- For high accuracy (slower):
--   SET hnsw.ef_search = 200;
--
--   -- For balanced performance (default):
--   SET hnsw.ef_search = 100;
--
--   -- For fastest queries (lower accuracy):
--   SET hnsw.ef_search = 40;
--
-- Example semantic search query:
--
--   SET hnsw.ef_search = 100;
--   SELECT id, problem, solution, embedding <=> $1 AS distance
--   FROM patterns
--   WHERE category = 'testing'
--   ORDER BY embedding <=> $1
--   LIMIT 10;

-- ============================================================================
-- STATISTICS UPDATE
-- ============================================================================

-- Analyze tables to update statistics after index creation
ANALYZE learnings;
ANALYZE patterns;
