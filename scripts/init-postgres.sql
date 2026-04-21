-- Nagual RuVector-Postgres Init Script
-- Runs all .up.sql migrations in order during docker-entrypoint-initdb

-- Run schema migrations in order
-- (001 creates ruvector + pg_trgm extensions)
\i /docker-entrypoint-initdb.d/migrations/001_initial_schema.up.sql
\i /docker-entrypoint-initdb.d/migrations/002_hnsw_indexes.up.sql
\i /docker-entrypoint-initdb.d/migrations/008_audit_log.sql
\i /docker-entrypoint-initdb.d/migrations/009_retrieval_pairs.up.sql
\i /docker-entrypoint-initdb.d/migrations/010_context_graph.up.sql
\i /docker-entrypoint-initdb.d/migrations/011_ab_testing.up.sql
\i /docker-entrypoint-initdb.d/migrations/012_profdag_schema.up.sql
\i /docker-entrypoint-initdb.d/migrations/012_profdag_hnsw.up.sql
\i /docker-entrypoint-initdb.d/migrations/013_wormhole_tables.up.sql
\i /docker-entrypoint-initdb.d/migrations/014_hyperbolic_embeddings.up.sql

-- ReasoningBank patterns table (used by PatternStorage dual-write)
CREATE TABLE IF NOT EXISTS reasoning_patterns (
    id TEXT PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    category TEXT NOT NULL,
    problem TEXT NOT NULL,
    solution TEXT NOT NULL,
    context TEXT DEFAULT '',
    effectiveness DOUBLE PRECISION DEFAULT 0.5,
    reuse_count INTEGER DEFAULT 0,
    reward DOUBLE PRECISION DEFAULT 0.5,
    success BOOLEAN DEFAULT TRUE,
    critique TEXT DEFAULT '',
    agent_id TEXT,
    session_id TEXT,
    confidence DOUBLE PRECISION DEFAULT 0.5,
    embedding REAL[],
    tags JSONB DEFAULT '[]',
    related_patterns JSONB DEFAULT '[]',
    metadata JSONB DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_rp_category ON reasoning_patterns(category);
CREATE INDEX IF NOT EXISTS idx_rp_timestamp ON reasoning_patterns(timestamp);
CREATE INDEX IF NOT EXISTS idx_rp_updated_at ON reasoning_patterns(updated_at);
CREATE INDEX IF NOT EXISTS idx_rp_effectiveness ON reasoning_patterns(effectiveness);
CREATE INDEX IF NOT EXISTS idx_rp_agent_id ON reasoning_patterns(agent_id);
CREATE INDEX IF NOT EXISTS idx_rp_session_id ON reasoning_patterns(session_id);
CREATE INDEX IF NOT EXISTS idx_rp_tags ON reasoning_patterns USING GIN(tags);
