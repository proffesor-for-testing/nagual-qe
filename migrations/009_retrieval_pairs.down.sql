-- Rollback Migration: 009_retrieval_pairs
-- Description: Remove co-retrieval tracking and pattern edges tables

-- Drop functions first (they depend on tables)
DROP FUNCTION IF EXISTS get_coretrieval_candidates(INTEGER, INTEGER);
DROP FUNCTION IF EXISTS get_co_retrieval_count(TEXT, TEXT);
DROP FUNCTION IF EXISTS record_co_retrieval(TEXT, TEXT, TEXT, TEXT);

-- Drop triggers
DROP TRIGGER IF EXISTS pattern_edges_updated_at ON pattern_edges;

-- Drop indexes
DROP INDEX IF EXISTS edge_audit_log_job_id_idx;
DROP INDEX IF EXISTS edge_audit_log_created_at_idx;
DROP INDEX IF EXISTS edge_audit_log_operation_idx;
DROP INDEX IF EXISTS edge_audit_log_edge_id_idx;

DROP INDEX IF EXISTS pattern_edges_similar_idx;
DROP INDEX IF EXISTS pattern_edges_created_at_idx;
DROP INDEX IF EXISTS pattern_edges_auto_idx;
DROP INDEX IF EXISTS pattern_edges_weak_idx;
DROP INDEX IF EXISTS pattern_edges_type_idx;
DROP INDEX IF EXISTS pattern_edges_target_idx;
DROP INDEX IF EXISTS pattern_edges_source_idx;

DROP INDEX IF EXISTS retrieval_pairs_last_retrieved_idx;
DROP INDEX IF EXISTS retrieval_pairs_count_idx;
DROP INDEX IF EXISTS retrieval_pairs_pattern_b_idx;
DROP INDEX IF EXISTS retrieval_pairs_pattern_a_idx;

-- Drop tables
DROP TABLE IF EXISTS edge_audit_log;
DROP TABLE IF EXISTS pattern_edges;
DROP TABLE IF EXISTS retrieval_pairs;

-- Drop custom types
DROP TYPE IF EXISTS edge_operation;
DROP TYPE IF EXISTS edge_type;
