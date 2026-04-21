-- KOS P0: Pattern Lineage (rollback)
-- Note: SQLite does not support DROP COLUMN directly.
-- This rollback drops and recreates the indexes only.

DROP INDEX IF EXISTS idx_patterns_parent_id;
DROP INDEX IF EXISTS idx_patterns_derivation_type;
