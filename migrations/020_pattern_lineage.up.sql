-- KOS P0: Pattern Lineage
-- Adds parent-child relationship tracking to reasoning_patterns.
-- Enables tracking merges, consolidations, improvements, forks, and transfers.

ALTER TABLE reasoning_patterns ADD COLUMN parent_id TEXT;
ALTER TABLE reasoning_patterns ADD COLUMN derivation_type TEXT;
ALTER TABLE reasoning_patterns ADD COLUMN lineage_depth INTEGER DEFAULT 0;

CREATE INDEX IF NOT EXISTS idx_patterns_parent_id ON reasoning_patterns(parent_id);
CREATE INDEX IF NOT EXISTS idx_patterns_derivation_type ON reasoning_patterns(derivation_type);
