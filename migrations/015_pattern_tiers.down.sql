-- SQLite doesn't support DROP COLUMN, so we recreate
-- For now, just drop indexes
DROP INDEX IF EXISTS idx_reasoning_patterns_tier;
DROP INDEX IF EXISTS idx_reasoning_patterns_tier_reward;
