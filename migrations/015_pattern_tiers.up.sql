-- Add tier column to reasoning_patterns
ALTER TABLE reasoning_patterns ADD COLUMN tier TEXT NOT NULL DEFAULT 'booster' CHECK(tier IN ('booster', 'crystal', 'reflex'));

-- Index for tier-based queries
CREATE INDEX IF NOT EXISTS idx_reasoning_patterns_tier ON reasoning_patterns(tier);

-- Composite index for tier + reward (for promotion queries)
CREATE INDEX IF NOT EXISTS idx_reasoning_patterns_tier_reward ON reasoning_patterns(tier, reward);
