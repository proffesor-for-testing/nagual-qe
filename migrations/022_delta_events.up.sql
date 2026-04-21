-- KOS P2: Delta Event Sourcing
-- Field-level change tracking for patterns with time-travel reconstruction.
-- Each delta records the exact fields that changed, enabling
-- reconstruct_at(timestamp) queries and change velocity analysis.

CREATE TABLE IF NOT EXISTS pattern_deltas (
    id          TEXT PRIMARY KEY,
    pattern_id  TEXT NOT NULL,
    seq         INTEGER NOT NULL,
    timestamp   TEXT NOT NULL,
    operation   TEXT NOT NULL,
    field_diffs TEXT NOT NULL,
    agent_id    TEXT,
    snapshot    TEXT,
    UNIQUE(pattern_id, seq)
);

CREATE INDEX IF NOT EXISTS idx_deltas_pattern_id ON pattern_deltas(pattern_id);
CREATE INDEX IF NOT EXISTS idx_deltas_timestamp ON pattern_deltas(timestamp);
