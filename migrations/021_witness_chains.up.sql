-- KOS P1: Witness Chains
-- Tamper-evident audit log for pattern mutations.
-- Each entry is hash-chained using BLAKE3 so retroactive modification is detectable.

CREATE TABLE IF NOT EXISTS witness_log (
    seq         INTEGER PRIMARY KEY AUTOINCREMENT,
    pattern_id  TEXT NOT NULL,
    operation   TEXT NOT NULL,
    action_hash BLOB NOT NULL,
    prev_hash   BLOB NOT NULL,
    entry_hash  BLOB NOT NULL,
    timestamp   TEXT NOT NULL,
    witness_type INTEGER NOT NULL DEFAULT 1,
    agent_id    TEXT,
    metadata    TEXT
);

CREATE INDEX IF NOT EXISTS idx_witness_pattern_id ON witness_log(pattern_id);
CREATE INDEX IF NOT EXISTS idx_witness_timestamp ON witness_log(timestamp);
