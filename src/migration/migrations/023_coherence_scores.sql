-- KOS P3: Coherence Scoring
CREATE TABLE IF NOT EXISTS coherence_scores (
    id              TEXT PRIMARY KEY,
    pattern_a_id    TEXT NOT NULL,
    pattern_b_id    TEXT NOT NULL,
    similarity      REAL NOT NULL,
    contradiction   REAL NOT NULL,
    coherence_type  TEXT NOT NULL,
    detected_at     TEXT NOT NULL,
    resolved        INTEGER DEFAULT 0,
    resolution_note TEXT,
    UNIQUE(pattern_a_id, pattern_b_id)
);

CREATE INDEX IF NOT EXISTS idx_coherence_contradiction ON coherence_scores(contradiction DESC);
CREATE INDEX IF NOT EXISTS idx_coherence_type ON coherence_scores(coherence_type);
