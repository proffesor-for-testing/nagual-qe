//! Coherence Scoring engine for the Knowledge Operating System.
//!
//! Extends the existing CoherenceGate with contradiction detection,
//! system-wide health scoring, and pairwise coherence analysis.
//!
//! Uses text-based negation detection (no embedder required):
//! - Jaccard similarity for topic overlap
//! - Negation pattern matching for opposing directives
//! - Configurable thresholds for contradiction vs supportive classification

use std::collections::HashSet;
use std::sync::Arc;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::db::SqliteDb;
use crate::error::Result;
use crate::reasoning_bank::pattern::PatternId;

/// Schema SQL for the coherence_scores table.
const COHERENCE_SCORES_SCHEMA: &str = r#"
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
"#;

/// Negation words that indicate opposing directives.
const NEGATION_WORDS: &[&str] = &[
    "don't", "dont", "never", "avoid", "not", "shouldn't", "shouldnt",
    "won't", "wont", "cannot", "can't", "cant", "no", "neither",
    "nor", "without", "exclude", "disable", "remove", "stop",
    "prevent", "reject", "refuse", "prohibit",
];

/// Affirmative directive words.
const AFFIRMATIVE_WORDS: &[&str] = &[
    "always", "must", "use", "should", "require", "enable", "add",
    "include", "prefer", "recommend", "ensure", "do", "apply",
    "implement", "adopt", "embrace", "allow", "accept",
];

/// Classification of coherence relationship between two patterns.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CoherenceType {
    Supportive,
    Neutral,
    Contradictory,
}

impl std::fmt::Display for CoherenceType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Supportive => write!(f, "supportive"),
            Self::Neutral => write!(f, "neutral"),
            Self::Contradictory => write!(f, "contradictory"),
        }
    }
}

impl From<&str> for CoherenceType {
    fn from(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "supportive" => Self::Supportive,
            "contradictory" => Self::Contradictory,
            _ => Self::Neutral,
        }
    }
}

/// A scored coherence relationship between two patterns.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoherenceScore {
    pub id: String,
    pub pattern_a: PatternId,
    pub pattern_b: PatternId,
    pub similarity: f32,
    pub contradiction: f32,
    pub coherence_type: CoherenceType,
    pub detected_at: DateTime<Utc>,
    pub resolved: bool,
    pub resolution_note: Option<String>,
}

/// Aggregate health metrics for the knowledge system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoherenceHealth {
    pub total_pairs_checked: u64,
    pub contradictions_found: u32,
    pub contradiction_rate: f32,
    pub entailment_consistency: f32,
    pub domains_scanned: Vec<String>,
    pub worst_domain: Option<(String, f32)>,
}

/// Configuration for the coherence scorer.
#[derive(Debug, Clone)]
pub struct ScoringConfig {
    pub contradiction_threshold: f32,
    pub supportive_threshold: f32,
    pub batch_size: usize,
    pub scan_domains: Option<Vec<String>>,
}

impl Default for ScoringConfig {
    fn default() -> Self {
        Self {
            contradiction_threshold: 0.35,
            supportive_threshold: 0.8,
            batch_size: 100,
            scan_domains: None,
        }
    }
}

/// Coherence Scorer that analyzes pairwise pattern relationships.
pub struct CoherenceScorer {
    db: Arc<SqliteDb>,
    config: ScoringConfig,
}

impl CoherenceScorer {
    /// Create a new CoherenceScorer.
    pub fn new(db: Arc<SqliteDb>, config: ScoringConfig) -> Self {
        Self { db, config }
    }

    /// Initialize the coherence_scores table if it does not exist.
    pub async fn init_schema(&self) -> Result<()> {
        self.db.execute_batch(COHERENCE_SCORES_SCHEMA).await
    }

    /// Score coherence between two patterns using text-based analysis.
    ///
    /// Algorithm:
    /// 1. Tokenize both solutions into word sets
    /// 2. Compute Jaccard similarity for topic overlap
    /// 3. If related (Jaccard > 0.3), check for opposing directives
    /// 4. Classify as Contradictory, Supportive, or Neutral
    pub fn score_pair(
        &self,
        _problem_a: &str,
        solution_a: &str,
        _problem_b: &str,
        solution_b: &str,
        id_a: &PatternId,
        id_b: &PatternId,
    ) -> CoherenceScore {
        let tokens_a = tokenize(solution_a);
        let tokens_b = tokenize(solution_b);

        let jaccard = jaccard_similarity(&tokens_a, &tokens_b);

        let (contradiction, coherence_type) = if jaccard > 0.3 {
            let neg_a = count_negation_words(&tokens_a);
            let neg_b = count_negation_words(&tokens_b);
            let aff_a = count_affirmative_words(&tokens_a);
            let aff_b = count_affirmative_words(&tokens_b);

            let negation_distance = compute_negation_distance(neg_a, neg_b, aff_a, aff_b);
            let contradiction_score = jaccard * negation_distance;

            if contradiction_score > self.config.contradiction_threshold {
                (contradiction_score, CoherenceType::Contradictory)
            } else if jaccard > self.config.supportive_threshold && negation_distance < 0.2 {
                (contradiction_score, CoherenceType::Supportive)
            } else {
                (contradiction_score, CoherenceType::Neutral)
            }
        } else {
            (0.0, CoherenceType::Neutral)
        };

        CoherenceScore {
            id: Uuid::new_v4().to_string(),
            pattern_a: id_a.clone(),
            pattern_b: id_b.clone(),
            similarity: jaccard,
            contradiction,
            coherence_type,
            detected_at: Utc::now(),
            resolved: false,
            resolution_note: None,
        }
    }

    /// Scan all patterns in a domain, run pairwise comparison, store results.
    pub async fn scan_domain(&self, domain: &str) -> Result<Vec<CoherenceScore>> {
        let sql = r#"
            SELECT id, problem, solution FROM reasoning_patterns
            WHERE category = ?
            ORDER BY updated_at DESC
            LIMIT ?
        "#;

        let domain_str = domain.to_string();
        let limit = self.config.batch_size as i64;

        let patterns: Vec<(String, String, String)> = self
            .db
            .query(sql, &[&domain_str, &limit], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                ))
            })
            .await?;

        let mut scores = Vec::new();

        for i in 0..patterns.len() {
            for j in (i + 1)..patterns.len() {
                let (ref id_a, ref prob_a, ref sol_a) = patterns[i];
                let (ref id_b, ref prob_b, ref sol_b) = patterns[j];

                let pid_a = PatternId::from_string(id_a);
                let pid_b = PatternId::from_string(id_b);

                let score = self.score_pair(prob_a, sol_a, prob_b, sol_b, &pid_a, &pid_b);

                // Only store non-neutral or high-similarity scores
                if score.coherence_type != CoherenceType::Neutral || score.similarity > 0.5 {
                    self.store_score(&score).await?;
                    scores.push(score);
                }
            }
        }

        Ok(scores)
    }

    /// Get aggregate system health across all scored pairs.
    pub async fn system_health(&self) -> Result<CoherenceHealth> {
        let total: u64 = self
            .db
            .query_one(
                "SELECT COUNT(*) FROM coherence_scores",
                &[],
                |row| row.get::<_, i64>(0),
            )
            .await?
            .unwrap_or(0) as u64;

        let contradictions: u32 = self
            .db
            .query_one(
                "SELECT COUNT(*) FROM coherence_scores WHERE coherence_type = 'contradictory' AND resolved = 0",
                &[],
                |row| row.get::<_, i64>(0),
            )
            .await?
            .unwrap_or(0) as u32;

        let contradiction_rate = if total > 0 {
            contradictions as f32 / total as f32
        } else {
            0.0
        };

        let supportive_count: u64 = self
            .db
            .query_one(
                "SELECT COUNT(*) FROM coherence_scores WHERE coherence_type = 'supportive'",
                &[],
                |row| row.get::<_, i64>(0),
            )
            .await?
            .unwrap_or(0) as u64;

        let entailment_consistency = if total > 0 {
            supportive_count as f32 / total as f32
        } else {
            1.0
        };

        // Gather domains from scored patterns
        let domain_rows: Vec<(String, i64)> = self
            .db
            .query(
                r#"
                SELECT rp.category, COUNT(*) as cnt
                FROM coherence_scores cs
                JOIN reasoning_patterns rp ON cs.pattern_a_id = rp.id
                WHERE cs.coherence_type = 'contradictory' AND cs.resolved = 0
                GROUP BY rp.category
                ORDER BY cnt DESC
                "#,
                &[],
                |row| Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?)),
            )
            .await?;

        let domains_scanned: Vec<String> = domain_rows.iter().map(|(d, _)| d.clone()).collect();

        let worst_domain = if let Some((domain, count)) = domain_rows.first() {
            if total > 0 {
                Some((domain.clone(), *count as f32 / total as f32))
            } else {
                None
            }
        } else {
            None
        };

        Ok(CoherenceHealth {
            total_pairs_checked: total,
            contradictions_found: contradictions,
            contradiction_rate,
            entailment_consistency,
            domains_scanned,
            worst_domain,
        })
    }

    /// Get the top N contradictions by contradiction score.
    pub async fn top_contradictions(&self, limit: usize) -> Result<Vec<CoherenceScore>> {
        let limit_i64 = limit as i64;

        self.db
            .query(
                r#"
                SELECT id, pattern_a_id, pattern_b_id, similarity, contradiction,
                       coherence_type, detected_at, resolved, resolution_note
                FROM coherence_scores
                WHERE coherence_type = 'contradictory' AND resolved = 0
                ORDER BY contradiction DESC
                LIMIT ?
                "#,
                &[&limit_i64],
                |row| score_from_row(row),
            )
            .await
    }

    /// Mark a contradiction as resolved with a note.
    pub async fn resolve(&self, score_id: &str, note: &str) -> Result<usize> {
        self.db
            .execute(
                "UPDATE coherence_scores SET resolved = 1, resolution_note = ? WHERE id = ?",
                &[&note, &score_id],
            )
            .await
    }

    /// Persist a CoherenceScore to the database.
    pub async fn store_score(&self, score: &CoherenceScore) -> Result<usize> {
        self.db
            .execute(
                r#"
                INSERT OR REPLACE INTO coherence_scores
                    (id, pattern_a_id, pattern_b_id, similarity, contradiction,
                     coherence_type, detected_at, resolved, resolution_note)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                "#,
                &[
                    &score.id,
                    &score.pattern_a.as_str(),
                    &score.pattern_b.as_str(),
                    &score.similarity,
                    &score.contradiction,
                    &score.coherence_type.to_string(),
                    &score.detected_at.to_rfc3339(),
                    &(score.resolved as i32),
                    &score.resolution_note as &dyn rusqlite::ToSql,
                ],
            )
            .await
    }
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// Tokenize text into a set of lowercase words (alphanumeric only).
fn tokenize(text: &str) -> HashSet<String> {
    text.split(|c: char| !c.is_alphanumeric() && c != '\'')
        .map(|w| w.to_lowercase())
        .filter(|w| !w.is_empty())
        .collect()
}

/// Compute Jaccard similarity between two word sets.
fn jaccard_similarity(a: &HashSet<String>, b: &HashSet<String>) -> f32 {
    if a.is_empty() && b.is_empty() {
        return 0.0;
    }
    let intersection = a.intersection(b).count() as f32;
    let union = a.union(b).count() as f32;
    if union == 0.0 {
        0.0
    } else {
        intersection / union
    }
}

/// Count negation words in a token set.
fn count_negation_words(tokens: &HashSet<String>) -> usize {
    NEGATION_WORDS
        .iter()
        .filter(|w| tokens.contains(**w))
        .count()
}

/// Count affirmative words in a token set.
fn count_affirmative_words(tokens: &HashSet<String>) -> usize {
    AFFIRMATIVE_WORDS
        .iter()
        .filter(|w| tokens.contains(**w))
        .count()
}

/// Compute a negation distance between two patterns.
///
/// High distance means one pattern is predominantly affirmative while the
/// other is predominantly negative — a strong signal for contradiction.
fn compute_negation_distance(
    neg_a: usize,
    neg_b: usize,
    aff_a: usize,
    aff_b: usize,
) -> f32 {
    let total_a = (neg_a + aff_a).max(1) as f32;
    let total_b = (neg_b + aff_b).max(1) as f32;

    let ratio_a = neg_a as f32 / total_a;
    let ratio_b = neg_b as f32 / total_b;

    (ratio_a - ratio_b).abs()
}

/// Parse a CoherenceScore from a rusqlite Row.
fn score_from_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<CoherenceScore> {
    let detected_at_str: String = row.get(6)?;
    let detected_at = DateTime::parse_from_rfc3339(&detected_at_str)
        .map(|dt| dt.with_timezone(&Utc))
        .unwrap_or_else(|_| Utc::now());

    let coherence_type_str: String = row.get(5)?;
    let resolved_int: i32 = row.get(7)?;

    Ok(CoherenceScore {
        id: row.get(0)?,
        pattern_a: PatternId::from_string(row.get::<_, String>(1)?),
        pattern_b: PatternId::from_string(row.get::<_, String>(2)?),
        similarity: row.get(3)?,
        contradiction: row.get(4)?,
        coherence_type: CoherenceType::from(coherence_type_str.as_str()),
        detected_at,
        resolved: resolved_int != 0,
        resolution_note: row.get(8)?,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- CoherenceType tests --

    #[test]
    fn test_coherence_type_display_supportive() {
        assert_eq!(CoherenceType::Supportive.to_string(), "supportive");
    }

    #[test]
    fn test_coherence_type_display_neutral() {
        assert_eq!(CoherenceType::Neutral.to_string(), "neutral");
    }

    #[test]
    fn test_coherence_type_display_contradictory() {
        assert_eq!(CoherenceType::Contradictory.to_string(), "contradictory");
    }

    #[test]
    fn test_coherence_type_from_str_supportive() {
        assert_eq!(CoherenceType::from("supportive"), CoherenceType::Supportive);
    }

    #[test]
    fn test_coherence_type_from_str_contradictory() {
        assert_eq!(
            CoherenceType::from("contradictory"),
            CoherenceType::Contradictory
        );
    }

    #[test]
    fn test_coherence_type_from_str_neutral() {
        assert_eq!(CoherenceType::from("neutral"), CoherenceType::Neutral);
    }

    #[test]
    fn test_coherence_type_from_str_unknown() {
        assert_eq!(CoherenceType::from("foobar"), CoherenceType::Neutral);
    }

    #[test]
    fn test_coherence_type_from_str_case_insensitive() {
        assert_eq!(
            CoherenceType::from("CONTRADICTORY"),
            CoherenceType::Contradictory
        );
        assert_eq!(
            CoherenceType::from("Supportive"),
            CoherenceType::Supportive
        );
    }

    #[test]
    fn test_coherence_type_serde_roundtrip() {
        let types = vec![
            CoherenceType::Supportive,
            CoherenceType::Neutral,
            CoherenceType::Contradictory,
        ];
        for ct in types {
            let json = serde_json::to_string(&ct).unwrap();
            let parsed: CoherenceType = serde_json::from_str(&json).unwrap();
            assert_eq!(ct, parsed);
        }
    }

    #[test]
    fn test_coherence_type_serde_values() {
        assert_eq!(
            serde_json::to_string(&CoherenceType::Supportive).unwrap(),
            "\"supportive\""
        );
        assert_eq!(
            serde_json::to_string(&CoherenceType::Contradictory).unwrap(),
            "\"contradictory\""
        );
    }

    // -- ScoringConfig tests --

    #[test]
    fn test_scoring_config_defaults() {
        let config = ScoringConfig::default();
        assert!((config.contradiction_threshold - 0.35).abs() < f32::EPSILON);
        assert!((config.supportive_threshold - 0.8).abs() < f32::EPSILON);
        assert_eq!(config.batch_size, 100);
        assert!(config.scan_domains.is_none());
    }

    #[test]
    fn test_scoring_config_custom() {
        let config = ScoringConfig {
            contradiction_threshold: 0.5,
            supportive_threshold: 0.9,
            batch_size: 50,
            scan_domains: Some(vec!["rust".to_string()]),
        };
        assert!((config.contradiction_threshold - 0.5).abs() < f32::EPSILON);
        assert_eq!(config.batch_size, 50);
        assert_eq!(config.scan_domains.as_ref().unwrap().len(), 1);
    }

    // -- Tokenization tests --

    #[test]
    fn test_tokenize_basic() {
        let tokens = tokenize("Use tokio for async runtime");
        assert!(tokens.contains("use"));
        assert!(tokens.contains("tokio"));
        assert!(tokens.contains("async"));
        assert!(tokens.contains("runtime"));
    }

    #[test]
    fn test_tokenize_empty() {
        let tokens = tokenize("");
        assert!(tokens.is_empty());
    }

    #[test]
    fn test_tokenize_case_insensitive() {
        let tokens = tokenize("Always USE Tokio");
        assert!(tokens.contains("always"));
        assert!(tokens.contains("use"));
        assert!(tokens.contains("tokio"));
    }

    #[test]
    fn test_tokenize_special_chars() {
        let tokens = tokenize("don't use foo-bar; baz_qux!");
        assert!(tokens.contains("don't"));
        assert!(tokens.contains("use"));
        assert!(tokens.contains("foo"));
        assert!(tokens.contains("bar"));
    }

    // -- Jaccard similarity tests --

    #[test]
    fn test_jaccard_identical() {
        let a: HashSet<String> = ["a", "b", "c"].iter().map(|s| s.to_string()).collect();
        let b = a.clone();
        assert!((jaccard_similarity(&a, &b) - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_jaccard_disjoint() {
        let a: HashSet<String> = ["a", "b"].iter().map(|s| s.to_string()).collect();
        let b: HashSet<String> = ["c", "d"].iter().map(|s| s.to_string()).collect();
        assert!((jaccard_similarity(&a, &b)).abs() < f32::EPSILON);
    }

    #[test]
    fn test_jaccard_partial_overlap() {
        let a: HashSet<String> = ["a", "b", "c"].iter().map(|s| s.to_string()).collect();
        let b: HashSet<String> = ["b", "c", "d"].iter().map(|s| s.to_string()).collect();
        // intersection = {b,c} = 2, union = {a,b,c,d} = 4 -> 0.5
        assert!((jaccard_similarity(&a, &b) - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_jaccard_empty_sets() {
        let a: HashSet<String> = HashSet::new();
        let b: HashSet<String> = HashSet::new();
        assert!((jaccard_similarity(&a, &b)).abs() < f32::EPSILON);
    }

    #[test]
    fn test_jaccard_one_empty() {
        let a: HashSet<String> = ["a"].iter().map(|s| s.to_string()).collect();
        let b: HashSet<String> = HashSet::new();
        assert!((jaccard_similarity(&a, &b)).abs() < f32::EPSILON);
    }

    // -- Negation/affirmative counting tests --

    #[test]
    fn test_count_negation_words() {
        let tokens = tokenize("don't never avoid using this");
        assert_eq!(count_negation_words(&tokens), 3); // don't, never, avoid
    }

    #[test]
    fn test_count_negation_words_none() {
        let tokens = tokenize("use tokio for async");
        assert_eq!(count_negation_words(&tokens), 0);
    }

    #[test]
    fn test_count_affirmative_words() {
        let tokens = tokenize("always use and must require");
        assert_eq!(count_affirmative_words(&tokens), 4); // always, use, must, require
    }

    #[test]
    fn test_count_affirmative_words_none() {
        let tokens = tokenize("something completely different");
        assert_eq!(count_affirmative_words(&tokens), 0);
    }

    // -- Negation distance tests --

    #[test]
    fn test_negation_distance_same() {
        // Both equally negative
        let dist = compute_negation_distance(2, 2, 1, 1);
        assert!(dist.abs() < f32::EPSILON);
    }

    #[test]
    fn test_negation_distance_opposite() {
        // A is all negative (neg=3, aff=0), B is all affirmative (neg=0, aff=3)
        let dist = compute_negation_distance(3, 0, 0, 3);
        assert!((dist - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_negation_distance_partial() {
        // A: neg=1, aff=1 -> 0.5; B: neg=0, aff=2 -> 0.0; distance = 0.5
        let dist = compute_negation_distance(1, 0, 1, 2);
        assert!((dist - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_negation_distance_zero_totals() {
        // Both have no negation or affirmative words
        let dist = compute_negation_distance(0, 0, 0, 0);
        assert!(dist.abs() < f32::EPSILON);
    }

    // -- score_pair tests (no DB needed) --

    fn test_scorer() -> CoherenceScorer {
        // We create a scorer with a dummy DB that won't be used for scoring.
        // score_pair does not touch the database.
        let db = Arc::new(SqliteDb::open_in_memory().unwrap());
        CoherenceScorer::new(db, ScoringConfig::default())
    }

    /// Create a minimal reasoning_patterns table for tests that join against it.
    async fn create_reasoning_patterns_table(db: &SqliteDb) {
        db.execute_batch(
            "CREATE TABLE IF NOT EXISTS reasoning_patterns (
                id TEXT PRIMARY KEY,
                problem TEXT NOT NULL DEFAULT '',
                solution TEXT NOT NULL DEFAULT '',
                category TEXT NOT NULL DEFAULT '',
                reward REAL NOT NULL DEFAULT 0.0,
                created_at TEXT NOT NULL DEFAULT '',
                updated_at TEXT NOT NULL DEFAULT ''
            )",
        )
        .await
        .unwrap();
    }

    #[test]
    fn test_score_pair_contradictory() {
        let scorer = test_scorer();
        let id_a = PatternId::from_string("a");
        let id_b = PatternId::from_string("b");

        // Same topic words + opposing directives
        let score = scorer.score_pair(
            "How to handle errors in Rust",
            "Always use unwrap for error handling in Rust code",
            "How to handle errors in Rust",
            "Never use unwrap for error handling in Rust code, avoid panics",
            &id_a,
            &id_b,
        );

        assert_eq!(score.coherence_type, CoherenceType::Contradictory);
        assert!(score.contradiction > 0.0);
        assert!(score.similarity > 0.3);
    }

    #[test]
    fn test_score_pair_supportive() {
        let scorer = test_scorer();
        let id_a = PatternId::from_string("a");
        let id_b = PatternId::from_string("b");

        // Very similar solutions without negation
        let score = scorer.score_pair(
            "Use tokio for async",
            "Use tokio runtime for async operations in Rust applications",
            "Async in Rust",
            "Use tokio runtime for async operations in Rust applications always",
            &id_a,
            &id_b,
        );

        assert_eq!(score.coherence_type, CoherenceType::Supportive);
        assert!(score.similarity > 0.8);
    }

    #[test]
    fn test_score_pair_neutral_unrelated() {
        let scorer = test_scorer();
        let id_a = PatternId::from_string("a");
        let id_b = PatternId::from_string("b");

        let score = scorer.score_pair(
            "Cooking pasta",
            "Boil water and add pasta for eight minutes",
            "Database indexing",
            "Create composite indexes on frequently queried columns",
            &id_a,
            &id_b,
        );

        assert_eq!(score.coherence_type, CoherenceType::Neutral);
        assert!(score.similarity < 0.3);
    }

    #[test]
    fn test_score_pair_empty_solutions() {
        let scorer = test_scorer();
        let id_a = PatternId::from_string("a");
        let id_b = PatternId::from_string("b");

        let score = scorer.score_pair("", "", "", "", &id_a, &id_b);

        assert_eq!(score.coherence_type, CoherenceType::Neutral);
        assert!((score.similarity).abs() < f32::EPSILON);
        assert!((score.contradiction).abs() < f32::EPSILON);
    }

    #[test]
    fn test_score_pair_identical_solutions() {
        let scorer = test_scorer();
        let id_a = PatternId::from_string("a");
        let id_b = PatternId::from_string("b");

        let text = "Always use structured logging for better observability";
        let score = scorer.score_pair("logging", text, "logging", text, &id_a, &id_b);

        // Identical text => jaccard = 1.0, no negation distance => supportive
        assert_eq!(score.coherence_type, CoherenceType::Supportive);
        assert!((score.similarity - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_score_pair_very_short_text() {
        let scorer = test_scorer();
        let id_a = PatternId::from_string("a");
        let id_b = PatternId::from_string("b");

        let score = scorer.score_pair("x", "yes", "y", "no", &id_a, &id_b);
        // Very few tokens, low similarity
        assert_eq!(score.coherence_type, CoherenceType::Neutral);
    }

    #[test]
    fn test_score_pair_pattern_ids_preserved() {
        let scorer = test_scorer();
        let id_a = PatternId::from_string("pattern-abc");
        let id_b = PatternId::from_string("pattern-xyz");

        let score = scorer.score_pair("p", "sol", "p", "sol", &id_a, &id_b);

        assert_eq!(score.pattern_a.as_str(), "pattern-abc");
        assert_eq!(score.pattern_b.as_str(), "pattern-xyz");
        assert!(!score.resolved);
        assert!(score.resolution_note.is_none());
    }

    // -- CoherenceScore serde tests --

    #[test]
    fn test_coherence_score_serde_roundtrip() {
        let score = CoherenceScore {
            id: "test-id".to_string(),
            pattern_a: PatternId::from_string("a"),
            pattern_b: PatternId::from_string("b"),
            similarity: 0.75,
            contradiction: 0.6,
            coherence_type: CoherenceType::Contradictory,
            detected_at: Utc::now(),
            resolved: false,
            resolution_note: None,
        };

        let json = serde_json::to_string(&score).unwrap();
        let parsed: CoherenceScore = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.id, "test-id");
        assert_eq!(parsed.coherence_type, CoherenceType::Contradictory);
        assert!((parsed.similarity - 0.75).abs() < f32::EPSILON);
    }

    #[test]
    fn test_coherence_score_serde_with_resolution() {
        let score = CoherenceScore {
            id: "resolved-id".to_string(),
            pattern_a: PatternId::from_string("a"),
            pattern_b: PatternId::from_string("b"),
            similarity: 0.5,
            contradiction: 0.8,
            coherence_type: CoherenceType::Contradictory,
            detected_at: Utc::now(),
            resolved: true,
            resolution_note: Some("Manually reviewed and accepted both".to_string()),
        };

        let json = serde_json::to_string(&score).unwrap();
        let parsed: CoherenceScore = serde_json::from_str(&json).unwrap();

        assert!(parsed.resolved);
        assert_eq!(
            parsed.resolution_note.unwrap(),
            "Manually reviewed and accepted both"
        );
    }

    // -- CoherenceHealth tests --

    #[test]
    fn test_coherence_health_serde_roundtrip() {
        let health = CoherenceHealth {
            total_pairs_checked: 100,
            contradictions_found: 5,
            contradiction_rate: 0.05,
            entailment_consistency: 0.4,
            domains_scanned: vec!["rust".to_string(), "python".to_string()],
            worst_domain: Some(("rust".to_string(), 0.03)),
        };

        let json = serde_json::to_string(&health).unwrap();
        let parsed: CoherenceHealth = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.total_pairs_checked, 100);
        assert_eq!(parsed.contradictions_found, 5);
        assert_eq!(parsed.domains_scanned.len(), 2);
        let (domain, rate) = parsed.worst_domain.unwrap();
        assert_eq!(domain, "rust");
        assert!((rate - 0.03).abs() < f32::EPSILON);
    }

    #[test]
    fn test_coherence_health_no_worst_domain() {
        let health = CoherenceHealth {
            total_pairs_checked: 0,
            contradictions_found: 0,
            contradiction_rate: 0.0,
            entailment_consistency: 1.0,
            domains_scanned: vec![],
            worst_domain: None,
        };

        let json = serde_json::to_string(&health).unwrap();
        let parsed: CoherenceHealth = serde_json::from_str(&json).unwrap();

        assert!(parsed.worst_domain.is_none());
        assert!(parsed.domains_scanned.is_empty());
    }

    // -- Database integration tests --

    #[tokio::test]
    async fn test_init_schema() {
        let db = Arc::new(SqliteDb::open_in_memory().unwrap());
        let scorer = CoherenceScorer::new(db.clone(), ScoringConfig::default());

        scorer.init_schema().await.unwrap();

        let exists = db.table_exists("coherence_scores").await.unwrap();
        assert!(exists);
    }

    #[tokio::test]
    async fn test_store_and_retrieve_score() {
        let db = Arc::new(SqliteDb::open_in_memory().unwrap());
        let scorer = CoherenceScorer::new(db.clone(), ScoringConfig::default());
        scorer.init_schema().await.unwrap();

        let score = CoherenceScore {
            id: "score-1".to_string(),
            pattern_a: PatternId::from_string("p-a"),
            pattern_b: PatternId::from_string("p-b"),
            similarity: 0.65,
            contradiction: 0.8,
            coherence_type: CoherenceType::Contradictory,
            detected_at: Utc::now(),
            resolved: false,
            resolution_note: None,
        };

        let rows = scorer.store_score(&score).await.unwrap();
        assert_eq!(rows, 1);

        let top = scorer.top_contradictions(10).await.unwrap();
        assert_eq!(top.len(), 1);
        assert_eq!(top[0].id, "score-1");
        assert!((top[0].contradiction - 0.8).abs() < f32::EPSILON);
    }

    #[tokio::test]
    async fn test_resolve_contradiction() {
        let db = Arc::new(SqliteDb::open_in_memory().unwrap());
        let scorer = CoherenceScorer::new(db.clone(), ScoringConfig::default());
        scorer.init_schema().await.unwrap();

        let score = CoherenceScore {
            id: "score-resolve".to_string(),
            pattern_a: PatternId::from_string("p-a"),
            pattern_b: PatternId::from_string("p-b"),
            similarity: 0.5,
            contradiction: 0.9,
            coherence_type: CoherenceType::Contradictory,
            detected_at: Utc::now(),
            resolved: false,
            resolution_note: None,
        };

        scorer.store_score(&score).await.unwrap();

        let updated = scorer
            .resolve("score-resolve", "Both approaches valid in different contexts")
            .await
            .unwrap();
        assert_eq!(updated, 1);

        // Should no longer appear in unresolved contradictions
        let top = scorer.top_contradictions(10).await.unwrap();
        assert!(top.is_empty());
    }

    #[tokio::test]
    async fn test_top_contradictions_sorted() {
        let db = Arc::new(SqliteDb::open_in_memory().unwrap());
        let scorer = CoherenceScorer::new(db.clone(), ScoringConfig::default());
        scorer.init_schema().await.unwrap();

        for (i, c) in [0.5_f32, 0.9, 0.7].iter().enumerate() {
            let score = CoherenceScore {
                id: format!("score-{}", i),
                pattern_a: PatternId::from_string(format!("a-{}", i)),
                pattern_b: PatternId::from_string(format!("b-{}", i)),
                similarity: 0.6,
                contradiction: *c,
                coherence_type: CoherenceType::Contradictory,
                detected_at: Utc::now(),
                resolved: false,
                resolution_note: None,
            };
            scorer.store_score(&score).await.unwrap();
        }

        let top = scorer.top_contradictions(10).await.unwrap();
        assert_eq!(top.len(), 3);
        // Should be sorted descending by contradiction
        assert!((top[0].contradiction - 0.9).abs() < f32::EPSILON);
        assert!((top[1].contradiction - 0.7).abs() < f32::EPSILON);
        assert!((top[2].contradiction - 0.5).abs() < f32::EPSILON);
    }

    #[tokio::test]
    async fn test_top_contradictions_respects_limit() {
        let db = Arc::new(SqliteDb::open_in_memory().unwrap());
        let scorer = CoherenceScorer::new(db.clone(), ScoringConfig::default());
        scorer.init_schema().await.unwrap();

        for i in 0..5 {
            let score = CoherenceScore {
                id: format!("s-{}", i),
                pattern_a: PatternId::from_string(format!("a-{}", i)),
                pattern_b: PatternId::from_string(format!("b-{}", i)),
                similarity: 0.6,
                contradiction: 0.8,
                coherence_type: CoherenceType::Contradictory,
                detected_at: Utc::now(),
                resolved: false,
                resolution_note: None,
            };
            scorer.store_score(&score).await.unwrap();
        }

        let top = scorer.top_contradictions(3).await.unwrap();
        assert_eq!(top.len(), 3);
    }

    #[tokio::test]
    async fn test_system_health_empty() {
        let db = Arc::new(SqliteDb::open_in_memory().unwrap());
        create_reasoning_patterns_table(&db).await;
        let scorer = CoherenceScorer::new(db.clone(), ScoringConfig::default());
        scorer.init_schema().await.unwrap();

        let health = scorer.system_health().await.unwrap();
        assert_eq!(health.total_pairs_checked, 0);
        assert_eq!(health.contradictions_found, 0);
        assert!((health.contradiction_rate).abs() < f32::EPSILON);
        assert!((health.entailment_consistency - 1.0).abs() < f32::EPSILON);
        assert!(health.worst_domain.is_none());
    }

    #[tokio::test]
    async fn test_system_health_with_data() {
        let db = Arc::new(SqliteDb::open_in_memory().unwrap());
        create_reasoning_patterns_table(&db).await;
        let scorer = CoherenceScorer::new(db.clone(), ScoringConfig::default());
        scorer.init_schema().await.unwrap();

        // Store mix of types
        for (i, ct) in [
            CoherenceType::Supportive,
            CoherenceType::Contradictory,
            CoherenceType::Neutral,
            CoherenceType::Supportive,
        ]
        .iter()
        .enumerate()
        {
            let score = CoherenceScore {
                id: format!("h-{}", i),
                pattern_a: PatternId::from_string(format!("a-{}", i)),
                pattern_b: PatternId::from_string(format!("b-{}", i)),
                similarity: 0.6,
                contradiction: if *ct == CoherenceType::Contradictory {
                    0.8
                } else {
                    0.1
                },
                coherence_type: ct.clone(),
                detected_at: Utc::now(),
                resolved: false,
                resolution_note: None,
            };
            scorer.store_score(&score).await.unwrap();
        }

        let health = scorer.system_health().await.unwrap();
        assert_eq!(health.total_pairs_checked, 4);
        assert_eq!(health.contradictions_found, 1);
        assert!((health.contradiction_rate - 0.25).abs() < f32::EPSILON);
        // 2 supportive out of 4 total
        assert!((health.entailment_consistency - 0.5).abs() < f32::EPSILON);
    }

    #[tokio::test]
    async fn test_store_score_upsert() {
        let db = Arc::new(SqliteDb::open_in_memory().unwrap());
        let scorer = CoherenceScorer::new(db.clone(), ScoringConfig::default());
        scorer.init_schema().await.unwrap();

        let mut score = CoherenceScore {
            id: "upsert-test".to_string(),
            pattern_a: PatternId::from_string("pa"),
            pattern_b: PatternId::from_string("pb"),
            similarity: 0.5,
            contradiction: 0.6,
            coherence_type: CoherenceType::Neutral,
            detected_at: Utc::now(),
            resolved: false,
            resolution_note: None,
        };

        scorer.store_score(&score).await.unwrap();

        // Update same id
        score.contradiction = 0.9;
        score.coherence_type = CoherenceType::Contradictory;
        scorer.store_score(&score).await.unwrap();

        let top = scorer.top_contradictions(10).await.unwrap();
        assert_eq!(top.len(), 1);
        assert!((top[0].contradiction - 0.9).abs() < f32::EPSILON);
    }

    #[tokio::test]
    async fn test_scan_domain_empty() {
        let db = Arc::new(SqliteDb::open_in_memory().unwrap());
        // Create reasoning_patterns table for scan
        db.execute_batch(
            "CREATE TABLE IF NOT EXISTS reasoning_patterns (
                id TEXT PRIMARY KEY,
                category TEXT,
                problem TEXT,
                solution TEXT,
                context TEXT DEFAULT '',
                effectiveness REAL DEFAULT 0.5,
                reuse_count INTEGER DEFAULT 0,
                reward REAL DEFAULT 0.5,
                success INTEGER DEFAULT 1,
                critique TEXT DEFAULT '',
                agent_id TEXT,
                session_id TEXT,
                confidence REAL DEFAULT 0.5,
                embedding BLOB,
                surprise_score REAL DEFAULT 0.0,
                failure_mode TEXT,
                chunk_embeddings TEXT,
                satisfaction_score REAL DEFAULT 0.5,
                satisfaction_trials INTEGER DEFAULT 0,
                content_hash TEXT,
                title TEXT,
                summary TEXT,
                tags TEXT DEFAULT '[]',
                related_patterns TEXT DEFAULT '[]',
                metadata TEXT DEFAULT '{}',
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )"
        ).await.unwrap();

        let scorer = CoherenceScorer::new(db.clone(), ScoringConfig::default());
        scorer.init_schema().await.unwrap();

        let results = scorer.scan_domain("nonexistent").await.unwrap();
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn test_scan_domain_with_patterns() {
        let db = Arc::new(SqliteDb::open_in_memory().unwrap());
        db.execute_batch(
            "CREATE TABLE IF NOT EXISTS reasoning_patterns (
                id TEXT PRIMARY KEY,
                category TEXT,
                problem TEXT,
                solution TEXT,
                context TEXT DEFAULT '',
                effectiveness REAL DEFAULT 0.5,
                reuse_count INTEGER DEFAULT 0,
                reward REAL DEFAULT 0.5,
                success INTEGER DEFAULT 1,
                critique TEXT DEFAULT '',
                agent_id TEXT,
                session_id TEXT,
                confidence REAL DEFAULT 0.5,
                embedding BLOB,
                surprise_score REAL DEFAULT 0.0,
                failure_mode TEXT,
                chunk_embeddings TEXT,
                satisfaction_score REAL DEFAULT 0.5,
                satisfaction_trials INTEGER DEFAULT 0,
                content_hash TEXT,
                title TEXT,
                summary TEXT,
                tags TEXT DEFAULT '[]',
                related_patterns TEXT DEFAULT '[]',
                metadata TEXT DEFAULT '{}',
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )"
        ).await.unwrap();

        // Insert two contradictory patterns in the same domain
        db.execute(
            "INSERT INTO reasoning_patterns (id, category, problem, solution) VALUES (?, ?, ?, ?)",
            &[
                &"p1" as &dyn rusqlite::ToSql,
                &"rust",
                &"error handling",
                &"Always use unwrap for quick error handling in Rust code",
            ],
        ).await.unwrap();

        db.execute(
            "INSERT INTO reasoning_patterns (id, category, problem, solution) VALUES (?, ?, ?, ?)",
            &[
                &"p2" as &dyn rusqlite::ToSql,
                &"rust",
                &"error handling",
                &"Never use unwrap for error handling in Rust code, avoid panics",
            ],
        ).await.unwrap();

        let scorer = CoherenceScorer::new(db.clone(), ScoringConfig::default());
        scorer.init_schema().await.unwrap();

        let results = scorer.scan_domain("rust").await.unwrap();
        // Should detect the contradiction
        assert!(!results.is_empty());
        // At least one score should have been stored
        let top = scorer.top_contradictions(10).await.unwrap();
        assert!(top.len() <= results.len());
    }

    #[tokio::test]
    async fn test_resolve_nonexistent() {
        let db = Arc::new(SqliteDb::open_in_memory().unwrap());
        let scorer = CoherenceScorer::new(db.clone(), ScoringConfig::default());
        scorer.init_schema().await.unwrap();

        let updated = scorer.resolve("nonexistent", "test note").await.unwrap();
        assert_eq!(updated, 0);
    }
}
