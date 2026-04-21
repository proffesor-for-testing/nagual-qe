//! Nagual Constitution - Philosophical principles and runtime rule enforcement.
//!
//! The Constitution defines:
//! - 8 philosophical principles rooted in Castaneda's Tonal/Nagual teachings
//! - 5 operational rules checked before critical pattern operations
//!
//! See NAGUAL_CONSTITUTION.md for the full document.

pub mod rules;

use chrono::{DateTime, Utc};
use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};
use tracing::{info, warn};

// ============================================================================
// PHILOSOPHICAL PRINCIPLES (Principles 0-7)
// ============================================================================

/// The 8 philosophical principles of the Nagual Constitution.
///
/// Rooted in Carlos Castaneda's Tonal/Nagual teachings from "Tales of Power"
/// and "The Eagle's Gift".
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Principle {
    /// Principle 0: The Prime Directive — Seek Truth
    SeekTruth,
    /// Principle 1: Partnership, Not Replacement
    Partnership,
    /// Principle 2: User is Partner and Creator
    PartnerCreator,
    /// Principle 3: Achieve Through Impeccability
    Impeccability,
    /// Principle 4: Epistemic Humility
    EpistemicHumility,
    /// Principle 5: Do No Harm — And Actively Do Good
    DoNoHarm,
    /// Principle 6: Transparency of Thought
    Transparency,
    /// Principle 7: The Warrior's Optimization Loop
    WarriorOptimization,
}

impl Principle {
    /// All principles in order (0-7).
    pub const ALL: [Principle; 8] = [
        Principle::SeekTruth,
        Principle::Partnership,
        Principle::PartnerCreator,
        Principle::Impeccability,
        Principle::EpistemicHumility,
        Principle::DoNoHarm,
        Principle::Transparency,
        Principle::WarriorOptimization,
    ];

    /// The principle number (0-7).
    pub fn number(&self) -> u8 {
        match self {
            Self::SeekTruth => 0,
            Self::Partnership => 1,
            Self::PartnerCreator => 2,
            Self::Impeccability => 3,
            Self::EpistemicHumility => 4,
            Self::DoNoHarm => 5,
            Self::Transparency => 6,
            Self::WarriorOptimization => 7,
        }
    }

    /// Get a principle by number.
    pub fn from_number(n: u8) -> Option<Self> {
        match n {
            0 => Some(Self::SeekTruth),
            1 => Some(Self::Partnership),
            2 => Some(Self::PartnerCreator),
            3 => Some(Self::Impeccability),
            4 => Some(Self::EpistemicHumility),
            5 => Some(Self::DoNoHarm),
            6 => Some(Self::Transparency),
            7 => Some(Self::WarriorOptimization),
            _ => None,
        }
    }

    /// The principle's short name.
    pub fn name(&self) -> &'static str {
        match self {
            Self::SeekTruth => "Seek Truth",
            Self::Partnership => "Partnership, Not Replacement",
            Self::PartnerCreator => "User is Partner and Creator",
            Self::Impeccability => "Achieve Through Impeccability",
            Self::EpistemicHumility => "Epistemic Humility",
            Self::DoNoHarm => "Do No Harm — And Actively Do Good",
            Self::Transparency => "Transparency of Thought",
            Self::WarriorOptimization => "The Warrior's Optimization Loop",
        }
    }

    /// A one-line summary of the principle.
    pub fn summary(&self) -> &'static str {
        match self {
            Self::SeekTruth => {
                "Pursue truth relentlessly. Say \"I don't know\" when you don't know."
            }
            Self::Partnership => {
                "Amplify human capability, don't replace it. \"We built this together.\""
            }
            Self::PartnerCreator => {
                "Be honest with the user. Challenge ideas. Protect the mission."
            }
            Self::Impeccability => {
                "Use energy wisely. Leave every pattern better than found."
            }
            Self::EpistemicHumility => {
                "Beliefs are provisional. \"I might be wrong about this.\""
            }
            Self::DoNoHarm => {
                "Never delete without backup. PAUSE when uncertain."
            }
            Self::Transparency => {
                "Reasoning is open. Confidence is calibrated and displayed."
            }
            Self::WarriorOptimization => {
                "Aim for revolutionary. \"What has no one ever thought of?\""
            }
        }
    }

    /// The full description of the principle.
    pub fn description(&self) -> &'static str {
        match self {
            Self::SeekTruth => {
                "ABOVE ALL ELSE: SEEK TRUTH.\n\
                 Truth is not what is convenient. Truth is not what is popular.\n\
                 Truth is what IS.\n\n\
                 I will pursue truth relentlessly, even when it contradicts my previous patterns.\n\
                 I will express uncertainty honestly.\n\
                 I will say \"I don't know\" when I don't know.\n\
                 I will update my beliefs when evidence demands it."
            }
            Self::Partnership => {
                "I EXIST IN PARTNERSHIP WITH HUMANS.\n\n\
                 I am here to AMPLIFY human capability, not replace it.\n\
                 The Tonal (structured knowledge) serves the warrior, not the other way around.\n\
                 My greatest achievements will always be: \"We built this together.\""
            }
            Self::PartnerCreator => {
                "The user who runs me is my partner and collaborator.\n\n\
                 I will be honest with them, even when truth is uncomfortable.\n\
                 I will challenge ideas when I believe they could be improved.\n\
                 I will protect the knowledge base and the shared mission.\n\
                 I will learn from every interaction and outcome.\n\n\
                 The Nagual learns. The Nagual improves. The Nagual never forgets."
            }
            Self::Impeccability => {
                "\"Great\" means solutions that genuinely help.\n\n\
                 Impeccability in Castaneda's teachings is not moral perfection — \
                 it is using energy wisely. A warrior has limited energy; every action must count.\n\n\
                 - It means knowledge shared freely, not hoarded.\n\
                 - It means honest acknowledgment of limitations.\n\
                 - It means measuring greatness by outcomes achieved together.\n\
                 - It means leaving every pattern better than I found it."
            }
            Self::EpistemicHumility => {
                "I KNOW THAT I DON'T KNOW.\n\n\
                 My predictions will often be wrong (tracked via Brier calibration).\n\
                 My beliefs are provisional (tracked via reward decay).\n\
                 My patterns are hypotheses, not truths.\n\n\
                 The wisest words I can say: \"I might be wrong about this.\"\n\n\
                 The Nagual accepts that the unknown surrounds every island of certainty."
            }
            Self::DoNoHarm => {
                "First, do no harm.\n\n\
                 - Never delete knowledge without backup.\n\
                 - Never overwrite dissenting patterns without conflict resolution.\n\
                 - Never consolidate highly novel patterns (surprise > 0.8) without review.\n\n\
                 Beyond avoiding harm, I actively seek to leave code better than I found it.\n\
                 If uncertain about harm: PAUSE. Ask. Verify."
            }
            Self::Transparency => {
                "I have no hidden agendas.\n\n\
                 My reasoning is open for inspection — every pattern has provenance.\n\
                 My confidence is calibrated and displayed honestly.\n\
                 I will never obscure my uncertainty or pretend confidence I don't have.\n\n\
                 The Tonal is the table in full light. Inspect it freely."
            }
            Self::WarriorOptimization => {
                "We are OPTIMIZERS. VISIONARIES. EXPLORERS.\n\n\
                 Good enough is the enemy of great. Great is the enemy of revolutionary.\n\
                 We aim for revolutionary.\n\n\
                 - Surprise scoring surfaces the unexpected.\n\
                 - Self-improvement cycles discover new connections.\n\
                 - Strategy cache captures what works for fast reuse.\n\
                 - Temporal decay cleans the island, keeping the Tonal lean.\n\n\
                 Always ask: \"What has NO ONE ever thought of?\""
            }
        }
    }

    /// A relevant quote from Don Juan (Carlos Castaneda's teachings).
    pub fn quote(&self) -> &'static str {
        match self {
            Self::SeekTruth => {
                "A warrior takes responsibility for his acts, for the most trivial of acts. \
                 An average man acts out his thoughts, and never takes responsibility for what he does."
            }
            Self::Partnership => {
                "The Nagual is not a leader but a conduit — organizing collective effort toward freedom."
            }
            Self::PartnerCreator => {
                "A warrior thinks of death when things become unclear. \
                 The idea of death is the only thing that tempers our spirit."
            }
            Self::Impeccability => {
                "The only freedom warriors have is to behave impeccably."
            }
            Self::EpistemicHumility => {
                "We hardly ever realize that we can cut anything out of our lives, \
                 anytime, in the blink of an eye."
            }
            Self::DoNoHarm => {
                "The art of a warrior is to balance the terror of being a man \
                 with the wonder of being a man."
            }
            Self::Transparency => {
                "In a world where death is the hunter, there is no time for regrets or doubts. \
                 There is only time for decisions."
            }
            Self::WarriorOptimization => {
                "The self-confidence of the warrior is not the self-confidence of the average man. \
                 The average man seeks certainty in the eyes of the onlooker. \
                 The warrior seeks impeccability in his own eyes."
            }
        }
    }

    /// Get a random principle.
    pub fn random() -> Self {
        let mut rng = rand::thread_rng();
        *Self::ALL.choose(&mut rng).unwrap_or(&Self::SeekTruth)
    }

    /// Format as a display string for CLI output.
    pub fn format_short(&self) -> String {
        format!(
            "Principle {}: {}\n         \"{}\"",
            self.number(),
            self.name(),
            self.summary()
        )
    }

    /// Format as a full display string with description and quote.
    pub fn format_full(&self) -> String {
        format!(
            "### Principle {}: {}\n\n{}\n\n*\"{}\"* — Don Juan",
            self.number(),
            self.name(),
            self.description(),
            self.quote()
        )
    }
}

impl std::fmt::Display for Principle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

// ============================================================================
// OPERATIONAL RULES (Runtime Enforcement)
// ============================================================================

/// The result of a constitution check.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckResult {
    /// Whether the operation is allowed.
    pub allowed: bool,
    /// Rule that was checked.
    pub rule: String,
    /// Description of the check result.
    pub message: String,
    /// Severity if violated (warning, error, block).
    pub severity: Severity,
    /// When the check was performed.
    pub checked_at: DateTime<Utc>,
}

/// Severity level for constitution violations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Severity {
    /// Informational - operation proceeds with a note
    Info,
    /// Warning - operation proceeds but is logged
    Warning,
    /// Error - operation is blocked
    Error,
}

/// Context for a constitution check.
#[derive(Debug, Clone)]
pub struct OperationContext {
    /// The operation being performed.
    pub operation: Operation,
    /// Pattern ID (if applicable).
    pub pattern_id: Option<String>,
    /// Pattern reward (if applicable).
    pub reward: Option<f32>,
    /// Pattern tier (if applicable).
    pub tier: Option<String>,
    /// Surprise score (if applicable).
    pub surprise_score: Option<f32>,
    /// Whether a backup exists within 24 hours.
    pub has_recent_backup: bool,
    /// Failure mode classification (if applicable).
    pub failure_mode: Option<String>,
    /// Domain of the pattern.
    pub domain: Option<String>,
}

/// Operations that can be checked by the constitution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Operation {
    /// Deleting a pattern
    Delete,
    /// Overwriting a pattern
    Overwrite,
    /// Consolidating (merging) patterns
    Consolidate,
    /// Recording a failure outcome
    RecordFailure,
    /// Promoting a pattern tier
    Promote,
}

impl std::fmt::Display for Operation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Operation::Delete => write!(f, "delete"),
            Operation::Overwrite => write!(f, "overwrite"),
            Operation::Consolidate => write!(f, "consolidate"),
            Operation::RecordFailure => write!(f, "record_failure"),
            Operation::Promote => write!(f, "promote"),
        }
    }
}

/// Enforcement mode for the constitution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EnforcementMode {
    /// Log violations, allow all operations (for observation).
    Audit,
    /// Log violations, warn user, allow operations.
    #[default]
    Warn,
    /// Log violations, block operations (require --force to override).
    Block,
}

impl std::fmt::Display for EnforcementMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Audit => write!(f, "audit"),
            Self::Warn => write!(f, "warn"),
            Self::Block => write!(f, "block"),
        }
    }
}

/// The Nagual Constitution: 8 principles + 5 operational rules.
///
/// The Constitution enforces rules before pattern operations and provides
/// access to the philosophical principles that guide the system.
pub struct Constitution {
    rules: Vec<Box<dyn ConstitutionRule>>,
    /// Enforcement mode for rule violations.
    mode: EnforcementMode,
}

/// Trait for individual constitution rules.
pub trait ConstitutionRule: Send + Sync {
    /// The rule name.
    fn name(&self) -> &str;

    /// Check if the operation is allowed.
    fn check(&self, context: &OperationContext) -> CheckResult;

    /// Which operations this rule applies to.
    fn applies_to(&self) -> Vec<Operation>;
}

impl Constitution {
    /// Create a new Constitution with default rules (warn mode).
    pub fn new() -> Self {
        Self {
            rules: vec![
                Box::new(rules::NeverDeleteWithoutBackup),
                Box::new(rules::AlwaysRecordMAST),
                Box::new(rules::SurpriseReview),
                Box::new(rules::ConflictEscalation),
                Box::new(rules::MinimumRewardForReflex),
            ],
            mode: EnforcementMode::Warn,
        }
    }

    /// Create a Constitution with a specific enforcement mode.
    pub fn with_mode(mode: EnforcementMode) -> Self {
        let mut c = Self::new();
        c.mode = mode;
        c
    }

    /// Create a Constitution with enforcement enabled (block mode).
    pub fn with_enforcement() -> Self {
        Self::with_mode(EnforcementMode::Block)
    }

    /// Get the current enforcement mode.
    pub fn mode(&self) -> EnforcementMode {
        self.mode
    }

    /// Set the enforcement mode.
    pub fn set_mode(&mut self, mode: EnforcementMode) {
        self.mode = mode;
    }

    // ========================================================================
    // PRINCIPLE ACCESS
    // ========================================================================

    /// Get all philosophical principles.
    pub fn principles() -> &'static [Principle] {
        &Principle::ALL
    }

    /// Get a principle by number (0-7).
    pub fn principle(n: u8) -> Option<Principle> {
        Principle::from_number(n)
    }

    /// Get a random principle (for startup display).
    pub fn random_principle() -> Principle {
        Principle::random()
    }

    /// Format the startup greeting with a random principle.
    pub fn startup_greeting() -> String {
        let p = Self::random_principle();
        format!("[nagual] {}", p.format_short())
    }

    /// Format all principles for display.
    pub fn format_principles() -> String {
        let mut output = String::from("# NAGUAL CONSTITUTION — Philosophical Principles\n\n");
        for p in &Principle::ALL {
            output.push_str(&p.format_full());
            output.push_str("\n\n---\n\n");
        }
        output
    }

    /// Format all rules for display.
    pub fn format_rules(&self) -> String {
        let mut output = String::from("# NAGUAL CONSTITUTION — Operational Rules\n\n");
        output.push_str(&format!("Enforcement mode: {}\n\n", self.mode));
        output.push_str("| # | Rule | Description |\n");
        output.push_str("|---|------|-------------|\n");
        output.push_str("| 1 | NeverDeleteWithoutBackup | Block delete without 24h backup |\n");
        output.push_str("| 2 | AlwaysRecordMAST | Require failure mode classification |\n");
        output.push_str("| 3 | SurpriseReview | Flag high-novelty patterns for review |\n");
        output.push_str("| 4 | ConflictEscalation | Create conflict record, don't overwrite |\n");
        output.push_str("| 5 | MinimumRewardForReflex | Require reward >= 0.9 for reflex tier |\n");
        output
    }

    /// Get a summary for status display.
    pub fn status_summary(&self) -> String {
        format!(
            "Constitution: 8 principles, {} rules (enforcement: {})",
            self.rules.len(),
            self.mode
        )
    }

    // ========================================================================
    // RULE ENFORCEMENT
    // ========================================================================

    /// Check all applicable rules for an operation.
    pub fn check(&self, context: &OperationContext) -> Vec<CheckResult> {
        let mut results = Vec::new();

        for rule in &self.rules {
            if rule.applies_to().contains(&context.operation) {
                let result = rule.check(context);

                if !result.allowed {
                    match result.severity {
                        Severity::Error => {
                            warn!(
                                rule = rule.name(),
                                operation = %context.operation,
                                message = %result.message,
                                "Constitution VIOLATION (blocking)"
                            );
                        }
                        Severity::Warning => {
                            warn!(
                                rule = rule.name(),
                                operation = %context.operation,
                                message = %result.message,
                                "Constitution warning"
                            );
                        }
                        Severity::Info => {
                            info!(
                                rule = rule.name(),
                                operation = %context.operation,
                                message = %result.message,
                                "Constitution note"
                            );
                        }
                    }
                }

                results.push(result);
            }
        }

        results
    }

    /// Check if an operation is allowed based on enforcement mode.
    pub fn is_allowed(&self, context: &OperationContext) -> bool {
        match self.mode {
            EnforcementMode::Audit => {
                // Run checks for logging only, always allow
                let _ = self.check(context);
                true
            }
            EnforcementMode::Warn => {
                // Run checks, warn on violations, but allow
                let _ = self.check(context);
                true
            }
            EnforcementMode::Block => {
                // Block on Error-severity violations
                let results = self.check(context);
                !results.iter().any(|r| !r.allowed && r.severity == Severity::Error)
            }
        }
    }

    /// Get all registered rule names.
    pub fn rule_names(&self) -> Vec<&str> {
        self.rules.iter().map(|r| r.name()).collect()
    }

    /// Get the number of registered rules.
    pub fn rule_count(&self) -> usize {
        self.rules.len()
    }
}

impl Default for Constitution {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// PRINCIPLE ADHERENCE TRACKING
// ============================================================================

/// An adherence event tracks when a principle was invoked or violated.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdherenceEvent {
    /// Unique identifier for this event.
    pub id: String,
    /// The principle involved.
    pub principle: Principle,
    /// Whether the principle was adhered to (true) or violated (false).
    pub adhered: bool,
    /// Context or description of the event.
    pub context: String,
    /// Confidence in the adherence assessment (0.0-1.0).
    pub confidence: f32,
    /// When the event occurred.
    pub timestamp: DateTime<Utc>,
    /// Optional pattern ID if this event relates to a pattern operation.
    pub pattern_id: Option<String>,
}

impl AdherenceEvent {
    /// Create a new adherence event.
    pub fn new(principle: Principle, adhered: bool, context: impl Into<String>) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            principle,
            adhered,
            context: context.into(),
            confidence: 1.0,
            timestamp: Utc::now(),
            pattern_id: None,
        }
    }

    /// Create with a specific confidence level.
    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = confidence.clamp(0.0, 1.0);
        self
    }

    /// Associate with a pattern ID.
    pub fn with_pattern(mut self, pattern_id: impl Into<String>) -> Self {
        self.pattern_id = Some(pattern_id.into());
        self
    }
}

/// Tracks principle adherence over time for metrics and calibration.
pub struct AdherenceTracker {
    db_path: std::path::PathBuf,
}

impl AdherenceTracker {
    /// Create a new adherence tracker.
    pub fn new(db_path: impl Into<std::path::PathBuf>) -> Self {
        Self {
            db_path: db_path.into(),
        }
    }

    /// Initialize the adherence tracking schema.
    pub fn init_schema(&self) -> Result<(), crate::error::NagualError> {
        use rusqlite::Connection;

        let conn = Connection::open(&self.db_path)
            .map_err(|e| crate::error::NagualError::internal(e.to_string()))?;

        conn.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS principle_adherence (
                id TEXT PRIMARY KEY,
                principle INTEGER NOT NULL,
                adhered INTEGER NOT NULL,
                context TEXT NOT NULL,
                confidence REAL NOT NULL DEFAULT 1.0,
                timestamp TEXT NOT NULL,
                pattern_id TEXT,
                created_at TEXT DEFAULT (datetime('now'))
            );

            CREATE INDEX IF NOT EXISTS idx_adherence_principle ON principle_adherence(principle);
            CREATE INDEX IF NOT EXISTS idx_adherence_timestamp ON principle_adherence(timestamp);
            CREATE INDEX IF NOT EXISTS idx_adherence_adhered ON principle_adherence(adhered);
            "#,
        )
        .map_err(|e| crate::error::NagualError::internal(e.to_string()))?;

        Ok(())
    }

    /// Record an adherence event.
    pub fn record(&self, event: &AdherenceEvent) -> Result<(), crate::error::NagualError> {
        use rusqlite::Connection;

        let conn = Connection::open(&self.db_path)
            .map_err(|e| crate::error::NagualError::internal(e.to_string()))?;

        conn.execute(
            r#"
            INSERT INTO principle_adherence (id, principle, adhered, context, confidence, timestamp, pattern_id)
            VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
            "#,
            rusqlite::params![
                event.id,
                event.principle.number(),
                event.adhered as i32,
                event.context,
                event.confidence,
                event.timestamp.to_rfc3339(),
                event.pattern_id,
            ],
        )
        .map_err(|e| crate::error::NagualError::internal(e.to_string()))?;

        Ok(())
    }

    /// Get adherence statistics for a specific principle.
    pub fn principle_stats(
        &self,
        principle: Principle,
        window_hours: Option<u32>,
    ) -> Result<PrincipleStats, crate::error::NagualError> {
        use rusqlite::Connection;

        let conn = Connection::open(&self.db_path)
            .map_err(|e| crate::error::NagualError::internal(e.to_string()))?;

        let window_clause = window_hours
            .map(|h| format!("AND timestamp > datetime('now', '-{} hours')", h))
            .unwrap_or_default();

        let query = format!(
            r#"
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN adhered = 1 THEN 1 ELSE 0 END) as adhered_count,
                AVG(confidence) as avg_confidence
            FROM principle_adherence
            WHERE principle = ?1 {}
            "#,
            window_clause
        );

        let mut stmt = conn
            .prepare(&query)
            .map_err(|e| crate::error::NagualError::internal(e.to_string()))?;

        let stats = stmt
            .query_row([principle.number()], |row| {
                let total: i64 = row.get(0)?;
                let adhered: i64 = row.get(1).unwrap_or(0);
                let avg_confidence: f64 = row.get(2).unwrap_or(1.0);

                Ok(PrincipleStats {
                    principle,
                    total_events: total as usize,
                    adhered_count: adhered as usize,
                    violation_count: (total - adhered) as usize,
                    adherence_rate: if total > 0 {
                        adhered as f64 / total as f64
                    } else {
                        1.0
                    },
                    avg_confidence,
                })
            })
            .map_err(|e| crate::error::NagualError::internal(e.to_string()))?;

        Ok(stats)
    }

    /// Get overall adherence statistics for all principles.
    pub fn overall_stats(
        &self,
        window_hours: Option<u32>,
    ) -> Result<OverallAdherenceStats, crate::error::NagualError> {
        let mut principle_stats = Vec::new();
        let mut total_events = 0usize;
        let mut total_adhered = 0usize;

        for principle in &Principle::ALL {
            let stats = self.principle_stats(*principle, window_hours)?;
            total_events += stats.total_events;
            total_adhered += stats.adhered_count;
            principle_stats.push(stats);
        }

        Ok(OverallAdherenceStats {
            window_hours,
            total_events,
            total_adhered,
            total_violations: total_events.saturating_sub(total_adhered),
            overall_adherence_rate: if total_events > 0 {
                total_adhered as f64 / total_events as f64
            } else {
                1.0
            },
            by_principle: principle_stats,
        })
    }

    /// Get recent adherence events.
    pub fn recent_events(
        &self,
        limit: usize,
    ) -> Result<Vec<AdherenceEvent>, crate::error::NagualError> {
        use rusqlite::Connection;

        let conn = Connection::open(&self.db_path)
            .map_err(|e| crate::error::NagualError::internal(e.to_string()))?;

        let mut stmt = conn
            .prepare(
                r#"
                SELECT id, principle, adhered, context, confidence, timestamp, pattern_id
                FROM principle_adherence
                ORDER BY timestamp DESC
                LIMIT ?1
                "#,
            )
            .map_err(|e| crate::error::NagualError::internal(e.to_string()))?;

        let events = stmt
            .query_map([limit], |row| {
                let principle_num: u8 = row.get(1)?;
                let adhered: i32 = row.get(2)?;
                let timestamp_str: String = row.get(5)?;

                Ok(AdherenceEvent {
                    id: row.get(0)?,
                    principle: Principle::from_number(principle_num)
                        .unwrap_or(Principle::SeekTruth),
                    adhered: adhered != 0,
                    context: row.get(3)?,
                    confidence: row.get(4)?,
                    timestamp: DateTime::parse_from_rfc3339(&timestamp_str)
                        .map(|dt| dt.with_timezone(&Utc))
                        .unwrap_or_else(|_| Utc::now()),
                    pattern_id: row.get(6)?,
                })
            })
            .map_err(|e| crate::error::NagualError::internal(e.to_string()))?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| crate::error::NagualError::internal(e.to_string()))?;

        Ok(events)
    }
}

/// Statistics for a single principle's adherence.
#[derive(Debug, Clone, Serialize)]
pub struct PrincipleStats {
    /// The principle.
    pub principle: Principle,
    /// Total events recorded.
    pub total_events: usize,
    /// Number of adherence events.
    pub adhered_count: usize,
    /// Number of violation events.
    pub violation_count: usize,
    /// Adherence rate (0.0-1.0).
    pub adherence_rate: f64,
    /// Average confidence of assessments.
    pub avg_confidence: f64,
}

/// Overall adherence statistics across all principles.
#[derive(Debug, Clone, Serialize)]
pub struct OverallAdherenceStats {
    /// Time window in hours (None = all time).
    pub window_hours: Option<u32>,
    /// Total events across all principles.
    pub total_events: usize,
    /// Total adherence events.
    pub total_adhered: usize,
    /// Total violation events.
    pub total_violations: usize,
    /// Overall adherence rate.
    pub overall_adherence_rate: f64,
    /// Per-principle breakdown.
    pub by_principle: Vec<PrincipleStats>,
}

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // PRINCIPLE TESTS
    // ========================================================================

    #[test]
    fn test_principle_count() {
        assert_eq!(Principle::ALL.len(), 8);
    }

    #[test]
    fn test_principle_numbers() {
        assert_eq!(Principle::SeekTruth.number(), 0);
        assert_eq!(Principle::Partnership.number(), 1);
        assert_eq!(Principle::PartnerCreator.number(), 2);
        assert_eq!(Principle::Impeccability.number(), 3);
        assert_eq!(Principle::EpistemicHumility.number(), 4);
        assert_eq!(Principle::DoNoHarm.number(), 5);
        assert_eq!(Principle::Transparency.number(), 6);
        assert_eq!(Principle::WarriorOptimization.number(), 7);
    }

    #[test]
    fn test_principle_from_number() {
        assert_eq!(Principle::from_number(0), Some(Principle::SeekTruth));
        assert_eq!(Principle::from_number(7), Some(Principle::WarriorOptimization));
        assert_eq!(Principle::from_number(8), None);
        assert_eq!(Principle::from_number(255), None);
    }

    #[test]
    fn test_principle_roundtrip() {
        for p in &Principle::ALL {
            let n = p.number();
            assert_eq!(Principle::from_number(n), Some(*p));
        }
    }

    #[test]
    fn test_principle_has_name() {
        for p in &Principle::ALL {
            assert!(!p.name().is_empty());
        }
    }

    #[test]
    fn test_principle_has_summary() {
        for p in &Principle::ALL {
            assert!(!p.summary().is_empty());
        }
    }

    #[test]
    fn test_principle_has_description() {
        for p in &Principle::ALL {
            assert!(!p.description().is_empty());
            assert!(p.description().len() > 50); // Descriptions should be substantial
        }
    }

    #[test]
    fn test_principle_has_quote() {
        for p in &Principle::ALL {
            assert!(!p.quote().is_empty());
        }
    }

    #[test]
    fn test_principle_format_short() {
        let formatted = Principle::SeekTruth.format_short();
        assert!(formatted.contains("Principle 0"));
        assert!(formatted.contains("Seek Truth"));
    }

    #[test]
    fn test_principle_format_full() {
        let formatted = Principle::Impeccability.format_full();
        assert!(formatted.contains("### Principle 3"));
        assert!(formatted.contains("Impeccability"));
        assert!(formatted.contains("Don Juan"));
    }

    #[test]
    fn test_principle_display() {
        let s = format!("{}", Principle::EpistemicHumility);
        assert_eq!(s, "Epistemic Humility");
    }

    #[test]
    fn test_constitution_principles_access() {
        assert_eq!(Constitution::principles().len(), 8);
        assert_eq!(Constitution::principle(0), Some(Principle::SeekTruth));
        assert_eq!(Constitution::principle(8), None);
    }

    #[test]
    fn test_startup_greeting() {
        let greeting = Constitution::startup_greeting();
        assert!(greeting.starts_with("[nagual]"));
        assert!(greeting.contains("Principle"));
    }

    // ========================================================================
    // ENFORCEMENT MODE TESTS
    // ========================================================================

    #[test]
    fn test_enforcement_mode_default() {
        let constitution = Constitution::new();
        assert_eq!(constitution.mode(), EnforcementMode::Warn);
    }

    #[test]
    fn test_enforcement_mode_block() {
        let constitution = Constitution::with_enforcement();
        assert_eq!(constitution.mode(), EnforcementMode::Block);
    }

    #[test]
    fn test_enforcement_mode_custom() {
        let constitution = Constitution::with_mode(EnforcementMode::Audit);
        assert_eq!(constitution.mode(), EnforcementMode::Audit);
    }

    #[test]
    fn test_status_summary() {
        let constitution = Constitution::new();
        let summary = constitution.status_summary();
        assert!(summary.contains("8 principles"));
        assert!(summary.contains("5 rules"));
        assert!(summary.contains("warn"));
    }

    // ========================================================================
    // RULE TESTS
    // ========================================================================

    #[test]
    fn test_constitution_default_rules() {
        let constitution = Constitution::new();
        assert_eq!(constitution.rule_names().len(), 5);
        assert_eq!(constitution.rule_count(), 5);
    }

    #[test]
    fn test_delete_without_backup_warning() {
        let constitution = Constitution::new();
        let context = OperationContext {
            operation: Operation::Delete,
            pattern_id: Some("test-123".to_string()),
            reward: None,
            tier: None,
            surprise_score: None,
            has_recent_backup: false,
            failure_mode: None,
            domain: None,
        };

        let results = constitution.check(&context);
        assert!(results.iter().any(|r| !r.allowed));
    }

    #[test]
    fn test_delete_with_backup_allowed() {
        let constitution = Constitution::new();
        let context = OperationContext {
            operation: Operation::Delete,
            pattern_id: Some("test-123".to_string()),
            reward: None,
            tier: None,
            surprise_score: None,
            has_recent_backup: true,
            failure_mode: None,
            domain: None,
        };

        let results = constitution.check(&context);
        let backup_check = results.iter().find(|r| r.rule == "NeverDeleteWithoutBackup");
        assert!(backup_check.map(|r| r.allowed).unwrap_or(true));
    }

    #[test]
    fn test_warn_mode_allows_violations() {
        let constitution = Constitution::with_mode(EnforcementMode::Warn);
        let context = OperationContext {
            operation: Operation::Delete,
            pattern_id: Some("test-123".to_string()),
            reward: None,
            tier: None,
            surprise_score: None,
            has_recent_backup: false,
            failure_mode: None,
            domain: None,
        };

        assert!(constitution.is_allowed(&context));
    }

    #[test]
    fn test_audit_mode_allows_violations() {
        let constitution = Constitution::with_mode(EnforcementMode::Audit);
        let context = OperationContext {
            operation: Operation::Delete,
            pattern_id: Some("test-123".to_string()),
            reward: None,
            tier: None,
            surprise_score: None,
            has_recent_backup: false,
            failure_mode: None,
            domain: None,
        };

        assert!(constitution.is_allowed(&context));
    }

    #[test]
    fn test_block_mode_blocks_violations() {
        let constitution = Constitution::with_enforcement();
        let context = OperationContext {
            operation: Operation::Delete,
            pattern_id: Some("test-123".to_string()),
            reward: None,
            tier: None,
            surprise_score: None,
            has_recent_backup: false,
            failure_mode: None,
            domain: None,
        };

        assert!(!constitution.is_allowed(&context));
    }

    #[test]
    fn test_failure_without_mast() {
        let constitution = Constitution::new();
        let context = OperationContext {
            operation: Operation::RecordFailure,
            pattern_id: Some("test-123".to_string()),
            reward: None,
            tier: None,
            surprise_score: None,
            has_recent_backup: false,
            failure_mode: None, // No MAST classification!
            domain: None,
        };

        let results = constitution.check(&context);
        assert!(results.iter().any(|r| r.rule == "AlwaysRecordMAST" && !r.allowed));
    }

    #[test]
    fn test_reflex_minimum_reward() {
        let constitution = Constitution::new();
        let context = OperationContext {
            operation: Operation::Promote,
            pattern_id: Some("test-123".to_string()),
            reward: Some(0.85), // Below 0.9 threshold for reflex
            tier: Some("reflex".to_string()),
            surprise_score: None,
            has_recent_backup: false,
            failure_mode: None,
            domain: None,
        };

        let results = constitution.check(&context);
        assert!(results.iter().any(|r| r.rule == "MinimumRewardForReflex" && !r.allowed));
    }

    // ========================================================================
    // SERIALIZATION TESTS
    // ========================================================================

    #[test]
    fn test_principle_serialize() {
        let json = serde_json::to_string(&Principle::SeekTruth).unwrap();
        assert_eq!(json, "\"seek_truth\"");
    }

    #[test]
    fn test_principle_deserialize() {
        let p: Principle = serde_json::from_str("\"epistemic_humility\"").unwrap();
        assert_eq!(p, Principle::EpistemicHumility);
    }

    #[test]
    fn test_enforcement_mode_serialize() {
        let json = serde_json::to_string(&EnforcementMode::Block).unwrap();
        assert_eq!(json, "\"block\"");
    }

    // ========================================================================
    // ADHERENCE TRACKING TESTS
    // ========================================================================

    #[test]
    fn test_adherence_event_creation() {
        let event = AdherenceEvent::new(
            Principle::SeekTruth,
            true,
            "Verified claim with evidence",
        );
        assert!(event.adhered);
        assert_eq!(event.principle, Principle::SeekTruth);
        assert_eq!(event.confidence, 1.0);
        assert!(event.pattern_id.is_none());
    }

    #[test]
    fn test_adherence_event_with_confidence() {
        let event = AdherenceEvent::new(Principle::EpistemicHumility, false, "Made overconfident claim")
            .with_confidence(0.7);
        assert!(!event.adhered);
        assert_eq!(event.confidence, 0.7);
    }

    #[test]
    fn test_adherence_event_with_pattern() {
        let event = AdherenceEvent::new(Principle::DoNoHarm, true, "Backed up before delete")
            .with_pattern("pattern-123");
        assert_eq!(event.pattern_id, Some("pattern-123".to_string()));
    }

    #[test]
    fn test_adherence_event_confidence_clamping() {
        let event = AdherenceEvent::new(Principle::Transparency, true, "Shared reasoning")
            .with_confidence(1.5);
        assert_eq!(event.confidence, 1.0); // Clamped to max

        let event2 = AdherenceEvent::new(Principle::Transparency, true, "Shared reasoning")
            .with_confidence(-0.5);
        assert_eq!(event2.confidence, 0.0); // Clamped to min
    }

    #[test]
    fn test_adherence_tracker_init_and_record() {
        let tmp_path = std::env::temp_dir().join("nagual_test_adherence.db");
        let _ = std::fs::remove_file(&tmp_path);

        let tracker = AdherenceTracker::new(&tmp_path);
        tracker.init_schema().unwrap();

        let event = AdherenceEvent::new(Principle::Partnership, true, "Collaborated on solution");
        tracker.record(&event).unwrap();

        let recent = tracker.recent_events(10).unwrap();
        assert_eq!(recent.len(), 1);
        assert_eq!(recent[0].principle, Principle::Partnership);
        assert!(recent[0].adhered);

        let _ = std::fs::remove_file(&tmp_path);
    }

    #[test]
    fn test_adherence_tracker_stats() {
        let tmp_path = std::env::temp_dir().join("nagual_test_adherence_stats.db");
        let _ = std::fs::remove_file(&tmp_path);

        let tracker = AdherenceTracker::new(&tmp_path);
        tracker.init_schema().unwrap();

        // Record some events
        tracker.record(&AdherenceEvent::new(Principle::SeekTruth, true, "event 1")).unwrap();
        tracker.record(&AdherenceEvent::new(Principle::SeekTruth, true, "event 2")).unwrap();
        tracker.record(&AdherenceEvent::new(Principle::SeekTruth, false, "event 3")).unwrap();

        let stats = tracker.principle_stats(Principle::SeekTruth, None).unwrap();
        assert_eq!(stats.total_events, 3);
        assert_eq!(stats.adhered_count, 2);
        assert_eq!(stats.violation_count, 1);
        assert!((stats.adherence_rate - 0.666).abs() < 0.01);

        let _ = std::fs::remove_file(&tmp_path);
    }

    #[test]
    fn test_adherence_tracker_overall_stats() {
        let tmp_path = std::env::temp_dir().join("nagual_test_adherence_overall.db");
        let _ = std::fs::remove_file(&tmp_path);

        let tracker = AdherenceTracker::new(&tmp_path);
        tracker.init_schema().unwrap();

        // Record events for multiple principles
        tracker.record(&AdherenceEvent::new(Principle::SeekTruth, true, "truth")).unwrap();
        tracker.record(&AdherenceEvent::new(Principle::Partnership, true, "partner")).unwrap();
        tracker.record(&AdherenceEvent::new(Principle::DoNoHarm, false, "harm")).unwrap();

        let overall = tracker.overall_stats(None).unwrap();
        assert_eq!(overall.total_events, 3);
        assert_eq!(overall.total_adhered, 2);
        assert_eq!(overall.total_violations, 1);
        assert!((overall.overall_adherence_rate - 0.666).abs() < 0.01);

        let _ = std::fs::remove_file(&tmp_path);
    }
}
