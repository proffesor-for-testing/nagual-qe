//! MCP Tool Definitions for Nagual.
//!
//! Defines the MCP (Model Context Protocol) tool schemas that allow
//! LLMs to interact with the Nagual system through structured APIs.
//!
//! # Tool Categories
//!
//! - **Pattern Tools**: Store, search, and manage reasoning patterns
//! - **Learning Tools**: Record outcomes and get learning insights
//! - **Prediction Tools**: Create and query predictions
//!
//! # JSON Schema
//!
//! Each tool has a defined JSON Schema for input validation and
//! output formatting, ensuring type-safe interaction with LLMs.

use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::HashMap;

/// Tool definition with name, description, and JSON schema.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    /// Unique tool name
    pub name: String,
    /// Human-readable description for the LLM
    pub description: String,
    /// JSON Schema for input parameters
    pub input_schema: serde_json::Value,
    /// JSON Schema for output (optional, for documentation)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_schema: Option<serde_json::Value>,
    /// Whether this tool requires authentication
    #[serde(default)]
    pub requires_auth: bool,
    /// Category for grouping
    #[serde(default)]
    pub category: String,
}

impl ToolDefinition {
    /// Create a new tool definition.
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        input_schema: serde_json::Value,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            input_schema,
            output_schema: None,
            requires_auth: false,
            category: "general".to_string(),
        }
    }

    /// Set the output schema.
    pub fn with_output_schema(mut self, schema: serde_json::Value) -> Self {
        self.output_schema = Some(schema);
        self
    }

    /// Set whether authentication is required.
    pub fn requires_authentication(mut self) -> Self {
        self.requires_auth = true;
        self
    }

    /// Set the category.
    pub fn with_category(mut self, category: impl Into<String>) -> Self {
        self.category = category.into();
        self
    }
}

// ============================================================================
// Tool Schemas - Input Types
// ============================================================================

/// Input for storing a pattern.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorePatternInput {
    /// The problem description
    pub problem: String,
    /// The solution description
    pub solution: String,
    /// Domain/category (e.g., "rust.async", "database.postgres")
    #[serde(default)]
    pub domain: Option<String>,
    /// Additional context
    #[serde(default)]
    pub context: Option<String>,
    /// Initial confidence score (0.0-1.0)
    #[serde(default)]
    pub confidence: Option<f32>,
    /// Tags for categorization
    #[serde(default)]
    pub tags: Vec<String>,
    /// Session ID
    #[serde(default)]
    pub session_id: Option<String>,
    /// Agent ID
    #[serde(default)]
    pub agent_id: Option<String>,
    /// Additional metadata
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Input for searching patterns.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchPatternsInput {
    /// Search query (natural language or keywords)
    pub query: String,
    /// Maximum number of results
    #[serde(default = "default_limit")]
    pub limit: usize,
    /// Domains to filter by
    #[serde(default)]
    pub domains: Vec<String>,
    /// Minimum reward threshold (0.0-1.0)
    #[serde(default)]
    pub min_reward: Option<f32>,
    /// Minimum effectiveness threshold (0.0-1.0)
    #[serde(default)]
    pub min_effectiveness: Option<f32>,
    /// Include only successful patterns
    #[serde(default)]
    pub success_only: bool,
    /// Tags to filter by (any match)
    #[serde(default)]
    pub tags: Vec<String>,
    /// Use MMR for diversity
    #[serde(default = "default_true")]
    pub use_mmr: bool,
    /// MMR lambda (0.0 = max diversity, 1.0 = max relevance)
    #[serde(default = "default_mmr_lambda")]
    pub mmr_lambda: f32,
}

fn default_limit() -> usize {
    10
}

fn default_true() -> bool {
    true
}

fn default_mmr_lambda() -> f32 {
    0.5
}

/// Input for recording an outcome.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecordOutcomeInput {
    /// Pattern ID the outcome applies to
    pub pattern_id: String,
    /// Outcome type
    pub outcome: OutcomeType,
    /// Optional feedback
    #[serde(default)]
    pub feedback: Option<String>,
    /// Confidence in the assessment (0.0-1.0)
    #[serde(default)]
    pub confidence: Option<f32>,
    /// Context relevance (0.0-1.0)
    #[serde(default)]
    pub context_relevance: Option<f32>,
    /// Whether the outcome was verified
    #[serde(default)]
    pub verified: bool,
    /// Session ID
    #[serde(default)]
    pub session_id: Option<String>,
    /// Agent ID
    #[serde(default)]
    pub agent_id: Option<String>,
}

/// Outcome type for pattern application.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OutcomeType {
    /// Pattern was fully successful
    Success,
    /// Pattern was partially successful
    PartialSuccess,
    /// Pattern had neutral results
    Neutral,
    /// Pattern failed
    Failure,
}

impl OutcomeType {
    /// Convert to SONA Outcome.
    pub fn to_outcome_string(&self) -> &'static str {
        match self {
            OutcomeType::Success => "success",
            OutcomeType::PartialSuccess => "partial_success",
            OutcomeType::Neutral => "neutral",
            OutcomeType::Failure => "failure",
        }
    }
}

/// Input for getting insights.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GetInsightsInput {
    /// Domain to get insights for (optional, for all domains if not specified)
    #[serde(default)]
    pub domain: Option<String>,
    /// Time window for analysis
    #[serde(default)]
    pub time_window: Option<TimeWindow>,
    /// Include trend analysis
    #[serde(default = "default_true")]
    pub include_trends: bool,
    /// Include recommendations
    #[serde(default = "default_true")]
    pub include_recommendations: bool,
    /// Maximum patterns to analyze
    #[serde(default = "default_insights_limit")]
    pub max_patterns: usize,
}

fn default_insights_limit() -> usize {
    100
}

/// Time window for analysis.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TimeWindow {
    /// Last 24 hours
    Day,
    /// Last 7 days
    Week,
    /// Last 30 days
    Month,
    /// Last 90 days
    Quarter,
    /// Last 365 days
    Year,
    /// All time
    AllTime,
}

/// Input for creating a prediction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictInput {
    /// Description of what to predict
    pub description: String,
    /// Domain for the prediction
    #[serde(default)]
    pub domain: Option<String>,
    /// Context for the prediction
    #[serde(default)]
    pub context: Option<String>,
    /// Pattern IDs to use as evidence
    #[serde(default)]
    pub evidence_patterns: Vec<String>,
    /// Minimum timeline in days
    #[serde(default)]
    pub timeline_min_days: Option<u32>,
    /// Maximum timeline in days
    #[serde(default)]
    pub timeline_max_days: Option<u32>,
    /// Tags for the prediction
    #[serde(default)]
    pub tags: Vec<String>,
    /// Session ID
    #[serde(default)]
    pub session_id: Option<String>,
    /// Agent ID
    #[serde(default)]
    pub agent_id: Option<String>,
}

// ============================================================================
// Tool Schemas - Output Types
// ============================================================================

/// Output for store pattern operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorePatternOutput {
    /// Whether the operation succeeded
    pub success: bool,
    /// The pattern ID (new or existing)
    pub pattern_id: String,
    /// Message
    pub message: String,
    /// Domain it was stored in
    pub domain: String,
}

/// Output for search patterns operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchPatternsOutput {
    /// Whether the operation succeeded
    pub success: bool,
    /// List of matching patterns
    pub patterns: Vec<PatternResult>,
    /// Total patterns found
    pub total_found: usize,
    /// Query that was executed
    pub query: String,
    /// Search latency in milliseconds
    pub latency_ms: u64,
}

/// A single pattern result from search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternResult {
    /// Pattern ID
    pub id: String,
    /// Problem description
    pub problem: String,
    /// Solution description
    pub solution: String,
    /// Domain
    pub domain: String,
    /// Similarity score (0.0-1.0)
    pub similarity: f32,
    /// Reward/quality score
    pub reward: f32,
    /// Effectiveness score
    pub effectiveness: f32,
    /// Success status
    pub success: bool,
    /// Reuse count
    pub reuse_count: u32,
    /// Tags
    pub tags: Vec<String>,
}

/// Output for record outcome operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecordOutcomeOutput {
    /// Whether the operation succeeded
    pub success: bool,
    /// Calculated reward value
    pub reward: f32,
    /// Updated pattern effectiveness
    pub new_effectiveness: f32,
    /// Updated pattern reward
    pub new_reward: f32,
    /// Message
    pub message: String,
}

/// Output for get insights operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GetInsightsOutput {
    /// Whether the operation succeeded
    pub success: bool,
    /// Domain analyzed
    pub domain: String,
    /// Total patterns analyzed
    pub patterns_analyzed: usize,
    /// Average reward across patterns
    pub average_reward: f32,
    /// Average effectiveness
    pub average_effectiveness: f32,
    /// Success rate
    pub success_rate: f32,
    /// Top performing patterns
    pub top_patterns: Vec<TopPatternSummary>,
    /// Trend analysis
    #[serde(skip_serializing_if = "Option::is_none")]
    pub trend: Option<TrendSummary>,
    /// Recommendations
    #[serde(default)]
    pub recommendations: Vec<String>,
}

/// Summary of a top pattern.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopPatternSummary {
    /// Pattern ID
    pub id: String,
    /// Problem snippet
    pub problem_snippet: String,
    /// Quality score
    pub quality_score: f32,
    /// Reuse count
    pub reuse_count: u32,
}

/// Summary of trends.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendSummary {
    /// Trend direction
    pub direction: String,
    /// Change percentage
    pub change_percent: f32,
    /// Time period
    pub period: String,
}

/// Output for predict operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictOutput {
    /// Whether the operation succeeded
    pub success: bool,
    /// Prediction ID
    pub prediction_id: String,
    /// Predicted probability
    pub probability: f64,
    /// Confidence in the prediction
    pub confidence: f64,
    /// Expected timeline
    pub timeline: TimelineRange,
    /// Number of evidence patterns used
    pub evidence_count: usize,
    /// Probability breakdown
    #[serde(skip_serializing_if = "Option::is_none")]
    pub breakdown: Option<ProbabilityBreakdown>,
    /// Message
    pub message: String,
}

/// Timeline range.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimelineRange {
    /// Minimum days
    pub min_days: u32,
    /// Maximum days
    pub max_days: u32,
    /// Midpoint (expected)
    pub expected_days: u32,
}

/// Breakdown of probability calculation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProbabilityBreakdown {
    /// Base probability
    pub base: f64,
    /// Pattern-based adjustment
    pub pattern_adjustment: f64,
    /// Confidence adjustment
    pub confidence_adjustment: f64,
    /// Calibration adjustment
    #[serde(skip_serializing_if = "Option::is_none")]
    pub calibration_adjustment: Option<f64>,
}

// ============================================================================
// Tool Definitions Factory
// ============================================================================

/// Get all Nagual MCP tool definitions.
pub fn get_all_tool_definitions() -> Vec<ToolDefinition> {
    vec![
        nagual_store_pattern(),
        nagual_search_patterns(),
        nagual_record_outcome(),
        nagual_get_insights(),
        nagual_predict(),
    ]
}

/// Tool definition for storing a pattern.
pub fn nagual_store_pattern() -> ToolDefinition {
    ToolDefinition::new(
        "nagual_store_pattern",
        "Store a new reasoning pattern in the Nagual knowledge base. \
         Patterns capture problem-solution pairs that can be retrieved later \
         for similar situations. Use this to record successful approaches, \
         best practices, and learned solutions.",
        json!({
            "type": "object",
            "properties": {
                "problem": {
                    "type": "string",
                    "description": "Clear description of the problem or challenge"
                },
                "solution": {
                    "type": "string",
                    "description": "The solution or approach that addresses the problem"
                },
                "domain": {
                    "type": "string",
                    "description": "Domain/category using dot notation (e.g., 'rust.async', 'database.postgres')"
                },
                "context": {
                    "type": "string",
                    "description": "Additional context about when/where this pattern applies"
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "Initial confidence in this pattern (0.0-1.0)"
                },
                "tags": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "Tags for categorization and search"
                },
                "session_id": {
                    "type": "string",
                    "description": "Session identifier for tracking"
                },
                "agent_id": {
                    "type": "string",
                    "description": "Agent identifier that created this pattern"
                },
                "metadata": {
                    "type": "object",
                    "description": "Additional metadata as key-value pairs"
                }
            },
            "required": ["problem", "solution"]
        }),
    )
    .with_output_schema(json!({
        "type": "object",
        "properties": {
            "success": { "type": "boolean" },
            "pattern_id": { "type": "string" },
            "message": { "type": "string" },
            "domain": { "type": "string" }
        }
    }))
    .with_category("patterns")
}

/// Tool definition for searching patterns.
pub fn nagual_search_patterns() -> ToolDefinition {
    ToolDefinition::new(
        "nagual_search_patterns",
        "Search for relevant reasoning patterns in the Nagual knowledge base. \
         Uses semantic similarity to find patterns that match the query. \
         Supports filtering by domain, reward threshold, and tags. \
         Use MMR (Maximal Marginal Relevance) for diverse results.",
        json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query (natural language or keywords)"
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 100,
                    "default": 10,
                    "description": "Maximum number of results to return"
                },
                "domains": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "Filter results to these domains"
                },
                "min_reward": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "Minimum reward score (0.0-1.0)"
                },
                "min_effectiveness": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "Minimum effectiveness score (0.0-1.0)"
                },
                "success_only": {
                    "type": "boolean",
                    "default": false,
                    "description": "Only return successful patterns"
                },
                "tags": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "Filter by tags (any match)"
                },
                "use_mmr": {
                    "type": "boolean",
                    "default": true,
                    "description": "Use Maximal Marginal Relevance for diverse results"
                },
                "mmr_lambda": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "default": 0.5,
                    "description": "MMR lambda (0.0 = max diversity, 1.0 = max relevance)"
                }
            },
            "required": ["query"]
        }),
    )
    .with_output_schema(json!({
        "type": "object",
        "properties": {
            "success": { "type": "boolean" },
            "patterns": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": { "type": "string" },
                        "problem": { "type": "string" },
                        "solution": { "type": "string" },
                        "domain": { "type": "string" },
                        "similarity": { "type": "number" },
                        "reward": { "type": "number" },
                        "effectiveness": { "type": "number" },
                        "success": { "type": "boolean" },
                        "reuse_count": { "type": "integer" },
                        "tags": { "type": "array", "items": { "type": "string" } }
                    }
                }
            },
            "total_found": { "type": "integer" },
            "query": { "type": "string" },
            "latency_ms": { "type": "integer" }
        }
    }))
    .with_category("patterns")
}

/// Tool definition for recording an outcome.
pub fn nagual_record_outcome() -> ToolDefinition {
    ToolDefinition::new(
        "nagual_record_outcome",
        "Record the outcome of applying a pattern. This is essential for the \
         SONA learning loop - patterns improve based on feedback. Record success, \
         partial success, neutral results, or failures to help the system learn \
         which patterns work best in different situations.",
        json!({
            "type": "object",
            "properties": {
                "pattern_id": {
                    "type": "string",
                    "description": "The ID of the pattern that was applied"
                },
                "outcome": {
                    "type": "string",
                    "enum": ["success", "partial_success", "neutral", "failure"],
                    "description": "The outcome of applying the pattern"
                },
                "feedback": {
                    "type": "string",
                    "description": "Optional feedback or notes about the outcome"
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "Confidence in this assessment (0.0-1.0)"
                },
                "context_relevance": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "How relevant the pattern was to the context (0.0-1.0)"
                },
                "verified": {
                    "type": "boolean",
                    "default": false,
                    "description": "Whether this outcome was verified (increases confidence)"
                },
                "session_id": {
                    "type": "string",
                    "description": "Session identifier"
                },
                "agent_id": {
                    "type": "string",
                    "description": "Agent identifier"
                }
            },
            "required": ["pattern_id", "outcome"]
        }),
    )
    .with_output_schema(json!({
        "type": "object",
        "properties": {
            "success": { "type": "boolean" },
            "reward": { "type": "number" },
            "new_effectiveness": { "type": "number" },
            "new_reward": { "type": "number" },
            "message": { "type": "string" }
        }
    }))
    .with_category("learning")
}

/// Tool definition for getting insights.
pub fn nagual_get_insights() -> ToolDefinition {
    ToolDefinition::new(
        "nagual_get_insights",
        "Get insights and analytics about patterns in the knowledge base. \
         Shows performance metrics, trends over time, top performing patterns, \
         and recommendations for improvement. Use this to understand how \
         well patterns are working and identify areas for optimization.",
        json!({
            "type": "object",
            "properties": {
                "domain": {
                    "type": "string",
                    "description": "Domain to analyze (optional, all domains if not specified)"
                },
                "time_window": {
                    "type": "string",
                    "enum": ["day", "week", "month", "quarter", "year", "all_time"],
                    "default": "month",
                    "description": "Time window for analysis"
                },
                "include_trends": {
                    "type": "boolean",
                    "default": true,
                    "description": "Include trend analysis"
                },
                "include_recommendations": {
                    "type": "boolean",
                    "default": true,
                    "description": "Include recommendations"
                },
                "max_patterns": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 1000,
                    "default": 100,
                    "description": "Maximum patterns to analyze"
                }
            }
        }),
    )
    .with_output_schema(json!({
        "type": "object",
        "properties": {
            "success": { "type": "boolean" },
            "domain": { "type": "string" },
            "patterns_analyzed": { "type": "integer" },
            "average_reward": { "type": "number" },
            "average_effectiveness": { "type": "number" },
            "success_rate": { "type": "number" },
            "top_patterns": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": { "type": "string" },
                        "problem_snippet": { "type": "string" },
                        "quality_score": { "type": "number" },
                        "reuse_count": { "type": "integer" }
                    }
                }
            },
            "trend": {
                "type": "object",
                "properties": {
                    "direction": { "type": "string" },
                    "change_percent": { "type": "number" },
                    "period": { "type": "string" }
                }
            },
            "recommendations": {
                "type": "array",
                "items": { "type": "string" }
            }
        }
    }))
    .with_category("learning")
}

/// Tool definition for creating predictions.
pub fn nagual_predict() -> ToolDefinition {
    ToolDefinition::new(
        "nagual_predict",
        "Create a probabilistic prediction about a future outcome based on \
         reasoning patterns. The prediction includes a probability estimate, \
         confidence level, and expected timeline. Predictions are tracked and \
         calibrated using Brier scores for continuous improvement.",
        json!({
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "Description of what is being predicted"
                },
                "domain": {
                    "type": "string",
                    "description": "Domain for the prediction"
                },
                "context": {
                    "type": "string",
                    "description": "Additional context for the prediction"
                },
                "evidence_patterns": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "Pattern IDs to use as evidence"
                },
                "timeline_min_days": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Minimum expected days until resolution"
                },
                "timeline_max_days": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Maximum expected days until resolution"
                },
                "tags": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "Tags for the prediction"
                },
                "session_id": {
                    "type": "string",
                    "description": "Session identifier"
                },
                "agent_id": {
                    "type": "string",
                    "description": "Agent identifier"
                }
            },
            "required": ["description"]
        }),
    )
    .with_output_schema(json!({
        "type": "object",
        "properties": {
            "success": { "type": "boolean" },
            "prediction_id": { "type": "string" },
            "probability": { "type": "number" },
            "confidence": { "type": "number" },
            "timeline": {
                "type": "object",
                "properties": {
                    "min_days": { "type": "integer" },
                    "max_days": { "type": "integer" },
                    "expected_days": { "type": "integer" }
                }
            },
            "evidence_count": { "type": "integer" },
            "breakdown": {
                "type": "object",
                "properties": {
                    "base": { "type": "number" },
                    "pattern_adjustment": { "type": "number" },
                    "confidence_adjustment": { "type": "number" },
                    "calibration_adjustment": { "type": "number" }
                }
            },
            "message": { "type": "string" }
        }
    }))
    .with_category("predictions")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tool_definition_creation() {
        let tool = nagual_store_pattern();
        assert_eq!(tool.name, "nagual_store_pattern");
        assert_eq!(tool.category, "patterns");
        assert!(!tool.requires_auth);
        assert!(tool.output_schema.is_some());
    }

    #[test]
    fn test_all_tool_definitions() {
        let tools = get_all_tool_definitions();
        assert_eq!(tools.len(), 5);

        let names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();
        assert!(names.contains(&"nagual_store_pattern"));
        assert!(names.contains(&"nagual_search_patterns"));
        assert!(names.contains(&"nagual_record_outcome"));
        assert!(names.contains(&"nagual_get_insights"));
        assert!(names.contains(&"nagual_predict"));
    }

    #[test]
    fn test_store_pattern_input_serialization() {
        let input = StorePatternInput {
            problem: "How to handle errors?".to_string(),
            solution: "Use Result type".to_string(),
            domain: Some("rust.error_handling".to_string()),
            context: Some("Production code".to_string()),
            confidence: Some(0.8),
            tags: vec!["rust".to_string(), "errors".to_string()],
            session_id: None,
            agent_id: None,
            metadata: HashMap::new(),
        };

        let json = serde_json::to_string(&input).unwrap();
        let deserialized: StorePatternInput = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.problem, input.problem);
        assert_eq!(deserialized.domain, input.domain);
    }

    #[test]
    fn test_search_patterns_input_defaults() {
        let json = r#"{"query": "error handling"}"#;
        let input: SearchPatternsInput = serde_json::from_str(json).unwrap();

        assert_eq!(input.query, "error handling");
        assert_eq!(input.limit, 10);
        assert!(input.use_mmr);
        assert!((input.mmr_lambda - 0.5).abs() < 0.001);
        assert!(!input.success_only);
    }

    #[test]
    fn test_outcome_type() {
        assert_eq!(OutcomeType::Success.to_outcome_string(), "success");
        assert_eq!(OutcomeType::PartialSuccess.to_outcome_string(), "partial_success");
        assert_eq!(OutcomeType::Neutral.to_outcome_string(), "neutral");
        assert_eq!(OutcomeType::Failure.to_outcome_string(), "failure");
    }

    #[test]
    fn test_pattern_result_serialization() {
        let result = PatternResult {
            id: "test-123".to_string(),
            problem: "Test problem".to_string(),
            solution: "Test solution".to_string(),
            domain: "test.domain".to_string(),
            similarity: 0.95,
            reward: 0.8,
            effectiveness: 0.85,
            success: true,
            reuse_count: 5,
            tags: vec!["tag1".to_string()],
        };

        let json = serde_json::to_string(&result).unwrap();
        let deserialized: PatternResult = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.id, result.id);
        assert!((deserialized.similarity - 0.95).abs() < 0.001);
    }

    #[test]
    fn test_tool_schema_validity() {
        // Verify that input schemas are valid JSON
        for tool in get_all_tool_definitions() {
            assert!(tool.input_schema.is_object());
            assert!(tool.input_schema.get("type").is_some());
            assert!(tool.input_schema.get("properties").is_some());
        }
    }

    #[test]
    fn test_predict_output_serialization() {
        let output = PredictOutput {
            success: true,
            prediction_id: "pred-123".to_string(),
            probability: 0.75,
            confidence: 0.85,
            timeline: TimelineRange {
                min_days: 7,
                max_days: 14,
                expected_days: 10,
            },
            evidence_count: 3,
            breakdown: Some(ProbabilityBreakdown {
                base: 0.5,
                pattern_adjustment: 0.2,
                confidence_adjustment: 0.05,
                calibration_adjustment: None,
            }),
            message: "Prediction created".to_string(),
        };

        let json = serde_json::to_string_pretty(&output).unwrap();
        assert!(json.contains("prediction_id"));
        assert!(json.contains("0.75"));
    }
}
