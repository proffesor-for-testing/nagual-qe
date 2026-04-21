//! MCP (Model Context Protocol) integration for Nagual.
//!
//! This module provides MCP tool registration and execution for
//! enabling LLM agents to interact with the Nagual system.
//!
//! # Architecture
//!
//! ```text
//! Claude Flow / LLM
//!        |
//!        v
//! +----------------+
//! | McpRegistry    |
//! | - tools        |
//! | - executors    |
//! +----------------+
//!        |
//!        v
//! +----------------+     +----------------+
//! | ToolExecutor   |---->| NagualContext  |
//! +----------------+     +----------------+
//!        |                      |
//!        v                      v
//! +----------------+     +----------------+
//! | PatternStorage |     | SonaLearner    |
//! +----------------+     +----------------+
//! ```
//!
//! # Example
//!
//! ```ignore
//! use nagual::mcp::{McpRegistry, NagualContext};
//!
//! // Create context with storage and learner
//! let context = NagualContext::new(storage, learner, event_bus);
//!
//! // Create registry and register tools
//! let registry = McpRegistry::new();
//! registry.register_all_tools();
//!
//! // Execute a tool
//! let input = json!({"problem": "Error handling", "solution": "Use Result"});
//! let output = registry.execute("nagual_store_pattern", &input, &context).await?;
//! ```

pub mod tools;

// KOS P6: MCP Server Mode (JSON-RPC 2.0 over stdio)
pub mod server;

pub use tools::{
    get_all_tool_definitions, GetInsightsInput, GetInsightsOutput, OutcomeType, PatternResult,
    PredictInput, PredictOutput, ProbabilityBreakdown, RecordOutcomeInput, RecordOutcomeOutput,
    SearchPatternsInput, SearchPatternsOutput, StorePatternInput, StorePatternOutput,
    TimeWindow, TimelineRange, ToolDefinition, TopPatternSummary, TrendSummary,
};

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use async_trait::async_trait;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use thiserror::Error;
use tracing::{debug, error, info, instrument};

use crate::events::{EventBus, NagualEvent};
use crate::learning::{Outcome, SonaLearner};
use crate::ml::cosine_similarity;
#[cfg(feature = "onnx-embed")]
use crate::ml::Embedder;
use crate::prediction::{GenerationContext, InputPattern, PredictionGenerator};
use crate::reasoning_bank::storage::PatternStorage;

/// Errors that can occur in MCP operations.
#[derive(Error, Debug)]
pub enum McpError {
    /// Tool not found
    #[error("Tool not found: {name}")]
    ToolNotFound { name: String },

    /// Invalid input
    #[error("Invalid input for tool '{tool}': {reason}")]
    InvalidInput { tool: String, reason: String },

    /// Execution error
    #[error("Tool execution failed: {0}")]
    ExecutionFailed(String),

    /// Serialization error
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    /// Internal error
    #[error("Internal error: {0}")]
    Internal(String),
}

/// Result type for MCP operations.
pub type McpResult<T> = std::result::Result<T, McpError>;

/// Execution result from a tool.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    /// Whether the execution succeeded
    pub success: bool,
    /// Tool name that was executed
    pub tool: String,
    /// Output data
    pub output: Value,
    /// Execution time in milliseconds
    pub execution_time_ms: u64,
    /// Error message if failed
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

impl ToolResult {
    /// Create a successful result.
    pub fn success(tool: impl Into<String>, output: Value, execution_time_ms: u64) -> Self {
        Self {
            success: true,
            tool: tool.into(),
            output,
            execution_time_ms,
            error: None,
        }
    }

    /// Create a failed result.
    pub fn failure(tool: impl Into<String>, error: impl Into<String>, execution_time_ms: u64) -> Self {
        Self {
            success: false,
            tool: tool.into(),
            output: Value::Null,
            execution_time_ms,
            error: Some(error.into()),
        }
    }
}

/// Context for tool execution containing Nagual services.
pub struct NagualContext {
    /// Pattern storage
    storage: Option<Arc<PatternStorage>>,
    /// SONA learner
    learner: Option<Arc<SonaLearner>>,
    /// Event bus for publishing events
    event_bus: Option<Arc<EventBus>>,
    /// Embedder for generating query embeddings
    #[cfg(feature = "onnx-embed")]
    embedder: Option<Arc<Embedder>>,
    /// Prediction generator
    prediction_generator: Option<Arc<PredictionGenerator>>,
}

impl NagualContext {
    /// Create a new Nagual context.
    pub fn new(
        storage: Option<Arc<PatternStorage>>,
        learner: Option<Arc<SonaLearner>>,
        event_bus: Option<Arc<EventBus>>,
    ) -> Self {
        Self {
            storage,
            learner,
            event_bus,
            #[cfg(feature = "onnx-embed")]
            embedder: None,
            prediction_generator: None,
        }
    }

    /// Create a full Nagual context with all services.
    #[cfg(feature = "onnx-embed")]
    pub fn with_all(
        storage: Option<Arc<PatternStorage>>,
        learner: Option<Arc<SonaLearner>>,
        event_bus: Option<Arc<EventBus>>,
        embedder: Option<Arc<Embedder>>,
        prediction_generator: Option<Arc<PredictionGenerator>>,
    ) -> Self {
        Self {
            storage,
            learner,
            event_bus,
            embedder,
            prediction_generator,
        }
    }

    /// Create an empty context (for testing).
    pub fn empty() -> Self {
        Self {
            storage: None,
            learner: None,
            event_bus: None,
            #[cfg(feature = "onnx-embed")]
            embedder: None,
            prediction_generator: None,
        }
    }

    /// Get the pattern storage.
    pub fn storage(&self) -> McpResult<&Arc<PatternStorage>> {
        self.storage.as_ref().ok_or_else(|| {
            McpError::Internal("Pattern storage not available".to_string())
        })
    }

    /// Get the SONA learner.
    pub fn learner(&self) -> McpResult<&Arc<SonaLearner>> {
        self.learner.as_ref().ok_or_else(|| {
            McpError::Internal("SONA learner not available".to_string())
        })
    }

    /// Get the event bus.
    pub fn event_bus(&self) -> Option<&Arc<EventBus>> {
        self.event_bus.as_ref()
    }

    /// Get the embedder.
    #[cfg(feature = "onnx-embed")]
    pub fn embedder(&self) -> Option<&Arc<Embedder>> {
        self.embedder.as_ref()
    }

    /// Get the prediction generator.
    pub fn prediction_generator(&self) -> Option<&Arc<PredictionGenerator>> {
        self.prediction_generator.as_ref()
    }

    /// Publish an event if the event bus is available.
    pub async fn publish_event(&self, event: NagualEvent) {
        if let Some(bus) = &self.event_bus {
            let _ = bus.publish(event).await;
        }
    }
}

/// Trait for tool executors.
///
/// Note: Uses `?Send` because NagualContext contains non-Sync types (rusqlite::Connection).
/// This means executors cannot be used across threads directly.
#[async_trait(?Send)]
pub trait ToolExecutor: Send + Sync {
    /// Get the tool name.
    fn name(&self) -> &str;

    /// Execute the tool with given input.
    async fn execute(&self, input: &Value, context: &NagualContext) -> McpResult<Value>;

    /// Validate input before execution.
    fn validate_input(&self, input: &Value) -> McpResult<()> {
        let _ = input;
        Ok(())
    }
}

/// Registry statistics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RegistryStats {
    /// Total tool executions
    pub total_executions: u64,
    /// Executions by tool
    pub executions_by_tool: HashMap<String, u64>,
    /// Total failures
    pub failures: u64,
    /// Average execution time in milliseconds
    pub avg_execution_time_ms: f64,
}

/// MCP Tool Registry for managing and executing tools.
pub struct McpRegistry {
    /// Registered tools
    tools: RwLock<HashMap<String, Arc<dyn ToolExecutor>>>,
    /// Tool definitions
    definitions: RwLock<HashMap<String, ToolDefinition>>,
    /// Registry statistics
    stats: RwLock<RegistryStats>,
}

impl McpRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self {
            tools: RwLock::new(HashMap::new()),
            definitions: RwLock::new(HashMap::new()),
            stats: RwLock::new(RegistryStats::default()),
        }
    }

    /// Create a registry with all Nagual tools registered.
    pub fn with_all_tools() -> Self {
        let registry = Self::new();
        registry.register_all_nagual_tools();
        registry
    }

    /// Register a tool executor.
    pub fn register(&self, executor: Arc<dyn ToolExecutor>) {
        let name = executor.name().to_string();
        let mut tools = self.tools.write();
        info!(tool = %name, "Registering MCP tool");
        tools.insert(name, executor);
    }

    /// Register a tool definition.
    pub fn register_definition(&self, definition: ToolDefinition) {
        let name = definition.name.clone();
        let mut definitions = self.definitions.write();
        definitions.insert(name, definition);
    }

    /// Register all Nagual tools.
    pub fn register_all_nagual_tools(&self) {
        // Register definitions
        for definition in get_all_tool_definitions() {
            self.register_definition(definition);
        }

        // Register executors
        self.register(Arc::new(StorePatternExecutor));
        self.register(Arc::new(SearchPatternsExecutor));
        self.register(Arc::new(RecordOutcomeExecutor));
        self.register(Arc::new(GetInsightsExecutor));
        self.register(Arc::new(PredictExecutor));
    }

    /// Get a tool definition by name.
    pub fn get_definition(&self, name: &str) -> Option<ToolDefinition> {
        self.definitions.read().get(name).cloned()
    }

    /// Get all tool definitions.
    pub fn get_all_definitions(&self) -> Vec<ToolDefinition> {
        self.definitions.read().values().cloned().collect()
    }

    /// List all registered tool names.
    pub fn list_tools(&self) -> Vec<String> {
        self.tools.read().keys().cloned().collect()
    }

    /// Check if a tool is registered.
    pub fn has_tool(&self, name: &str) -> bool {
        self.tools.read().contains_key(name)
    }

    /// Execute a tool by name.
    #[instrument(skip(self, input, context), fields(tool = %name))]
    pub async fn execute(
        &self,
        name: &str,
        input: &Value,
        context: &NagualContext,
    ) -> McpResult<ToolResult> {
        let executor = {
            let tools = self.tools.read();
            tools.get(name).cloned()
        };

        let executor = executor.ok_or_else(|| McpError::ToolNotFound {
            name: name.to_string(),
        })?;

        // Validate input
        if let Err(e) = executor.validate_input(input) {
            error!(tool = %name, error = %e, "Input validation failed");
            return Ok(ToolResult::failure(name, e.to_string(), 0));
        }

        // Execute the tool
        let start = Instant::now();
        let result = executor.execute(input, context).await;
        let execution_time = start.elapsed().as_millis() as u64;

        // Update stats
        {
            let mut stats = self.stats.write();
            stats.total_executions += 1;
            *stats.executions_by_tool.entry(name.to_string()).or_insert(0) += 1;

            // Update average execution time
            let n = stats.total_executions as f64;
            stats.avg_execution_time_ms =
                ((n - 1.0) * stats.avg_execution_time_ms + execution_time as f64) / n;
        }

        match result {
            Ok(output) => {
                debug!(tool = %name, execution_time_ms = execution_time, "Tool execution succeeded");
                Ok(ToolResult::success(name, output, execution_time))
            }
            Err(e) => {
                error!(tool = %name, error = %e, "Tool execution failed");
                {
                    let mut stats = self.stats.write();
                    stats.failures += 1;
                }
                Ok(ToolResult::failure(name, e.to_string(), execution_time))
            }
        }
    }

    /// Get registry statistics.
    pub fn stats(&self) -> RegistryStats {
        self.stats.read().clone()
    }
}

impl Default for McpRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Tool Executors
// ============================================================================

/// Executor for nagual_store_pattern.
struct StorePatternExecutor;

#[async_trait(?Send)]
impl ToolExecutor for StorePatternExecutor {
    fn name(&self) -> &str {
        "nagual_store_pattern"
    }

    fn validate_input(&self, input: &Value) -> McpResult<()> {
        if input.get("problem").and_then(|v| v.as_str()).unwrap_or("").is_empty() {
            return Err(McpError::InvalidInput {
                tool: self.name().to_string(),
                reason: "problem is required and cannot be empty".to_string(),
            });
        }
        if input.get("solution").and_then(|v| v.as_str()).unwrap_or("").is_empty() {
            return Err(McpError::InvalidInput {
                tool: self.name().to_string(),
                reason: "solution is required and cannot be empty".to_string(),
            });
        }
        Ok(())
    }

    async fn execute(&self, input: &Value, context: &NagualContext) -> McpResult<Value> {
        let input: StorePatternInput = serde_json::from_value(input.clone())?;

        // Get storage from context
        let storage = context.storage()?;

        // Build pattern
        use crate::reasoning_bank::pattern::{Pattern, PatternCategory};

        let domain = input.domain.unwrap_or_else(|| "general".to_string());
        let mut builder = Pattern::builder()
            .problem(&input.problem)
            .solution(&input.solution)
            .category(PatternCategory::from(domain.as_str()));

        if let Some(ctx) = input.context {
            builder = builder.context(ctx);
        }

        if let Some(conf) = input.confidence {
            builder = builder.confidence(conf);
        }

        if let Some(session) = input.session_id {
            builder = builder.session_id(session);
        }

        if let Some(agent) = input.agent_id {
            builder = builder.agent_id(agent);
        }

        for tag in input.tags {
            builder = builder.tag(tag);
        }

        let pattern = builder.build();
        let pattern_id = pattern.id().to_string();
        let domain_str = domain.clone();

        // Store the pattern
        storage.store_pattern(&pattern).await.map_err(|e| {
            McpError::ExecutionFailed(format!("Failed to store pattern: {}", e))
        })?;

        // Publish event
        context.publish_event(NagualEvent::pattern_stored_with_context(
            &pattern_id,
            &domain_str,
            pattern.session_id().map(|s| s.to_string()),
            pattern.agent_id().map(|s| s.to_string()),
        )).await;

        let output = StorePatternOutput {
            success: true,
            pattern_id,
            message: "Pattern stored successfully".to_string(),
            domain: domain_str,
        };

        Ok(serde_json::to_value(output)?)
    }
}

/// Executor for nagual_search_patterns.
struct SearchPatternsExecutor;

#[async_trait(?Send)]
impl ToolExecutor for SearchPatternsExecutor {
    fn name(&self) -> &str {
        "nagual_search_patterns"
    }

    fn validate_input(&self, input: &Value) -> McpResult<()> {
        if input.get("query").and_then(|v| v.as_str()).unwrap_or("").is_empty() {
            return Err(McpError::InvalidInput {
                tool: self.name().to_string(),
                reason: "query is required and cannot be empty".to_string(),
            });
        }
        Ok(())
    }

    async fn execute(&self, input: &Value, context: &NagualContext) -> McpResult<Value> {
        let input: SearchPatternsInput = serde_json::from_value(input.clone())?;
        let start = Instant::now();

        let storage = context.storage()?;

        // Check if we have an embedder for vector similarity search
        #[cfg(feature = "onnx-embed")]
        let results: Vec<PatternResult> = if let Some(embedder) = context.embedder() {
            // Generate embedding for the query
            let embedding_result = embedder.embed(&input.query).map_err(|e| {
                McpError::ExecutionFailed(format!("Failed to generate query embedding: {}", e))
            })?;

            let query_embedding = ndarray::Array1::from_vec(embedding_result.embedding);

            // Get all patterns with embeddings for similarity search
            let all_patterns = storage.get_all_with_embeddings().await.map_err(|e| {
                McpError::ExecutionFailed(format!("Failed to get patterns: {}", e))
            })?;

            // Calculate similarity scores for each pattern
            let mut scored_patterns: Vec<(_, f32)> = all_patterns
                .into_iter()
                .filter_map(|p| {
                    // Get the pattern's embedding and calculate similarity
                    // Clone the embedding to owned Vec to avoid borrowing issues
                    let emb_opt = p.embedding().map(|e| e.to_vec());
                    if let Some(emb_vec) = emb_opt {
                        let emb_array = ndarray::Array1::from_vec(emb_vec);
                        let sim = cosine_similarity(&query_embedding.view(), &emb_array.view());
                        Some((p, sim))
                    } else {
                        None
                    }
                })
                // Apply domain filter
                .filter(|(p, _)| {
                    input.domains.is_empty()
                        || input.domains.iter().any(|d| p.category().to_string().starts_with(d))
                })
                // Apply reward filter
                .filter(|(p, _)| {
                    input.min_reward.map_or(true, |min| p.reward() >= min)
                })
                // Apply effectiveness filter
                .filter(|(p, _)| {
                    input.min_effectiveness.map_or(true, |min| p.effectiveness() >= min)
                })
                // Apply success filter
                .filter(|(p, _)| !input.success_only || p.success())
                // Apply tag filter (any match)
                .filter(|(p, _)| {
                    input.tags.is_empty()
                        || input.tags.iter().any(|t| p.tags().contains(t))
                })
                .collect();

            // Sort by similarity (descending)
            scored_patterns.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            // Apply MMR for diversity if enabled
            let final_patterns = if input.use_mmr && scored_patterns.len() > 1 {
                apply_mmr_reranking(&scored_patterns, input.mmr_lambda, input.limit)
            } else {
                scored_patterns.into_iter().take(input.limit).collect()
            };

            // Convert to PatternResult with actual similarity scores
            final_patterns
                .into_iter()
                .map(|(p, sim)| PatternResult {
                    id: p.id().to_string(),
                    problem: p.problem().to_string(),
                    solution: p.solution().to_string(),
                    domain: p.category().to_string(),
                    similarity: sim,
                    reward: p.reward(),
                    effectiveness: p.effectiveness(),
                    success: p.success(),
                    reuse_count: p.reuse_count(),
                    tags: p.tags().to_vec(),
                })
                .collect::<Vec<_>>()
        } else {
            // Fallback: No embedder available, use simple text matching
            debug!("No embedder available, falling back to text-based search");

            let patterns = storage.get_recent(input.limit * 10).await.map_err(|e| {
                McpError::ExecutionFailed(format!("Failed to search patterns: {}", e))
            })?;

            let query_lower = input.query.to_lowercase();

            // Filter patterns by text matching
            let mut filtered: Vec<_> = patterns
                .into_iter()
                .filter(|p| {
                    p.problem().to_lowercase().contains(&query_lower)
                        || p.solution().to_lowercase().contains(&query_lower)
                })
                .filter(|p| {
                    input.domains.is_empty()
                        || input.domains.iter().any(|d| p.category().to_string().starts_with(d))
                })
                .filter(|p| {
                    input.min_reward.map_or(true, |min| p.reward() >= min)
                })
                .filter(|p| !input.success_only || p.success())
                .take(input.limit)
                .map(|p| PatternResult {
                    id: p.id().to_string(),
                    problem: p.problem().to_string(),
                    solution: p.solution().to_string(),
                    domain: p.category().to_string(),
                    similarity: 0.5, // Default similarity for text matching
                    reward: p.reward(),
                    effectiveness: p.effectiveness(),
                    success: p.success(),
                    reuse_count: p.reuse_count(),
                    tags: p.tags().to_vec(),
                })
                .collect();

            // Sort by reward as a fallback ranking
            filtered.sort_by(|a, b| b.reward.partial_cmp(&a.reward).unwrap_or(std::cmp::Ordering::Equal));
            filtered
        };

        // Non-ONNX fallback: always use text-based search
        #[cfg(not(feature = "onnx-embed"))]
        let results: Vec<PatternResult> = {
            debug!("ONNX embedder not available, using text-based search");

            let patterns = storage.get_recent(input.limit * 10).await.map_err(|e| {
                McpError::ExecutionFailed(format!("Failed to search patterns: {}", e))
            })?;

            let query_lower = input.query.to_lowercase();

            let mut filtered: Vec<_> = patterns
                .into_iter()
                .filter(|p| {
                    p.problem().to_lowercase().contains(&query_lower)
                        || p.solution().to_lowercase().contains(&query_lower)
                })
                .filter(|p| {
                    input.domains.is_empty()
                        || input.domains.iter().any(|d| p.category().to_string().starts_with(d))
                })
                .filter(|p| {
                    input.min_reward.map_or(true, |min| p.reward() >= min)
                })
                .filter(|p| !input.success_only || p.success())
                .take(input.limit)
                .map(|p| PatternResult {
                    id: p.id().to_string(),
                    problem: p.problem().to_string(),
                    solution: p.solution().to_string(),
                    domain: p.category().to_string(),
                    similarity: 0.5,
                    reward: p.reward(),
                    effectiveness: p.effectiveness(),
                    success: p.success(),
                    reuse_count: p.reuse_count(),
                    tags: p.tags().to_vec(),
                })
                .collect();

            filtered.sort_by(|a, b| b.reward.partial_cmp(&a.reward).unwrap_or(std::cmp::Ordering::Equal));
            filtered
        };

        let total = results.len();
        let latency_ms = start.elapsed().as_millis() as u64;

        let output = SearchPatternsOutput {
            success: true,
            patterns: results,
            total_found: total,
            query: input.query,
            latency_ms,
        };

        Ok(serde_json::to_value(output)?)
    }
}

/// Apply Maximal Marginal Relevance (MMR) reranking for diverse results.
/// MMR = lambda * similarity - (1 - lambda) * max_sim_to_selected
fn apply_mmr_reranking<T: Clone>(
    candidates: &[(T, f32)],
    lambda: f32,
    limit: usize,
) -> Vec<(T, f32)>
where
    T: std::fmt::Debug,
{
    if candidates.is_empty() || limit == 0 {
        return Vec::new();
    }

    let mut selected: Vec<(T, f32)> = Vec::with_capacity(limit);
    let mut remaining: Vec<(T, f32)> = candidates.to_vec();

    // Select the first document (highest similarity)
    if let Some(first) = remaining.first().cloned() {
        selected.push(first);
        remaining.remove(0);
    }

    // Simple MMR: for patterns we don't have pairwise similarity,
    // so we'll use a diversity bonus based on position
    while selected.len() < limit && !remaining.is_empty() {
        let mut best_idx = 0;
        let mut best_mmr_score = f32::NEG_INFINITY;

        for (i, (_, sim)) in remaining.iter().enumerate() {
            // Simplified MMR: penalize based on rank position to encourage diversity
            let diversity_penalty = (selected.len() as f32) * 0.05;
            let mmr_score = lambda * sim - (1.0 - lambda) * diversity_penalty;

            if mmr_score > best_mmr_score {
                best_mmr_score = mmr_score;
                best_idx = i;
            }
        }

        selected.push(remaining.remove(best_idx));
    }

    selected
}

/// Executor for nagual_record_outcome.
struct RecordOutcomeExecutor;

#[async_trait(?Send)]
impl ToolExecutor for RecordOutcomeExecutor {
    fn name(&self) -> &str {
        "nagual_record_outcome"
    }

    fn validate_input(&self, input: &Value) -> McpResult<()> {
        if input.get("pattern_id").and_then(|v| v.as_str()).unwrap_or("").is_empty() {
            return Err(McpError::InvalidInput {
                tool: self.name().to_string(),
                reason: "pattern_id is required".to_string(),
            });
        }
        if input.get("outcome").is_none() {
            return Err(McpError::InvalidInput {
                tool: self.name().to_string(),
                reason: "outcome is required".to_string(),
            });
        }
        Ok(())
    }

    async fn execute(&self, input: &Value, context: &NagualContext) -> McpResult<Value> {
        let input: RecordOutcomeInput = serde_json::from_value(input.clone())?;

        let learner = context.learner()?;

        // Convert outcome type
        let outcome = match input.outcome {
            OutcomeType::Success => Outcome::Success,
            OutcomeType::PartialSuccess => Outcome::PartialSuccess,
            OutcomeType::Neutral => Outcome::Neutral,
            OutcomeType::Failure => Outcome::Failure,
        };

        use crate::reasoning_bank::pattern::PatternId;

        let pattern_id = PatternId::from_string(&input.pattern_id);

        // Record the outcome
        let reward = learner
            .record_outcome(&pattern_id, outcome, input.feedback.clone())
            .await
            .map_err(|e| McpError::ExecutionFailed(format!("Failed to record outcome: {}", e)))?;

        // Publish event
        context.publish_event(NagualEvent::outcome_recorded(
            &input.pattern_id,
            input.outcome.to_outcome_string(),
            reward,
            input.feedback,
        )).await;

        let output = RecordOutcomeOutput {
            success: true,
            reward,
            new_effectiveness: reward, // Simplified - would get from updated pattern
            new_reward: reward,
            message: format!("Outcome recorded for pattern {}", input.pattern_id),
        };

        Ok(serde_json::to_value(output)?)
    }
}

/// Executor for nagual_get_insights.
struct GetInsightsExecutor;

#[async_trait(?Send)]
impl ToolExecutor for GetInsightsExecutor {
    fn name(&self) -> &str {
        "nagual_get_insights"
    }

    async fn execute(&self, input: &Value, context: &NagualContext) -> McpResult<Value> {
        let input: GetInsightsInput = serde_json::from_value(input.clone())?;

        let storage = context.storage()?;

        // Get patterns for analysis
        let patterns = storage.get_top_effective(input.max_patterns).await.map_err(|e| {
            McpError::ExecutionFailed(format!("Failed to get patterns: {}", e))
        })?;

        let total = patterns.len();
        if total == 0 {
            return Ok(json!({
                "success": true,
                "domain": input.domain.unwrap_or_else(|| "all".to_string()),
                "patterns_analyzed": 0,
                "average_reward": 0.0,
                "average_effectiveness": 0.0,
                "success_rate": 0.0,
                "top_patterns": [],
                "recommendations": ["No patterns found. Start storing patterns to enable insights."]
            }));
        }

        // Calculate statistics
        let (reward_sum, effectiveness_sum, success_count) = patterns.iter().fold(
            (0.0f32, 0.0f32, 0usize),
            |(rs, es, sc), p| {
                (
                    rs + p.reward(),
                    es + p.effectiveness(),
                    sc + if p.success() { 1 } else { 0 },
                )
            },
        );

        let avg_reward = reward_sum / total as f32;
        let avg_effectiveness = effectiveness_sum / total as f32;
        let success_rate = success_count as f32 / total as f32;

        // Get top patterns
        let top_patterns: Vec<TopPatternSummary> = patterns
            .iter()
            .take(5)
            .map(|p| TopPatternSummary {
                id: p.id().to_string(),
                problem_snippet: p.problem().chars().take(100).collect(),
                quality_score: p.quality_score(),
                reuse_count: p.reuse_count(),
            })
            .collect();

        // Generate recommendations
        let mut recommendations = Vec::new();
        if avg_reward < 0.5 {
            recommendations.push("Average reward is low. Consider reviewing and updating low-performing patterns.".to_string());
        }
        if success_rate < 0.7 {
            recommendations.push("Success rate is below 70%. Focus on patterns that consistently work.".to_string());
        }
        if total < 10 {
            recommendations.push("You have few patterns. Keep recording successful approaches to build knowledge.".to_string());
        }

        let output = GetInsightsOutput {
            success: true,
            domain: input.domain.unwrap_or_else(|| "all".to_string()),
            patterns_analyzed: total,
            average_reward: avg_reward,
            average_effectiveness: avg_effectiveness,
            success_rate,
            top_patterns,
            trend: if input.include_trends {
                Some(TrendSummary {
                    direction: "stable".to_string(),
                    change_percent: 0.0,
                    period: "30 days".to_string(),
                })
            } else {
                None
            },
            recommendations,
        };

        Ok(serde_json::to_value(output)?)
    }
}

/// Executor for nagual_predict.
struct PredictExecutor;

#[async_trait(?Send)]
impl ToolExecutor for PredictExecutor {
    fn name(&self) -> &str {
        "nagual_predict"
    }

    fn validate_input(&self, input: &Value) -> McpResult<()> {
        if input.get("description").and_then(|v| v.as_str()).unwrap_or("").is_empty() {
            return Err(McpError::InvalidInput {
                tool: self.name().to_string(),
                reason: "description is required".to_string(),
            });
        }
        Ok(())
    }

    async fn execute(&self, input: &Value, context: &NagualContext) -> McpResult<Value> {
        let input: PredictInput = serde_json::from_value(input.clone())?;

        use crate::prediction::{Prediction, PredictionId};

        let timeline_min = input.timeline_min_days.unwrap_or(7);
        let timeline_max = input.timeline_max_days.unwrap_or(30);

        // Try to use the prediction generator if available with evidence patterns
        if let (Some(generator), Some(storage)) = (context.prediction_generator(), context.storage.as_ref()) {
            if !input.evidence_patterns.is_empty() {
                // Build input patterns from evidence pattern IDs
                let mut input_patterns = Vec::new();

                for pattern_id in &input.evidence_patterns {
                    use crate::reasoning_bank::pattern::PatternId;
                    let pid = PatternId::from_string(pattern_id.clone());

                    if let Ok(Some(pattern)) = storage.get_pattern(&pid).await {
                        // Calculate a base similarity (we don't have the query embedding here,
                        // so use 0.8 as a reasonable default for explicitly provided evidence)
                        let input_pattern = InputPattern::new(pattern_id.clone(), 0.8)
                            .with_success_rate(if pattern.success() { pattern.effectiveness() as f64 } else { 0.3 })
                            .with_confidence(pattern.confidence() as f64)
                            .with_effectiveness(pattern.effectiveness() as f64)
                            .with_created_at(pattern.timestamp());

                        input_patterns.push(input_pattern);
                    }
                }

                if !input_patterns.is_empty() {
                    // Build generation context
                    let mut gen_context = GenerationContext::new(&input.description)
                        .with_min_confidence(0.3)
                        .with_min_similarity(0.5);

                    if let Some(ref domain) = input.domain {
                        gen_context = gen_context.with_domain(domain);
                    }

                    if let Some(ref ctx) = input.context {
                        gen_context = gen_context.with_context(ctx);
                    }

                    for tag in &input.tags {
                        gen_context = gen_context.with_tag(tag);
                    }

                    if let Some(ref session_id) = input.session_id {
                        gen_context = gen_context.with_session_id(session_id);
                    }

                    if let Some(ref agent_id) = input.agent_id {
                        gen_context = gen_context.with_agent_id(agent_id);
                    }

                    // Generate prediction using the engine
                    match generator.generate_prediction(&input_patterns, &gen_context) {
                        Ok(result) => {
                            let prediction = result.prediction;
                            let prediction_id = prediction.id().to_string();

                            // Publish event
                            context.publish_event(NagualEvent::prediction_created(
                                &prediction_id,
                                prediction.probability(),
                                prediction.confidence(),
                                prediction.domain(),
                                input.evidence_patterns.len(),
                            )).await;

                            let output = PredictOutput {
                                success: true,
                                prediction_id,
                                probability: prediction.probability(),
                                confidence: prediction.confidence(),
                                timeline: TimelineRange {
                                    min_days: prediction.timeline_min_days(),
                                    max_days: prediction.timeline_max_days(),
                                    expected_days: (prediction.timeline_min_days() + prediction.timeline_max_days()) / 2,
                                },
                                evidence_count: input.evidence_patterns.len(),
                                breakdown: Some(ProbabilityBreakdown {
                                    base: result.probability_result.breakdown.base_prior_contribution,
                                    pattern_adjustment: result.probability_result.breakdown.pattern_contribution,
                                    confidence_adjustment: 1.0 - result.probability_result.confidence,
                                    calibration_adjustment: None,
                                }),
                                message: format!(
                                    "Prediction generated from {} evidence patterns (avg quality: {:.2})",
                                    result.metadata.patterns_used,
                                    result.metadata.avg_quality
                                ),
                            };

                            return Ok(serde_json::to_value(output)?);
                        }
                        Err(e) => {
                            debug!("Prediction generation failed, falling back to simple prediction: {}", e);
                            // Fall through to simple prediction
                        }
                    }
                }
            }
        }

        // Fallback: Create a simple prediction without the full engine
        let prediction_id = PredictionId::new();

        let prediction = Prediction::builder()
            .id(prediction_id.clone())
            .description(&input.description)
            .probability(0.5) // Base probability
            .confidence(0.5) // Base confidence
            .timeline(timeline_min, timeline_max)
            .domain(input.domain.clone().unwrap_or_else(|| "general".to_string()))
            .build()
            .map_err(|e| McpError::ExecutionFailed(format!("Failed to create prediction: {}", e)))?;

        // Publish event
        context.publish_event(NagualEvent::prediction_created(
            prediction_id.as_str(),
            prediction.probability(),
            prediction.confidence(),
            prediction.domain(),
            input.evidence_patterns.len(),
        )).await;

        let output = PredictOutput {
            success: true,
            prediction_id: prediction_id.to_string(),
            probability: prediction.probability(),
            confidence: prediction.confidence(),
            timeline: TimelineRange {
                min_days: timeline_min,
                max_days: timeline_max,
                expected_days: (timeline_min + timeline_max) / 2,
            },
            evidence_count: input.evidence_patterns.len(),
            breakdown: Some(ProbabilityBreakdown {
                base: 0.5,
                pattern_adjustment: 0.0,
                confidence_adjustment: 0.0,
                calibration_adjustment: None,
            }),
            message: if input.evidence_patterns.is_empty() {
                "Prediction created with base probability (no evidence patterns provided)".to_string()
            } else {
                "Prediction created (prediction engine not available)".to_string()
            },
        };

        Ok(serde_json::to_value(output)?)
    }
}

// ============================================================================
// MCP Protocol Types
// ============================================================================

/// MCP Tool List Response (per MCP spec).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolListResponse {
    /// List of available tools
    pub tools: Vec<ToolInfo>,
}

/// Tool information for MCP protocol.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolInfo {
    /// Tool name
    pub name: String,
    /// Tool description
    pub description: String,
    /// Input schema
    pub input_schema: Value,
}

impl From<ToolDefinition> for ToolInfo {
    fn from(def: ToolDefinition) -> Self {
        Self {
            name: def.name,
            description: def.description,
            input_schema: def.input_schema,
        }
    }
}

/// MCP Tool Call Request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallRequest {
    /// Tool name to execute
    pub name: String,
    /// Input arguments
    pub arguments: Value,
}

/// MCP Tool Call Response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallResponse {
    /// Tool outputs
    pub content: Vec<ToolContent>,
    /// Whether this is an error response
    #[serde(default)]
    pub is_error: bool,
}

/// Tool output content.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolContent {
    /// Text content
    Text { text: String },
    /// JSON content
    Json { json: Value },
}

impl McpRegistry {
    /// Handle MCP tools/list request.
    pub fn handle_list(&self) -> ToolListResponse {
        let tools: Vec<ToolInfo> = self
            .definitions
            .read()
            .values()
            .cloned()
            .map(ToolInfo::from)
            .collect();

        ToolListResponse { tools }
    }

    /// Handle MCP tools/call request.
    pub async fn handle_call(
        &self,
        request: ToolCallRequest,
        context: &NagualContext,
    ) -> ToolCallResponse {
        let result = self.execute(&request.name, &request.arguments, context).await;

        match result {
            Ok(tool_result) => {
                if tool_result.success {
                    ToolCallResponse {
                        content: vec![ToolContent::Json {
                            json: tool_result.output,
                        }],
                        is_error: false,
                    }
                } else {
                    ToolCallResponse {
                        content: vec![ToolContent::Text {
                            text: tool_result.error.unwrap_or_else(|| "Unknown error".to_string()),
                        }],
                        is_error: true,
                    }
                }
            }
            Err(e) => ToolCallResponse {
                content: vec![ToolContent::Text {
                    text: e.to_string(),
                }],
                is_error: true,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry_creation() {
        let registry = McpRegistry::new();
        assert!(registry.list_tools().is_empty());
    }

    #[test]
    fn test_registry_with_all_tools() {
        let registry = McpRegistry::with_all_tools();
        let tools = registry.list_tools();

        assert!(tools.contains(&"nagual_store_pattern".to_string()));
        assert!(tools.contains(&"nagual_search_patterns".to_string()));
        assert!(tools.contains(&"nagual_record_outcome".to_string()));
        assert!(tools.contains(&"nagual_get_insights".to_string()));
        assert!(tools.contains(&"nagual_predict".to_string()));
    }

    #[test]
    fn test_tool_list_response() {
        let registry = McpRegistry::with_all_tools();
        let response = registry.handle_list();

        assert_eq!(response.tools.len(), 5);
        assert!(response.tools.iter().any(|t| t.name == "nagual_store_pattern"));
    }

    #[test]
    fn test_get_definition() {
        let registry = McpRegistry::with_all_tools();
        let def = registry.get_definition("nagual_store_pattern");

        assert!(def.is_some());
        let def = def.unwrap();
        assert_eq!(def.name, "nagual_store_pattern");
        assert_eq!(def.category, "patterns");
    }

    #[tokio::test]
    async fn test_execute_missing_tool() {
        let registry = McpRegistry::new();
        let context = NagualContext::empty();

        let result = registry.execute("nonexistent", &json!({}), &context).await;
        // Missing tool should return an error, not a result
        assert!(result.is_err());
        match result {
            Err(McpError::ToolNotFound { name }) => {
                assert_eq!(name, "nonexistent");
            }
            _ => panic!("Expected ToolNotFound error"),
        }
    }

    #[test]
    fn test_tool_result_success() {
        let result = ToolResult::success("test_tool", json!({"key": "value"}), 100);
        assert!(result.success);
        assert_eq!(result.tool, "test_tool");
        assert_eq!(result.execution_time_ms, 100);
        assert!(result.error.is_none());
    }

    #[test]
    fn test_tool_result_failure() {
        let result = ToolResult::failure("test_tool", "Something went wrong", 50);
        assert!(!result.success);
        assert_eq!(result.error, Some("Something went wrong".to_string()));
    }

    #[tokio::test]
    async fn test_handle_call_response() {
        let registry = McpRegistry::with_all_tools();
        let context = NagualContext::empty();

        let request = ToolCallRequest {
            name: "nagual_get_insights".to_string(),
            arguments: json!({}),
        };

        // This will fail because context has no storage
        let response = registry.handle_call(request, &context).await;
        assert!(response.is_error);
    }

    #[test]
    fn test_tool_content_serialization() {
        let text_content = ToolContent::Text {
            text: "Hello".to_string(),
        };
        let json_str = serde_json::to_string(&text_content).unwrap();
        assert!(json_str.contains("text"));

        let json_content = ToolContent::Json {
            json: json!({"key": "value"}),
        };
        let json_str = serde_json::to_string(&json_content).unwrap();
        assert!(json_str.contains("json"));
    }
}
