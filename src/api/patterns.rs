//! Patterns namespace API for pattern storage and retrieval.
//!
//! The patterns API provides direct access to the ReasoningBank pattern
//! storage system, enabling storage, search, retrieval, and statistics
//! for self-learning patterns.
//!
//! # Example
//!
//! ```rust,ignore
//! use nagual::reasoning_bank::PatternCategory;
//!
//! // Store a pattern
//! let id = nagual.patterns.store(
//!     "How to optimize database queries",
//!     "Add indexes and use query explain",
//!     PatternCategory::Performance
//! ).await?;
//!
//! // Search for patterns
//! let results = nagual.patterns.search("database optimization").await?;
//!
//! // Retrieve a specific pattern
//! let pattern = nagual.patterns.retrieve(&id).await?;
//!
//! // Get pattern statistics
//! let stats = nagual.patterns.stats().await?;
//! ```


use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tracing::{debug, info, instrument};

use super::NagualState;
use crate::error::{NagualError, Result};
use crate::reasoning_bank::pattern::{Pattern, PatternCategory, PatternId};

/// Result of a pattern search operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternSearchResult {
    /// Matching patterns
    pub patterns: Vec<PatternInfo>,

    /// Search query used
    pub query: String,

    /// Total matches found
    pub total_matches: usize,

    /// Search time in milliseconds
    pub search_time_ms: u64,
}

/// Abbreviated pattern information for search results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternInfo {
    /// Pattern ID
    pub id: String,

    /// Problem description
    pub problem: String,

    /// Solution (truncated)
    pub solution_preview: String,

    /// Category/domain
    pub category: String,

    /// Reward score
    pub reward: f32,

    /// Effectiveness score
    pub effectiveness: f32,

    /// Usage count
    pub usage_count: u32,

    /// Tags
    pub tags: Vec<String>,

    /// Last updated
    pub updated_at: DateTime<Utc>,
}

impl From<Pattern> for PatternInfo {
    fn from(pattern: Pattern) -> Self {
        let solution = pattern.solution();
        let solution_preview = if solution.len() > 200 {
            format!("{}...", &solution[..197])
        } else {
            solution.to_string()
        };

        Self {
            id: pattern.id().to_string(),
            problem: pattern.problem().to_string(),
            solution_preview,
            category: pattern.category().to_string(),
            reward: pattern.reward(),
            effectiveness: pattern.effectiveness(),
            usage_count: pattern.reuse_count(),
            tags: pattern.tags().to_vec(),
            updated_at: pattern.updated_at(),
        }
    }
}

/// Full pattern data for retrieval.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternData {
    /// Pattern ID
    pub id: String,

    /// Problem description
    pub problem: String,

    /// Full solution
    pub solution: String,

    /// Additional context
    pub context: String,

    /// Category/domain
    pub category: String,

    /// Confidence score
    pub confidence: f32,

    /// Reward score
    pub reward: f32,

    /// Effectiveness score
    pub effectiveness: f32,

    /// Success flag
    pub success: bool,

    /// Usage count
    pub usage_count: u32,

    /// Critique/notes
    pub critique: String,

    /// Agent ID (if any)
    pub agent_id: Option<String>,

    /// Session ID (if any)
    pub session_id: Option<String>,

    /// Tags
    pub tags: Vec<String>,

    /// Created timestamp
    pub created_at: DateTime<Utc>,

    /// Updated timestamp
    pub updated_at: DateTime<Utc>,
}

impl From<Pattern> for PatternData {
    fn from(pattern: Pattern) -> Self {
        Self {
            id: pattern.id().to_string(),
            problem: pattern.problem().to_string(),
            solution: pattern.solution().to_string(),
            context: pattern.context().to_string(),
            category: pattern.category().to_string(),
            confidence: pattern.confidence(),
            reward: pattern.reward(),
            effectiveness: pattern.effectiveness(),
            success: pattern.success(),
            usage_count: pattern.reuse_count(),
            critique: pattern.critique().to_string(),
            agent_id: pattern.agent_id().map(|s| s.to_string()),
            session_id: pattern.session_id().map(|s| s.to_string()),
            tags: pattern.tags().to_vec(),
            created_at: pattern.timestamp(),
            updated_at: pattern.updated_at(),
        }
    }
}

/// Pattern statistics result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternStatsResult {
    /// Total pattern count
    pub total_patterns: usize,

    /// Active patterns (reward >= 0.5)
    pub active_patterns: usize,

    /// High-performing patterns (reward >= 0.8)
    pub high_performing: usize,

    /// Low-performing patterns (reward < 0.4)
    pub low_performing: usize,

    /// Average reward across all patterns
    pub avg_reward: f32,

    /// Average effectiveness
    pub avg_effectiveness: f32,

    /// Total usage count
    pub total_usage: u64,

    /// Patterns by category
    pub by_category: std::collections::HashMap<String, usize>,

    /// Top patterns by reward
    pub top_by_reward: Vec<String>,

    /// Top patterns by usage
    pub top_by_usage: Vec<String>,

    /// When stats were generated
    pub generated_at: DateTime<Utc>,
}

/// Options for pattern storage.
#[derive(Debug, Clone, Default)]
pub struct PatternStoreOptions {
    /// Additional context
    pub context: Option<String>,

    /// Initial confidence
    pub confidence: Option<f32>,

    /// Initial reward
    pub reward: Option<f32>,

    /// Tags
    pub tags: Option<Vec<String>>,

    /// Critique/notes
    pub critique: Option<String>,

    /// Agent ID
    pub agent_id: Option<String>,

    /// Session ID
    pub session_id: Option<String>,

    /// Pre-computed embedding vector
    pub embedding: Option<Vec<f32>>,
}

impl PatternStoreOptions {
    /// Create new options with defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set context.
    pub fn context(mut self, context: impl Into<String>) -> Self {
        self.context = Some(context.into());
        self
    }

    /// Set confidence.
    pub fn confidence(mut self, confidence: f32) -> Self {
        self.confidence = Some(confidence.clamp(0.0, 1.0));
        self
    }

    /// Set reward.
    pub fn reward(mut self, reward: f32) -> Self {
        self.reward = Some(reward.clamp(0.0, 1.0));
        self
    }

    /// Set tags.
    pub fn tags(mut self, tags: Vec<String>) -> Self {
        self.tags = Some(tags);
        self
    }

    /// Set critique.
    pub fn critique(mut self, critique: impl Into<String>) -> Self {
        self.critique = Some(critique.into());
        self
    }

    /// Set agent ID.
    pub fn agent_id(mut self, agent_id: impl Into<String>) -> Self {
        self.agent_id = Some(agent_id.into());
        self
    }

    /// Set session ID.
    pub fn session_id(mut self, session_id: impl Into<String>) -> Self {
        self.session_id = Some(session_id.into());
        self
    }

    /// Set embedding.
    pub fn embedding(mut self, embedding: Vec<f32>) -> Self {
        self.embedding = Some(embedding);
        self
    }
}

/// Options for pattern search.
#[derive(Debug, Clone, Default)]
pub struct PatternSearchOptions {
    /// Maximum results
    pub limit: Option<usize>,

    /// Filter by category
    pub category: Option<PatternCategory>,

    /// Minimum reward
    pub min_reward: Option<f32>,

    /// Minimum effectiveness
    pub min_effectiveness: Option<f32>,

    /// Filter by tags (must have all)
    pub tags: Option<Vec<String>>,

    /// Only successful patterns
    pub only_successful: bool,
}

impl PatternSearchOptions {
    /// Create new options with defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set limit.
    pub fn limit(mut self, limit: usize) -> Self {
        self.limit = Some(limit);
        self
    }

    /// Filter by category.
    pub fn category(mut self, category: PatternCategory) -> Self {
        self.category = Some(category);
        self
    }

    /// Set minimum reward.
    pub fn min_reward(mut self, reward: f32) -> Self {
        self.min_reward = Some(reward.clamp(0.0, 1.0));
        self
    }

    /// Set minimum effectiveness.
    pub fn min_effectiveness(mut self, effectiveness: f32) -> Self {
        self.min_effectiveness = Some(effectiveness.clamp(0.0, 1.0));
        self
    }

    /// Filter by tags.
    pub fn tags(mut self, tags: Vec<String>) -> Self {
        self.tags = Some(tags);
        self
    }

    /// Only successful patterns.
    pub fn only_successful(mut self) -> Self {
        self.only_successful = true;
        self
    }
}

/// Pattern storage and retrieval API.
///
/// This API provides direct access to the ReasoningBank pattern storage
/// system for storing, searching, and analyzing patterns.
#[derive(Clone)]
pub struct PatternsApi {
    state: NagualState,
}

impl PatternsApi {
    /// Create a new PatternsApi instance.
    pub(crate) fn new(state: NagualState) -> Self {
        Self { state }
    }

    /// Store a new pattern.
    ///
    /// # Arguments
    ///
    /// * `problem` - The problem description
    /// * `solution` - The solution
    /// * `category` - Pattern category
    ///
    /// # Returns
    ///
    /// The pattern ID.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let id = nagual.patterns.store(
    ///     "How to handle concurrent requests",
    ///     "Use async/await with proper synchronization",
    ///     PatternCategory::Performance
    /// ).await?;
    /// ```
    #[instrument(skip(self, problem, solution), fields(category = %category))]
    pub async fn store(
        &self,
        problem: impl Into<String>,
        solution: impl Into<String>,
        category: PatternCategory,
    ) -> Result<String> {
        self.store_with_options(problem, solution, category, PatternStoreOptions::default())
            .await
    }

    /// Store a new pattern with options.
    ///
    /// # Arguments
    ///
    /// * `problem` - The problem description
    /// * `solution` - The solution
    /// * `category` - Pattern category
    /// * `options` - Storage options
    ///
    /// # Returns
    ///
    /// The pattern ID.
    #[instrument(skip(self, problem, solution, options), fields(category = %category))]
    pub async fn store_with_options(
        &self,
        problem: impl Into<String>,
        solution: impl Into<String>,
        category: PatternCategory,
        options: PatternStoreOptions,
    ) -> Result<String> {
        let problem = problem.into();
        let solution = solution.into();
        let category_for_log = category.clone();

        let mut builder = Pattern::builder()
            .problem(&problem)
            .solution(&solution)
            .category(category);

        if let Some(context) = options.context {
            builder = builder.context(context);
        }

        if let Some(confidence) = options.confidence {
            builder = builder.confidence(confidence);
        }

        if let Some(reward) = options.reward {
            builder = builder.reward(reward);
        }

        if let Some(tags) = options.tags {
            builder = builder.tags(tags);
        }

        if let Some(critique) = options.critique {
            builder = builder.critique(critique);
        }

        if let Some(agent_id) = options.agent_id {
            builder = builder.agent_id(agent_id);
        }

        if let Some(session_id) = options.session_id {
            builder = builder.session_id(session_id);
        }

        if let Some(embedding) = options.embedding {
            builder = builder.embedding(embedding);
        }

        let pattern = builder.build();
        let id = self.state.pattern_storage.store_pattern(&pattern).await?;

        info!(id = %id, category = %category_for_log, "Pattern stored");

        Ok(id.to_string())
    }

    /// Search for patterns.
    ///
    /// # Arguments
    ///
    /// * `query` - Search query
    ///
    /// # Returns
    ///
    /// Search results.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let results = nagual.patterns.search("error handling").await?;
    /// for pattern in results.patterns {
    ///     println!("{}: {}", pattern.id, pattern.problem);
    /// }
    /// ```
    #[instrument(skip(self, query))]
    pub async fn search(&self, query: impl Into<String>) -> Result<PatternSearchResult> {
        self.search_with_options(query, PatternSearchOptions::default())
            .await
    }

    /// Search for patterns with options.
    ///
    /// # Arguments
    ///
    /// * `query` - Search query
    /// * `options` - Search options
    ///
    /// # Returns
    ///
    /// Search results.
    #[instrument(skip(self, query, options))]
    pub async fn search_with_options(
        &self,
        query: impl Into<String>,
        options: PatternSearchOptions,
    ) -> Result<PatternSearchResult> {
        let query_str = query.into();
        let start = std::time::Instant::now();
        let limit = options.limit.unwrap_or(10);

        // Use FTS5 search for efficient O(log n) text search with BM25 ranking
        // Request more results than needed to allow for post-filtering
        let fts_limit = limit * 5;
        let mut patterns = self.state.pattern_storage.fts_search(&query_str, fts_limit).await?;

        // Apply additional filters from options
        if let Some(ref category) = options.category {
            patterns.retain(|p| p.category() == category);
        }

        if let Some(min_reward) = options.min_reward {
            patterns.retain(|p| p.reward() >= min_reward);
        }

        if let Some(min_eff) = options.min_effectiveness {
            patterns.retain(|p| p.effectiveness() >= min_eff);
        }

        if let Some(ref tags) = options.tags {
            patterns.retain(|p| {
                let pattern_tags = p.tags();
                tags.iter().all(|t| pattern_tags.contains(t))
            });
        }

        if options.only_successful {
            patterns.retain(|p| p.success());
        }

        // Truncate to limit
        patterns.truncate(limit);

        let total_matches = patterns.len();
        let pattern_infos: Vec<PatternInfo> = patterns.into_iter().map(PatternInfo::from).collect();

        let search_time_ms = start.elapsed().as_millis() as u64;

        debug!(
            query = %query_str,
            matches = total_matches,
            time_ms = search_time_ms,
            "Pattern search completed (FTS5)"
        );

        Ok(PatternSearchResult {
            patterns: pattern_infos,
            query: query_str,
            total_matches,
            search_time_ms,
        })
    }

    /// Retrieve a pattern by ID.
    ///
    /// # Arguments
    ///
    /// * `id` - Pattern ID
    ///
    /// # Returns
    ///
    /// The full pattern data, or None if not found.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// if let Some(pattern) = nagual.patterns.retrieve(&id).await? {
    ///     println!("Solution: {}", pattern.solution);
    /// }
    /// ```
    #[instrument(skip(self))]
    pub async fn retrieve(&self, id: &str) -> Result<Option<PatternData>> {
        let pattern_id = PatternId::from_string(id);
        let pattern = self.state.pattern_storage.get_pattern(&pattern_id).await?;

        Ok(pattern.map(PatternData::from))
    }

    /// Get pattern statistics.
    ///
    /// # Returns
    ///
    /// Statistics about all stored patterns.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let stats = nagual.patterns.stats().await?;
    /// println!("Total patterns: {}", stats.total_patterns);
    /// println!("Average reward: {:.2}", stats.avg_reward);
    /// ```
    #[instrument(skip(self))]
    pub async fn stats(&self) -> Result<PatternStatsResult> {
        let patterns = self.state.pattern_storage.get_recent(10000).await?;

        let total_patterns = patterns.len();

        if total_patterns == 0 {
            return Ok(PatternStatsResult {
                total_patterns: 0,
                active_patterns: 0,
                high_performing: 0,
                low_performing: 0,
                avg_reward: 0.0,
                avg_effectiveness: 0.0,
                total_usage: 0,
                by_category: std::collections::HashMap::new(),
                top_by_reward: Vec::new(),
                top_by_usage: Vec::new(),
                generated_at: Utc::now(),
            });
        }

        let active_patterns = patterns.iter().filter(|p| p.reward() >= 0.5).count();
        let high_performing = patterns.iter().filter(|p| p.reward() >= 0.8).count();
        let low_performing = patterns.iter().filter(|p| p.reward() < 0.4).count();

        let total_reward: f32 = patterns.iter().map(|p| p.reward()).sum();
        let total_effectiveness: f32 = patterns.iter().map(|p| p.effectiveness()).sum();
        let total_usage: u64 = patterns.iter().map(|p| p.reuse_count() as u64).sum();

        let avg_reward = total_reward / total_patterns as f32;
        let avg_effectiveness = total_effectiveness / total_patterns as f32;

        // Count by category
        let mut by_category: std::collections::HashMap<String, usize> =
            std::collections::HashMap::new();
        for pattern in &patterns {
            *by_category
                .entry(pattern.category().to_string())
                .or_insert(0) += 1;
        }

        // Top by reward
        let mut sorted_by_reward = patterns.clone();
        sorted_by_reward.sort_by(|a, b| {
            b.reward()
                .partial_cmp(&a.reward())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let top_by_reward: Vec<String> = sorted_by_reward
            .iter()
            .take(5)
            .map(|p| p.id().to_string())
            .collect();

        // Top by usage
        let mut sorted_by_usage = patterns;
        sorted_by_usage.sort_by(|a, b| b.reuse_count().cmp(&a.reuse_count()));
        let top_by_usage: Vec<String> = sorted_by_usage
            .iter()
            .take(5)
            .map(|p| p.id().to_string())
            .collect();

        Ok(PatternStatsResult {
            total_patterns,
            active_patterns,
            high_performing,
            low_performing,
            avg_reward,
            avg_effectiveness,
            total_usage,
            by_category,
            top_by_reward,
            top_by_usage,
            generated_at: Utc::now(),
        })
    }

    /// Delete a pattern.
    ///
    /// # Arguments
    ///
    /// * `id` - Pattern ID
    #[instrument(skip(self))]
    pub async fn delete(&self, id: &str) -> Result<()> {
        let pattern_id = PatternId::from_string(id);
        self.state.pattern_storage.delete_pattern(&pattern_id).await?;

        info!(id = %id, "Pattern deleted");
        Ok(())
    }

    /// Update a pattern.
    ///
    /// # Arguments
    ///
    /// * `id` - Pattern ID
    /// * `updates` - Fields to update
    #[instrument(skip(self))]
    pub async fn update(&self, id: &str, updates: PatternStoreOptions) -> Result<()> {
        let pattern_id = PatternId::from_string(id);
        let pattern = self
            .state
            .pattern_storage
            .get_pattern(&pattern_id)
            .await?
            .ok_or_else(|| NagualError::internal(format!("Pattern not found: {}", id)))?;

        let mut builder = Pattern::builder()
            .id(pattern.id().clone())
            .problem(pattern.problem())
            .solution(pattern.solution())
            .category(pattern.category().clone())
            .context(updates.context.unwrap_or_else(|| pattern.context().to_string()))
            .confidence(updates.confidence.unwrap_or(pattern.confidence()))
            .reward(updates.reward.unwrap_or(pattern.reward()))
            .effectiveness(pattern.effectiveness())
            .reuse_count(pattern.reuse_count())
            .tags(updates.tags.unwrap_or_else(|| pattern.tags().to_vec()));

        if let Some(critique) = updates.critique {
            builder = builder.critique(critique);
        } else {
            builder = builder.critique(pattern.critique());
        }

        if let Some(agent_id) = updates.agent_id {
            builder = builder.agent_id(agent_id);
        } else if let Some(agent_id) = pattern.agent_id() {
            builder = builder.agent_id(agent_id);
        }

        if let Some(session_id) = updates.session_id {
            builder = builder.session_id(session_id);
        } else if let Some(session_id) = pattern.session_id() {
            builder = builder.session_id(session_id);
        }

        if let Some(embedding) = updates.embedding {
            builder = builder.embedding(embedding);
        } else if let Some(embedding) = pattern.embedding() {
            builder = builder.embedding(embedding.to_vec());
        }

        let updated = builder.build();
        self.state.pattern_storage.update_pattern(&updated).await?;

        info!(id = %id, "Pattern updated");
        Ok(())
    }

    /// Get patterns by category.
    ///
    /// # Arguments
    ///
    /// * `category` - Category to filter by
    /// * `limit` - Maximum results
    ///
    /// # Returns
    ///
    /// Vector of pattern info.
    pub async fn by_category(
        &self,
        category: PatternCategory,
        limit: usize,
    ) -> Result<Vec<PatternInfo>> {
        let patterns = self
            .state
            .pattern_storage
            .get_by_category(&category, limit)
            .await?;

        Ok(patterns.into_iter().map(PatternInfo::from).collect())
    }

    /// Get recent patterns.
    ///
    /// # Arguments
    ///
    /// * `limit` - Maximum results
    ///
    /// # Returns
    ///
    /// Vector of recent pattern info.
    pub async fn recent(&self, limit: usize) -> Result<Vec<PatternInfo>> {
        let patterns = self.state.pattern_storage.get_recent(limit).await?;
        Ok(patterns.into_iter().map(PatternInfo::from).collect())
    }

    /// Get top patterns by effectiveness.
    ///
    /// # Arguments
    ///
    /// * `limit` - Maximum results
    ///
    /// # Returns
    ///
    /// Vector of top pattern info.
    pub async fn top(&self, limit: usize) -> Result<Vec<PatternInfo>> {
        let patterns = self.state.pattern_storage.get_top_effective(limit).await?;
        Ok(patterns.into_iter().map(PatternInfo::from).collect())
    }

    /// Increment the usage count of a pattern.
    ///
    /// # Arguments
    ///
    /// * `id` - Pattern ID
    pub async fn increment_usage(&self, id: &str) -> Result<()> {
        let pattern_id = PatternId::from_string(id);
        self.state
            .pattern_storage
            .increment_reuse_count(&pattern_id)
            .await
    }

    /// Get patterns that have embeddings.
    ///
    /// # Returns
    ///
    /// Vector of patterns with embeddings.
    pub async fn with_embeddings(&self) -> Result<Vec<PatternInfo>> {
        let patterns = self.state.pattern_storage.get_all_with_embeddings().await?;
        Ok(patterns.into_iter().map(PatternInfo::from).collect())
    }

    /// Get total pattern count.
    pub async fn count(&self) -> Result<usize> {
        self.state.pattern_storage.count().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pattern_store_options_builder() {
        let options = PatternStoreOptions::new()
            .context("Test context")
            .confidence(0.9)
            .reward(0.8)
            .tags(vec!["tag1".to_string()])
            .critique("Good pattern")
            .agent_id("agent-1")
            .session_id("session-1");

        assert_eq!(options.context, Some("Test context".to_string()));
        assert_eq!(options.confidence, Some(0.9));
        assert_eq!(options.reward, Some(0.8));
        assert_eq!(options.tags, Some(vec!["tag1".to_string()]));
        assert_eq!(options.critique, Some("Good pattern".to_string()));
        assert_eq!(options.agent_id, Some("agent-1".to_string()));
        assert_eq!(options.session_id, Some("session-1".to_string()));
    }

    #[test]
    fn test_pattern_search_options_builder() {
        let options = PatternSearchOptions::new()
            .limit(20)
            .category(PatternCategory::Performance)
            .min_reward(0.7)
            .min_effectiveness(0.6)
            .only_successful();

        assert_eq!(options.limit, Some(20));
        assert_eq!(options.category, Some(PatternCategory::Performance));
        assert_eq!(options.min_reward, Some(0.7));
        assert_eq!(options.min_effectiveness, Some(0.6));
        assert!(options.only_successful);
    }

    #[test]
    fn test_value_clamping() {
        let options = PatternStoreOptions::new()
            .confidence(1.5)
            .reward(-0.5);

        assert_eq!(options.confidence, Some(1.0));
        assert_eq!(options.reward, Some(0.0));
    }
}
