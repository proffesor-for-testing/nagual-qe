//! Knowledge namespace API for storing and retrieving knowledge items.
//!
//! The knowledge API provides a high-level interface for managing knowledge items
//! in the Nagual system. Knowledge items are stored with embeddings for similarity
//! search and can be organized into hierarchical domains.
//!
//! # Example
//!
//! ```rust,ignore
//! // Store knowledge
//! let id = nagual.knowledge.store(
//!     "How to handle database timeouts",
//!     "Implement retry with exponential backoff",
//!     "database.resilience"
//! ).await?;
//!
//! // Search for similar knowledge
//! let results = nagual.knowledge.search("timeout handling").await?;
//!
//! // Get a specific item
//! let item = nagual.knowledge.get(&id).await?;
//!
//! // Delete knowledge
//! nagual.knowledge.delete(&id).await?;
//! ```


use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tracing::{debug, info, instrument};

use super::NagualState;
use crate::error::{NagualError, Result};
use crate::reasoning_bank::pattern::{Pattern, PatternCategory, PatternId};

/// A knowledge item stored in the system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeItem {
    /// Unique identifier
    pub id: String,

    /// The problem or question this knowledge addresses
    pub problem: String,

    /// The solution or answer
    pub solution: String,

    /// Domain/category (e.g., "rust.async", "database.postgres")
    pub domain: String,

    /// Additional context
    pub context: Option<String>,

    /// Confidence score (0.0 - 1.0)
    pub confidence: f32,

    /// Effectiveness score (0.0 - 1.0)
    pub effectiveness: f32,

    /// Number of times this knowledge has been used
    pub usage_count: u32,

    /// Tags for categorization
    pub tags: Vec<String>,

    /// When the knowledge was created
    pub created_at: DateTime<Utc>,

    /// When the knowledge was last updated
    pub updated_at: DateTime<Utc>,
}

impl From<Pattern> for KnowledgeItem {
    fn from(pattern: Pattern) -> Self {
        Self {
            id: pattern.id().to_string(),
            problem: pattern.problem().to_string(),
            solution: pattern.solution().to_string(),
            domain: pattern.category().to_string(),
            context: if pattern.context().is_empty() {
                None
            } else {
                Some(pattern.context().to_string())
            },
            confidence: pattern.confidence(),
            effectiveness: pattern.effectiveness(),
            usage_count: pattern.reuse_count(),
            tags: pattern.tags().to_vec(),
            created_at: pattern.timestamp(),
            updated_at: pattern.updated_at(),
        }
    }
}

/// Result of a knowledge search operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeSearchResult {
    /// The matching knowledge items
    pub items: Vec<KnowledgeItem>,

    /// Search query used
    pub query: String,

    /// Total number of matches (may be more than returned)
    pub total_matches: usize,

    /// Search execution time in milliseconds
    pub search_time_ms: u64,
}

/// Options for knowledge search.
#[derive(Debug, Clone, Default)]
pub struct KnowledgeSearchOptions {
    /// Maximum number of results to return
    pub limit: Option<usize>,

    /// Minimum similarity score (0.0 - 1.0)
    pub min_similarity: Option<f32>,

    /// Filter by domain (prefix match)
    pub domain: Option<String>,

    /// Filter by tags (must have all)
    pub tags: Option<Vec<String>>,

    /// Minimum effectiveness score
    pub min_effectiveness: Option<f32>,
}

impl KnowledgeSearchOptions {
    /// Create new options with defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the maximum number of results.
    pub fn limit(mut self, limit: usize) -> Self {
        self.limit = Some(limit);
        self
    }

    /// Set the minimum similarity score.
    pub fn min_similarity(mut self, score: f32) -> Self {
        self.min_similarity = Some(score.clamp(0.0, 1.0));
        self
    }

    /// Filter by domain.
    pub fn domain(mut self, domain: impl Into<String>) -> Self {
        self.domain = Some(domain.into());
        self
    }

    /// Filter by tags.
    pub fn tags(mut self, tags: Vec<String>) -> Self {
        self.tags = Some(tags);
        self
    }

    /// Set minimum effectiveness score.
    pub fn min_effectiveness(mut self, score: f32) -> Self {
        self.min_effectiveness = Some(score.clamp(0.0, 1.0));
        self
    }
}

/// Options for storing knowledge.
#[derive(Debug, Clone, Default)]
pub struct KnowledgeStoreOptions {
    /// Additional context for the knowledge
    pub context: Option<String>,

    /// Initial confidence score (0.0 - 1.0)
    pub confidence: Option<f32>,

    /// Tags for categorization
    pub tags: Option<Vec<String>>,

    /// Agent ID that is storing this knowledge
    pub agent_id: Option<String>,

    /// Session ID for tracking
    pub session_id: Option<String>,
}

impl KnowledgeStoreOptions {
    /// Create new options with defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set additional context.
    pub fn context(mut self, context: impl Into<String>) -> Self {
        self.context = Some(context.into());
        self
    }

    /// Set initial confidence score.
    pub fn confidence(mut self, confidence: f32) -> Self {
        self.confidence = Some(confidence.clamp(0.0, 1.0));
        self
    }

    /// Set tags.
    pub fn tags(mut self, tags: Vec<String>) -> Self {
        self.tags = Some(tags);
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
}

/// Knowledge storage and retrieval API.
///
/// This API provides methods for storing, searching, retrieving, and deleting
/// knowledge items in the Nagual system.
#[derive(Clone)]
pub struct KnowledgeApi {
    state: NagualState,
}

impl KnowledgeApi {
    /// Create a new KnowledgeApi instance.
    pub(crate) fn new(state: NagualState) -> Self {
        Self { state }
    }

    /// Store a new knowledge item.
    ///
    /// # Arguments
    ///
    /// * `problem` - The problem or question this knowledge addresses
    /// * `solution` - The solution or answer
    /// * `domain` - Domain/category (e.g., "rust.async", "database.postgres")
    ///
    /// # Returns
    ///
    /// The ID of the stored knowledge item.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let id = nagual.knowledge.store(
    ///     "How to implement caching?",
    ///     "Use LRU cache with TTL expiration",
    ///     "performance.caching"
    /// ).await?;
    /// ```
    #[instrument(skip(self, problem, solution, domain))]
    pub async fn store(
        &self,
        problem: impl Into<String>,
        solution: impl Into<String>,
        domain: impl Into<String>,
    ) -> Result<String> {
        self.store_with_options(problem, solution, domain, KnowledgeStoreOptions::default())
            .await
    }

    /// Store a new knowledge item with options.
    ///
    /// # Arguments
    ///
    /// * `problem` - The problem or question this knowledge addresses
    /// * `solution` - The solution or answer
    /// * `domain` - Domain/category
    /// * `options` - Additional options for storing
    ///
    /// # Returns
    ///
    /// The ID of the stored knowledge item.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let id = nagual.knowledge.store_with_options(
    ///     "How to implement caching?",
    ///     "Use LRU cache with TTL expiration",
    ///     "performance.caching",
    ///     KnowledgeStoreOptions::new()
    ///         .context("For web applications")
    ///         .confidence(0.9)
    ///         .tags(vec!["caching", "performance"])
    /// ).await?;
    /// ```
    #[instrument(skip(self, problem, solution, domain, options))]
    pub async fn store_with_options(
        &self,
        problem: impl Into<String>,
        solution: impl Into<String>,
        domain: impl Into<String>,
        options: KnowledgeStoreOptions,
    ) -> Result<String> {
        let problem = problem.into();
        let solution = solution.into();
        let domain_str = domain.into();

        // Build the pattern
        let mut builder = Pattern::builder()
            .problem(&problem)
            .solution(&solution)
            .category(PatternCategory::from(domain_str.as_str()));

        if let Some(context) = options.context {
            builder = builder.context(context);
        }

        if let Some(confidence) = options.confidence {
            builder = builder.confidence(confidence);
        }

        if let Some(tags) = options.tags {
            builder = builder.tags(tags);
        }

        if let Some(agent_id) = options.agent_id {
            builder = builder.agent_id(agent_id);
        }

        if let Some(session_id) = options.session_id {
            builder = builder.session_id(session_id);
        }

        let pattern = builder.build();
        let id = self.state.pattern_storage.store_pattern(&pattern).await?;

        info!(id = %id, domain = %domain_str, "Knowledge stored");

        Ok(id.to_string())
    }

    /// Search for knowledge items.
    ///
    /// # Arguments
    ///
    /// * `query` - Search query string
    ///
    /// # Returns
    ///
    /// A `KnowledgeSearchResult` containing matching items.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let results = nagual.knowledge.search("error handling").await?;
    /// for item in results.items {
    ///     println!("{}: {}", item.problem, item.solution);
    /// }
    /// ```
    #[instrument(skip(self, query))]
    pub async fn search(&self, query: impl Into<String>) -> Result<KnowledgeSearchResult> {
        self.search_with_options(query, KnowledgeSearchOptions::default())
            .await
    }

    /// Search for knowledge items with options.
    ///
    /// # Arguments
    ///
    /// * `query` - Search query string
    /// * `options` - Search options
    ///
    /// # Returns
    ///
    /// A `KnowledgeSearchResult` containing matching items.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let results = nagual.knowledge.search_with_options(
    ///     "error handling",
    ///     KnowledgeSearchOptions::new()
    ///         .limit(5)
    ///         .min_similarity(0.7)
    ///         .domain("rust")
    /// ).await?;
    /// ```
    #[instrument(skip(self, query, options))]
    pub async fn search_with_options(
        &self,
        query: impl Into<String>,
        options: KnowledgeSearchOptions,
    ) -> Result<KnowledgeSearchResult> {
        let query_str = query.into();
        let start = std::time::Instant::now();

        let limit = options.limit.unwrap_or(10);

        // Use FTS5 search for efficient O(log n) text search with BM25 ranking
        // Request more results than needed to allow for post-filtering
        let fts_limit = limit * 5;
        let patterns = self.state.pattern_storage.fts_search(&query_str, fts_limit).await?;

        // Apply additional filters from options
        let mut filtered: Vec<Pattern> = patterns;

        if let Some(ref domain) = options.domain {
            filtered.retain(|p| {
                let cat = p.category().to_string();
                cat == *domain || cat.starts_with(&format!("{}.", domain))
            });
        }

        if let Some(min_eff) = options.min_effectiveness {
            filtered.retain(|p| p.effectiveness() >= min_eff);
        }

        if let Some(ref tags) = options.tags {
            filtered.retain(|p| {
                let pattern_tags = p.tags();
                tags.iter().all(|t| pattern_tags.contains(t))
            });
        }

        // Limit to requested count
        filtered.truncate(limit);

        let total_matches = filtered.len();
        let items: Vec<KnowledgeItem> = filtered.into_iter().map(KnowledgeItem::from).collect();

        let search_time_ms = start.elapsed().as_millis() as u64;

        debug!(
            query = %query_str,
            matches = total_matches,
            time_ms = search_time_ms,
            "Knowledge search completed (FTS5)"
        );

        Ok(KnowledgeSearchResult {
            items,
            query: query_str,
            total_matches,
            search_time_ms,
        })
    }

    /// Get a knowledge item by ID.
    ///
    /// # Arguments
    ///
    /// * `id` - The knowledge item ID
    ///
    /// # Returns
    ///
    /// The knowledge item if found, or None.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// if let Some(item) = nagual.knowledge.get(&id).await? {
    ///     println!("Found: {}", item.problem);
    /// }
    /// ```
    #[instrument(skip(self))]
    pub async fn get(&self, id: &str) -> Result<Option<KnowledgeItem>> {
        let pattern_id = PatternId::from_string(id);
        let pattern = self.state.pattern_storage.get_pattern(&pattern_id).await?;

        Ok(pattern.map(KnowledgeItem::from))
    }

    /// Delete a knowledge item by ID.
    ///
    /// # Arguments
    ///
    /// * `id` - The knowledge item ID
    ///
    /// # Returns
    ///
    /// `Ok(())` if the item was deleted.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// nagual.knowledge.delete(&id).await?;
    /// ```
    #[instrument(skip(self))]
    pub async fn delete(&self, id: &str) -> Result<()> {
        let pattern_id = PatternId::from_string(id);
        self.state.pattern_storage.delete_pattern(&pattern_id).await?;

        info!(id = %id, "Knowledge deleted");
        Ok(())
    }

    /// Update a knowledge item.
    ///
    /// # Arguments
    ///
    /// * `id` - The knowledge item ID
    /// * `problem` - Updated problem (optional)
    /// * `solution` - Updated solution (optional)
    ///
    /// # Returns
    ///
    /// `Ok(())` if the item was updated.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// nagual.knowledge.update(
    ///     &id,
    ///     Some("Updated problem"),
    ///     Some("Updated solution")
    /// ).await?;
    /// ```
    #[instrument(skip(self))]
    pub async fn update(
        &self,
        id: &str,
        problem: Option<&str>,
        solution: Option<&str>,
    ) -> Result<()> {
        let pattern_id = PatternId::from_string(id);
        let pattern = self
            .state
            .pattern_storage
            .get_pattern(&pattern_id)
            .await?
            .ok_or_else(|| NagualError::internal(format!("Knowledge not found: {}", id)))?;

        // Build updated pattern
        let mut builder = Pattern::builder()
            .id(pattern.id().clone())
            .problem(problem.unwrap_or(pattern.problem()))
            .solution(solution.unwrap_or(pattern.solution()))
            .category(pattern.category().clone())
            .context(pattern.context())
            .confidence(pattern.confidence())
            .effectiveness(pattern.effectiveness())
            .reward(pattern.reward())
            .reuse_count(pattern.reuse_count())
            .tags(pattern.tags().to_vec());

        if let Some(agent_id) = pattern.agent_id() {
            builder = builder.agent_id(agent_id);
        }

        if let Some(session_id) = pattern.session_id() {
            builder = builder.session_id(session_id);
        }

        let updated = builder.build();
        self.state.pattern_storage.update_pattern(&updated).await?;

        info!(id = %id, "Knowledge updated");
        Ok(())
    }

    /// Get the total count of knowledge items.
    ///
    /// # Returns
    ///
    /// The number of knowledge items stored.
    pub async fn count(&self) -> Result<usize> {
        self.state.pattern_storage.count().await
    }

    /// Get recent knowledge items.
    ///
    /// # Arguments
    ///
    /// * `limit` - Maximum number of items to return
    ///
    /// # Returns
    ///
    /// A vector of recent knowledge items.
    pub async fn recent(&self, limit: usize) -> Result<Vec<KnowledgeItem>> {
        let patterns = self.state.pattern_storage.get_recent(limit).await?;
        Ok(patterns.into_iter().map(KnowledgeItem::from).collect())
    }

    /// Get top knowledge items by effectiveness.
    ///
    /// # Arguments
    ///
    /// * `limit` - Maximum number of items to return
    ///
    /// # Returns
    ///
    /// A vector of top knowledge items sorted by effectiveness.
    pub async fn top(&self, limit: usize) -> Result<Vec<KnowledgeItem>> {
        let patterns = self.state.pattern_storage.get_top_effective(limit).await?;
        Ok(patterns.into_iter().map(KnowledgeItem::from).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_knowledge_store_options_builder() {
        let options = KnowledgeStoreOptions::new()
            .context("Test context")
            .confidence(0.9)
            .tags(vec!["tag1".to_string(), "tag2".to_string()])
            .agent_id("agent-123")
            .session_id("session-456");

        assert_eq!(options.context, Some("Test context".to_string()));
        assert_eq!(options.confidence, Some(0.9));
        assert_eq!(options.tags, Some(vec!["tag1".to_string(), "tag2".to_string()]));
        assert_eq!(options.agent_id, Some("agent-123".to_string()));
        assert_eq!(options.session_id, Some("session-456".to_string()));
    }

    #[test]
    fn test_knowledge_search_options_builder() {
        let options = KnowledgeSearchOptions::new()
            .limit(20)
            .min_similarity(0.8)
            .domain("rust.async")
            .min_effectiveness(0.7);

        assert_eq!(options.limit, Some(20));
        assert_eq!(options.min_similarity, Some(0.8));
        assert_eq!(options.domain, Some("rust.async".to_string()));
        assert_eq!(options.min_effectiveness, Some(0.7));
    }

    #[test]
    fn test_confidence_clamping() {
        let options = KnowledgeStoreOptions::new().confidence(1.5);
        assert_eq!(options.confidence, Some(1.0));

        let options = KnowledgeStoreOptions::new().confidence(-0.5);
        assert_eq!(options.confidence, Some(0.0));
    }
}
