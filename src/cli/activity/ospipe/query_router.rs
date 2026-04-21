//! Query router for multi-modal search across activity data.
//!
//! Routes queries to the appropriate search backend based on query characteristics:
//! - Semantic: Embedding-based similarity search
//! - Keyword: Full-text search with FTS5
//! - Temporal: Time-range queries
//! - Graph: Knowledge graph traversal
//! - Hybrid: Combination of multiple modes

use std::sync::Arc;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::reasoning_bank::storage::PatternStorage;

/// Search mode for query routing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum QueryMode {
    /// Semantic similarity search using embeddings.
    Semantic,
    /// Full-text keyword search using FTS5.
    Keyword,
    /// Time-based search within a specific range.
    Temporal,
    /// Knowledge graph traversal and link-based search.
    Graph,
    /// Hybrid search combining multiple modes.
    #[default]
    Hybrid,
}

impl QueryMode {
    /// Parse a query mode from a string.
    pub fn from_str(s: &str) -> Option<Self> {
        match s.trim().to_lowercase().as_str() {
            "semantic" | "vector" | "embedding" => Some(QueryMode::Semantic),
            "keyword" | "fts" | "text" => Some(QueryMode::Keyword),
            "temporal" | "time" | "date" => Some(QueryMode::Temporal),
            "graph" | "link" | "relation" => Some(QueryMode::Graph),
            "hybrid" | "mixed" | "auto" => Some(QueryMode::Hybrid),
            _ => None,
        }
    }

    /// Get a description of this query mode.
    pub fn description(&self) -> &'static str {
        match self {
            QueryMode::Semantic => "Embedding-based similarity search",
            QueryMode::Keyword => "Full-text keyword search (FTS5)",
            QueryMode::Temporal => "Time-range based search",
            QueryMode::Graph => "Knowledge graph traversal",
            QueryMode::Hybrid => "Combined multi-modal search",
        }
    }
}

impl std::fmt::Display for QueryMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            QueryMode::Semantic => write!(f, "semantic"),
            QueryMode::Keyword => write!(f, "keyword"),
            QueryMode::Temporal => write!(f, "temporal"),
            QueryMode::Graph => write!(f, "graph"),
            QueryMode::Hybrid => write!(f, "hybrid"),
        }
    }
}

/// A search result from the query router.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    /// Pattern ID.
    pub id: String,
    /// Problem/title of the pattern.
    pub problem: String,
    /// Solution/content.
    pub solution: String,
    /// Relevance score (0.0-1.0).
    pub score: f32,
    /// Source mode that produced this result.
    pub source_mode: QueryMode,
    /// Timestamp of the pattern.
    pub timestamp: Option<DateTime<Utc>>,
    /// Tags associated with the pattern.
    pub tags: Vec<String>,
    /// Domain/category.
    pub domain: Option<String>,
}

/// Query parameters for routing.
#[derive(Debug, Clone)]
pub struct QueryParams {
    /// The search query text.
    pub query: String,
    /// Preferred search mode.
    pub mode: QueryMode,
    /// Maximum number of results.
    pub limit: usize,
    /// Minimum relevance score.
    pub min_score: f32,
    /// Time range start (for temporal queries).
    pub time_start: Option<DateTime<Utc>>,
    /// Time range end (for temporal queries).
    pub time_end: Option<DateTime<Utc>>,
    /// Filter by domain.
    pub domain: Option<String>,
    /// Filter by tags.
    pub tags: Vec<String>,
    /// Query embedding (for semantic search, if pre-computed).
    pub embedding: Option<Vec<f32>>,
}

impl QueryParams {
    /// Create a new query with default parameters.
    pub fn new(query: impl Into<String>) -> Self {
        Self {
            query: query.into(),
            mode: QueryMode::Hybrid,
            limit: 10,
            min_score: 0.0,
            time_start: None,
            time_end: None,
            domain: None,
            tags: Vec::new(),
            embedding: None,
        }
    }

    /// Builder-style method to set the mode.
    pub fn with_mode(mut self, mode: QueryMode) -> Self {
        self.mode = mode;
        self
    }

    /// Builder-style method to set the limit.
    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = limit;
        self
    }

    /// Builder-style method to set minimum score.
    pub fn with_min_score(mut self, min_score: f32) -> Self {
        self.min_score = min_score;
        self
    }

    /// Builder-style method to set time range.
    pub fn with_time_range(
        mut self,
        start: Option<DateTime<Utc>>,
        end: Option<DateTime<Utc>>,
    ) -> Self {
        self.time_start = start;
        self.time_end = end;
        self
    }

    /// Builder-style method to set domain filter.
    pub fn with_domain(mut self, domain: impl Into<String>) -> Self {
        self.domain = Some(domain.into());
        self
    }

    /// Builder-style method to set tags filter.
    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.tags = tags;
        self
    }

    /// Builder-style method to set pre-computed embedding.
    pub fn with_embedding(mut self, embedding: Vec<f32>) -> Self {
        self.embedding = Some(embedding);
        self
    }
}

/// Statistics for query routing.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RouterStats {
    /// Total queries routed.
    pub total_queries: u64,
    /// Queries by mode.
    pub queries_by_mode: std::collections::HashMap<String, u64>,
    /// Average results per query.
    pub avg_results: f64,
    /// Average latency in milliseconds.
    pub avg_latency_ms: f64,
}

/// Query router for multi-modal search.
pub struct QueryRouter {
    /// Storage backend for pattern retrieval.
    storage: Arc<PatternStorage>,
    /// Default mode for queries.
    default_mode: QueryMode,
    /// Statistics.
    stats: std::sync::Mutex<RouterStats>,
}

impl QueryRouter {
    /// Create a new query router with the given storage backend.
    pub fn new(storage: Arc<PatternStorage>) -> Self {
        Self {
            storage,
            default_mode: QueryMode::Hybrid,
            stats: std::sync::Mutex::new(RouterStats::default()),
        }
    }

    /// Set the default query mode.
    pub fn set_default_mode(&mut self, mode: QueryMode) {
        self.default_mode = mode;
    }

    /// Route a query to the appropriate search backend.
    pub async fn route(&self, params: &QueryParams) -> crate::error::Result<Vec<SearchResult>> {
        let start = std::time::Instant::now();
        let mode = params.mode;

        let results = match mode {
            QueryMode::Semantic => self.semantic_search(params).await?,
            QueryMode::Keyword => self.keyword_search(params).await?,
            QueryMode::Temporal => self.temporal_search(params).await?,
            QueryMode::Graph => self.graph_search(params).await?,
            QueryMode::Hybrid => self.hybrid_search(params).await?,
        };

        // Update stats
        if let Ok(mut stats) = self.stats.lock() {
            stats.total_queries += 1;
            *stats
                .queries_by_mode
                .entry(mode.to_string())
                .or_insert(0) += 1;

            let n = stats.total_queries as f64;
            stats.avg_results = stats.avg_results * (n - 1.0) / n + results.len() as f64 / n;
            stats.avg_latency_ms =
                stats.avg_latency_ms * (n - 1.0) / n + start.elapsed().as_millis() as f64 / n;
        }

        Ok(results)
    }

    /// Semantic search using embeddings.
    async fn semantic_search(
        &self,
        params: &QueryParams,
    ) -> crate::error::Result<Vec<SearchResult>> {
        // Use FTS search as the base - semantic would require HNSW index
        // For now, fall back to FTS
        let patterns = self
            .storage
            .fts_search(&params.query, params.limit)
            .await?;

        Ok(patterns
            .into_iter()
            .enumerate()
            .filter(|(idx, _)| {
                // Simple score based on position
                let score = 1.0 - (*idx as f32 / params.limit as f32);
                score >= params.min_score
            })
            .map(|(idx, p)| SearchResult {
                id: p.id().to_string(),
                problem: p.problem().to_string(),
                solution: p.solution().to_string(),
                score: 1.0 - (idx as f32 / params.limit as f32),
                source_mode: QueryMode::Semantic,
                timestamp: Some(p.updated_at()),
                tags: p.tags().to_vec(),
                domain: Some(p.category().to_string()),
            })
            .collect())
    }

    /// Keyword search using FTS5.
    async fn keyword_search(
        &self,
        params: &QueryParams,
    ) -> crate::error::Result<Vec<SearchResult>> {
        let patterns = self
            .storage
            .fts_search(&params.query, params.limit)
            .await?;

        Ok(patterns
            .into_iter()
            .enumerate()
            .filter(|(idx, _)| {
                let score = 1.0 - (*idx as f32 / params.limit as f32);
                score >= params.min_score
            })
            .map(|(idx, p)| SearchResult {
                id: p.id().to_string(),
                problem: p.problem().to_string(),
                solution: p.solution().to_string(),
                score: 1.0 - (idx as f32 / params.limit as f32),
                source_mode: QueryMode::Keyword,
                timestamp: Some(p.updated_at()),
                tags: p.tags().to_vec(),
                domain: Some(p.category().to_string()),
            })
            .collect())
    }

    /// Temporal search within a time range.
    async fn temporal_search(
        &self,
        params: &QueryParams,
    ) -> crate::error::Result<Vec<SearchResult>> {
        // For temporal search, we filter by time range
        // First get patterns, then filter by time
        let patterns = self
            .storage
            .fts_search(&params.query, params.limit * 2)
            .await?;

        let filtered: Vec<_> = patterns
            .into_iter()
            .filter(|p| {
                let timestamp = p.updated_at();
                let after_start = params.time_start.map_or(true, |s| timestamp >= s);
                let before_end = params.time_end.map_or(true, |e| timestamp <= e);
                after_start && before_end
            })
            .take(params.limit)
            .collect();

        Ok(filtered
            .into_iter()
            .enumerate()
            .map(|(idx, p)| SearchResult {
                id: p.id().to_string(),
                problem: p.problem().to_string(),
                solution: p.solution().to_string(),
                score: 1.0 - (idx as f32 / params.limit as f32),
                source_mode: QueryMode::Temporal,
                timestamp: Some(p.updated_at()),
                tags: p.tags().to_vec(),
                domain: Some(p.category().to_string()),
            })
            .collect())
    }

    /// Graph-based search using knowledge graph links.
    async fn graph_search(&self, params: &QueryParams) -> crate::error::Result<Vec<SearchResult>> {
        // Graph search would use ProfDAG relationships
        // For now, fall back to keyword search with category filtering
        let patterns = if let Some(ref domain) = params.domain {
            self.storage
                .get_by_category(
                    &crate::reasoning_bank::pattern::PatternCategory::Custom(domain.clone()),
                    params.limit,
                )
                .await?
        } else {
            self.storage
                .fts_search(&params.query, params.limit)
                .await?
        };

        Ok(patterns
            .into_iter()
            .enumerate()
            .map(|(idx, p)| SearchResult {
                id: p.id().to_string(),
                problem: p.problem().to_string(),
                solution: p.solution().to_string(),
                score: 1.0 - (idx as f32 / params.limit as f32),
                source_mode: QueryMode::Graph,
                timestamp: Some(p.updated_at()),
                tags: p.tags().to_vec(),
                domain: Some(p.category().to_string()),
            })
            .collect())
    }

    /// Hybrid search combining multiple modes.
    async fn hybrid_search(&self, params: &QueryParams) -> crate::error::Result<Vec<SearchResult>> {
        // Run keyword and semantic searches in parallel, then merge results
        let half_limit = (params.limit + 1) / 2;

        // Get results from both modes
        let keyword_results = self
            .keyword_search(&QueryParams {
                limit: half_limit,
                ..params.clone()
            })
            .await?;

        let semantic_results = self
            .semantic_search(&QueryParams {
                limit: half_limit,
                ..params.clone()
            })
            .await?;

        // Merge and deduplicate by ID
        let mut seen = std::collections::HashSet::new();
        let mut merged = Vec::with_capacity(params.limit);

        // Interleave results, preferring semantic slightly
        let mut sem_iter = semantic_results.into_iter().peekable();
        let mut kw_iter = keyword_results.into_iter().peekable();

        while merged.len() < params.limit && (sem_iter.peek().is_some() || kw_iter.peek().is_some())
        {
            // Take from semantic
            if let Some(r) = sem_iter.next() {
                if seen.insert(r.id.clone()) {
                    merged.push(SearchResult {
                        source_mode: QueryMode::Hybrid,
                        ..r
                    });
                }
            }

            // Take from keyword
            if merged.len() < params.limit {
                if let Some(r) = kw_iter.next() {
                    if seen.insert(r.id.clone()) {
                        merged.push(SearchResult {
                            source_mode: QueryMode::Hybrid,
                            ..r
                        });
                    }
                }
            }
        }

        // Re-score based on position
        for (idx, result) in merged.iter_mut().enumerate() {
            result.score = 1.0 - (idx as f32 / params.limit as f32);
        }

        Ok(merged)
    }

    /// Automatically detect the best query mode based on query characteristics.
    pub fn detect_mode(&self, query: &str) -> QueryMode {
        let query_lower = query.to_lowercase();

        // Check for temporal keywords
        if query_lower.contains("yesterday")
            || query_lower.contains("today")
            || query_lower.contains("last week")
            || query_lower.contains("this morning")
            || query.contains("202") // Year pattern
        {
            return QueryMode::Temporal;
        }

        // Check for relationship keywords
        if query_lower.contains("related to")
            || query_lower.contains("similar to")
            || query_lower.contains("connected")
            || query_lower.contains("link")
        {
            return QueryMode::Graph;
        }

        // Check for exact match indicators
        if query.contains('"') || query.contains("exact:") {
            return QueryMode::Keyword;
        }

        // Default to hybrid for natural language queries
        QueryMode::Hybrid
    }

    /// Get router statistics.
    pub fn stats(&self) -> RouterStats {
        self.stats.lock().unwrap_or_else(|p| p.into_inner()).clone()
    }

    /// Reset router statistics.
    pub fn reset_stats(&self) {
        if let Ok(mut stats) = self.stats.lock() {
            *stats = RouterStats::default();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_query_mode_from_str() {
        assert_eq!(QueryMode::from_str("semantic"), Some(QueryMode::Semantic));
        assert_eq!(QueryMode::from_str("vector"), Some(QueryMode::Semantic));
        assert_eq!(QueryMode::from_str("keyword"), Some(QueryMode::Keyword));
        assert_eq!(QueryMode::from_str("fts"), Some(QueryMode::Keyword));
        assert_eq!(QueryMode::from_str("temporal"), Some(QueryMode::Temporal));
        assert_eq!(QueryMode::from_str("graph"), Some(QueryMode::Graph));
        assert_eq!(QueryMode::from_str("hybrid"), Some(QueryMode::Hybrid));
        assert_eq!(QueryMode::from_str("auto"), Some(QueryMode::Hybrid));
        assert_eq!(QueryMode::from_str("invalid"), None);
    }

    #[test]
    fn test_query_params_builder() {
        let params = QueryParams::new("test query")
            .with_mode(QueryMode::Semantic)
            .with_limit(20)
            .with_min_score(0.5)
            .with_domain("coding");

        assert_eq!(params.query, "test query");
        assert_eq!(params.mode, QueryMode::Semantic);
        assert_eq!(params.limit, 20);
        assert_eq!(params.min_score, 0.5);
        assert_eq!(params.domain, Some("coding".to_string()));
    }

    #[test]
    fn test_search_result_serialization() {
        let result = SearchResult {
            id: "test-id".to_string(),
            problem: "Test problem".to_string(),
            solution: "Test solution".to_string(),
            score: 0.95,
            source_mode: QueryMode::Hybrid,
            timestamp: Some(Utc::now()),
            tags: vec!["tag1".to_string(), "tag2".to_string()],
            domain: Some("coding".to_string()),
        };

        let json = serde_json::to_string(&result).unwrap();
        let deserialized: SearchResult = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.id, "test-id");
        assert_eq!(deserialized.score, 0.95);
        assert_eq!(deserialized.source_mode, QueryMode::Hybrid);
    }
}
