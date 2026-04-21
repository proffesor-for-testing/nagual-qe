//! Main OSpipe pipeline implementation.
//!
//! Orchestrates the flow of activity data from Screenpipe through:
//! 1. PII Gate (redact/reject)
//! 2. Sliding Window Dedup
//! 3. Embedding Generation
//! 4. Storage

use std::path::PathBuf;
use std::sync::Arc;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::error::Result;
#[cfg(feature = "onnx-embed")]
use crate::ml::{CacheConfig, CachedEmbedder, Embedder, EmbedderConfig};
use crate::reasoning_bank::pattern::{Pattern, PatternCategory, PatternMetadata};
use crate::reasoning_bank::storage::PatternStorage;

use super::config::{EmbeddingDim, OSpipeConfig};
use super::dedup::{DedupResult, SlidingWindowDedup};
use super::pii_gate::{PiiGate, PiiGateResult, PiiPolicy};
use super::query_router::{QueryParams, QueryRouter, SearchResult};

/// Result of a single item's ingestion through the pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngestItemResult {
    /// Original content (before processing).
    pub original_length: usize,
    /// Processed content length (after PII redaction).
    pub processed_length: usize,
    /// Whether content was redacted.
    pub was_redacted: bool,
    /// Whether content was rejected.
    pub was_rejected: bool,
    /// Whether content was a duplicate.
    pub was_duplicate: bool,
    /// Similarity score if checked for duplicates.
    pub dedup_similarity: Option<f32>,
    /// Pattern ID if stored successfully.
    pub pattern_id: Option<String>,
    /// Error message if processing failed.
    pub error: Option<String>,
}

impl IngestItemResult {
    /// Create a result for rejected content.
    pub fn rejected(original_length: usize, reason: &str) -> Self {
        Self {
            original_length,
            processed_length: 0,
            was_redacted: false,
            was_rejected: true,
            was_duplicate: false,
            dedup_similarity: None,
            pattern_id: None,
            error: Some(reason.to_string()),
        }
    }

    /// Create a result for duplicate content.
    pub fn duplicate(original_length: usize, similarity: f32) -> Self {
        Self {
            original_length,
            processed_length: original_length,
            was_redacted: false,
            was_rejected: false,
            was_duplicate: true,
            dedup_similarity: Some(similarity),
            pattern_id: None,
            error: None,
        }
    }

    /// Create a result for successfully stored content.
    pub fn success(
        original_length: usize,
        processed_length: usize,
        was_redacted: bool,
        pattern_id: String,
    ) -> Self {
        Self {
            original_length,
            processed_length,
            was_redacted,
            was_rejected: false,
            was_duplicate: false,
            dedup_similarity: None,
            pattern_id: Some(pattern_id),
            error: None,
        }
    }

    /// Create a result for processing error.
    pub fn error(original_length: usize, error: &str) -> Self {
        Self {
            original_length,
            processed_length: 0,
            was_redacted: false,
            was_rejected: false,
            was_duplicate: false,
            dedup_similarity: None,
            pattern_id: None,
            error: Some(error.to_string()),
        }
    }
}

/// Summary of a batch ingestion operation.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct IngestResult {
    /// Total items processed.
    pub total_processed: usize,
    /// Items successfully stored.
    pub stored_count: usize,
    /// Items rejected due to PII.
    pub rejected_count: usize,
    /// Items redacted.
    pub redacted_count: usize,
    /// Items skipped as duplicates.
    pub duplicate_count: usize,
    /// Items that failed processing.
    pub error_count: usize,
    /// Embeddings generated.
    pub embeddings_generated: usize,
    /// Time range start.
    pub time_range_start: Option<DateTime<Utc>>,
    /// Time range end.
    pub time_range_end: Option<DateTime<Utc>>,
    /// Pattern IDs created.
    pub pattern_ids: Vec<String>,
}

impl IngestResult {
    /// Create a new empty result.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add an item result to the summary.
    pub fn add(&mut self, item: &IngestItemResult) {
        self.total_processed += 1;

        if item.was_rejected {
            self.rejected_count += 1;
        } else if item.was_duplicate {
            self.duplicate_count += 1;
        } else if item.error.is_some() {
            self.error_count += 1;
        } else {
            self.stored_count += 1;
            if item.was_redacted {
                self.redacted_count += 1;
            }
            if let Some(ref id) = item.pattern_id {
                self.pattern_ids.push(id.clone());
            }
        }
    }
}

/// OSpipe client - high-level interface for Screenpipe data access.
pub struct OSpipeClient {
    /// Screenpipe HTTP client.
    screenpipe_url: String,
    /// HTTP client for Screenpipe API.
    http_client: reqwest::Client,
}

impl OSpipeClient {
    /// Create a new OSpipe client.
    pub fn new(screenpipe_url: &str) -> Self {
        Self {
            screenpipe_url: screenpipe_url.to_string(),
            http_client: reqwest::Client::new(),
        }
    }

    /// Execute a raw SQL query against Screenpipe's database.
    pub async fn raw_sql(&self, sql: &str) -> Result<Vec<serde_json::Value>> {
        let url = format!("{}/raw_sql", self.screenpipe_url);

        let response = self
            .http_client
            .post(&url)
            .json(&serde_json::json!({ "query": sql }))
            .send()
            .await
            .map_err(|e| crate::error::NagualError::Http(e.to_string()))?;

        if !response.status().is_success() {
            return Err(crate::error::NagualError::Http(format!(
                "Screenpipe raw_sql failed: {}",
                response.status()
            )));
        }

        let body: serde_json::Value = response
            .json()
            .await
            .map_err(|e| crate::error::NagualError::Http(e.to_string()))?;

        // Extract rows from response
        if let Some(rows) = body.as_array() {
            Ok(rows.clone())
        } else if let Some(rows) = body.get("data").and_then(|d| d.as_array()) {
            Ok(rows.clone())
        } else {
            Ok(vec![])
        }
    }

    /// Check if Screenpipe is healthy.
    pub async fn health_check(&self) -> bool {
        let url = format!("{}/health", self.screenpipe_url);
        self.http_client
            .get(&url)
            .timeout(std::time::Duration::from_secs(5))
            .send()
            .await
            .map(|r| r.status().is_success())
            .unwrap_or(false)
    }
}

/// OSpipe pipeline - orchestrates the full ingestion flow.
pub struct OSpipePipeline {
    /// Configuration.
    config: OSpipeConfig,
    /// PII safety gate.
    pii_gate: PiiGate,
    /// Deduplication engine.
    dedup: SlidingWindowDedup,
    /// Embedder (optional, loaded on demand). Only available with onnx-embed feature.
    #[cfg(feature = "onnx-embed")]
    embedder: Option<Arc<CachedEmbedder>>,
    /// Pattern storage.
    storage: Arc<PatternStorage>,
    /// Query router.
    query_router: QueryRouter,
}

impl OSpipePipeline {
    /// Create a new OSpipe pipeline.
    pub fn new(storage: Arc<PatternStorage>, config: OSpipeConfig) -> Self {
        let pii_policy = PiiPolicy::from_str(&config.pii_policy).unwrap_or(PiiPolicy::Redact);
        let pii_gate = PiiGate::new(pii_policy);

        let dedup = SlidingWindowDedup::new(config.dedup_window, config.dedup_threshold);

        let query_router = QueryRouter::new(Arc::clone(&storage));

        Self {
            config,
            pii_gate,
            dedup,
            #[cfg(feature = "onnx-embed")]
            embedder: None,
            storage,
            query_router,
        }
    }

    /// Initialize the embedder (stub when onnx-embed is disabled).
    #[cfg(not(feature = "onnx-embed"))]
    pub fn init_embedder(&mut self) -> Result<()> {
        Err(crate::error::NagualError::Config {
            message: "ONNX embedding not available (onnx-embed feature disabled)".to_string(),
        })
    }

    /// Initialize the embedder (lazy initialization).
    /// Tries multiple model paths to find the ONNX model.
    #[cfg(feature = "onnx-embed")]
    pub fn init_embedder(&mut self) -> Result<()> {
        if self.embedder.is_some() {
            return Ok(());
        }

        // If explicit paths are configured, use them
        if let (Some(model_path), Some(tokenizer_path)) =
            (self.config.model_path.clone(), self.config.tokenizer_path.clone())
        {
            if model_path.exists() && tokenizer_path.exists() {
                return self.load_embedder_from_paths(&model_path, &tokenizer_path);
            }
        }

        // Try multiple fallback paths
        let home = std::env::var("HOME").unwrap_or_default();
        let home_model = format!("{}/.nagual/models/all-MiniLM-L6-v2.onnx", home);
        let home_tokenizer = format!("{}/.nagual/models/tokenizer.json", home);

        let paths: &[(&str, &str)] = &[
            ("models/all-MiniLM-L6-v2.onnx", "models/tokenizer.json"),
            ("../models/all-MiniLM-L6-v2.onnx", "../models/tokenizer.json"),
            (&home_model, &home_tokenizer),
        ];

        for (model_path, tokenizer_path) in paths {
            let model = PathBuf::from(model_path);
            let tokenizer = PathBuf::from(tokenizer_path);

            if model.exists() && tokenizer.exists() {
                tracing::info!(model = %model_path, "Found ONNX model");
                return self.load_embedder_from_paths(&model, &tokenizer);
            }
        }

        Err(crate::error::NagualError::Config {
            message: "ONNX model not found. Tried: ./models/, ../models/, ~/.nagual/models/. Download via scripts/download-models.sh".to_string(),
        })
    }

    /// Load embedder from specific paths.
    #[cfg(feature = "onnx-embed")]
    fn load_embedder_from_paths(
        &mut self,
        model_path: &PathBuf,
        tokenizer_path: &PathBuf,
    ) -> Result<()> {
        let embedder_config = match self.config.embedding_dim {
            EmbeddingDim::Dim128 => EmbedderConfig::dim_128(
                model_path.to_string_lossy().to_string(),
                tokenizer_path.to_string_lossy().to_string(),
            ),
            EmbeddingDim::Dim384 => EmbedderConfig::dim_384(
                model_path.to_string_lossy().to_string(),
                tokenizer_path.to_string_lossy().to_string(),
            ),
        };

        // Wrap in catch_unwind because ort can panic if ORT_DYLIB_PATH isn't set
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            Embedder::new(&embedder_config)
        }));

        match result {
            Ok(Ok(embedder)) => {
                let cached = CachedEmbedder::new(embedder, CacheConfig::with_size(10_000));
                self.embedder = Some(Arc::new(cached));
                Ok(())
            }
            Ok(Err(e)) => Err(crate::error::NagualError::Internal {
                message: format!("Failed to load embedder: {}", e),
            }),
            Err(_) => Err(crate::error::NagualError::Internal {
                message: "Embedder loading panicked - check ORT_DYLIB_PATH environment variable"
                    .to_string(),
            }),
        }
    }

    /// Process content through the PII gate.
    pub fn process_pii(&self, content: &str) -> PiiGateResult {
        if !self.config.pii_enabled {
            return PiiGateResult::Allowed {
                content: content.to_string(),
                had_pii: false,
                classification: crate::security::pii::PiiClassification::None,
            };
        }

        self.pii_gate.process(content)
    }

    /// Check for duplicate content.
    pub fn check_duplicate(
        &mut self,
        embedding: &[f32],
        timestamp: DateTime<Utc>,
    ) -> DedupResult {
        if !self.config.dedup_enabled {
            return DedupResult::first_entry();
        }

        self.dedup.is_duplicate(embedding, timestamp)
    }

    /// Generate an embedding for text.
    pub fn generate_embedding(&self, text: &str) -> Result<Option<Vec<f32>>> {
        if !self.config.generate_embeddings {
            return Ok(None);
        }

        #[cfg(feature = "onnx-embed")]
        {
            let embedder = self.embedder.as_ref().ok_or_else(|| {
                crate::error::NagualError::Config {
                    message: "Embedder not initialized".to_string(),
                }
            })?;

            let result = embedder.embed(text).map_err(|e| {
                crate::error::NagualError::Internal {
                    message: e.to_string(),
                }
            })?;
            return Ok(Some(result.embedding));
        }

        #[cfg(not(feature = "onnx-embed"))]
        {
            let _ = text;
            Ok(None)
        }
    }

    /// Process a single content item through the full pipeline.
    pub async fn process_item(
        &mut self,
        content: &str,
        app_name: &str,
        timestamp: DateTime<Utc>,
        metadata: Option<PatternMetadata>,
    ) -> IngestItemResult {
        let original_length = content.len();

        // Step 1: PII Gate
        let pii_result = self.process_pii(content);

        let processed_content = match &pii_result {
            PiiGateResult::Rejected { reason, .. } => {
                return IngestItemResult::rejected(original_length, reason);
            }
            PiiGateResult::Allowed { content, .. } => content.clone(),
            PiiGateResult::Redacted { content, .. } => content.clone(),
            PiiGateResult::Warned { content, .. } => content.clone(),
        };

        let was_redacted = matches!(pii_result, PiiGateResult::Redacted { .. });
        let processed_length = processed_content.len();

        // Step 2: Generate embedding (if enabled)
        #[cfg(feature = "onnx-embed")]
        let has_embedder = self.embedder.is_some();
        #[cfg(not(feature = "onnx-embed"))]
        let has_embedder = false;
        let embedding = if self.config.generate_embeddings && has_embedder {
            match self.generate_embedding(&processed_content) {
                Ok(Some(emb)) => Some(emb),
                Ok(None) => None,
                Err(e) => {
                    tracing::warn!(error = %e, "Embedding generation failed");
                    None
                }
            }
        } else {
            None
        };

        // Step 3: Deduplication check
        if self.config.dedup_enabled {
            if let Some(ref emb) = embedding {
                let dedup_result = self.dedup.check_and_add(emb.clone(), timestamp);
                if dedup_result.is_duplicate {
                    return IngestItemResult::duplicate(original_length, dedup_result.max_similarity);
                }
            }
        }

        // Step 4: Create and store pattern
        let domain = categorize_app(app_name);
        let problem = format!("Activity from {}: {}", app_name, truncate(&processed_content, 100));

        let mut pattern_metadata = metadata.unwrap_or_else(PatternMetadata::new);
        pattern_metadata = pattern_metadata
            .with_source("ospipe")
            .with_extra("app_name", serde_json::json!(app_name))
            .with_extra("was_redacted", serde_json::json!(was_redacted));

        let mut builder = Pattern::builder()
            .problem(&problem)
            .solution(&processed_content)
            .category(PatternCategory::Custom(format!("activity.{}", domain)))
            .context(format!("OSpipe activity capture from {}", app_name))
            .effectiveness(0.5)
            .confidence(0.5)
            .tags(vec![
                app_name.to_string(),
                domain.to_string(),
                "ospipe".to_string(),
            ])
            .metadata(pattern_metadata);

        if let Some(emb) = embedding {
            builder = builder.embedding(emb);
        }

        let pattern = builder.build();
        let pattern_id = pattern.id().to_string();

        let has_embedding = pattern.embedding().is_some();

        match self.storage.store_pattern(&pattern).await {
            Ok(_) => IngestItemResult::success(
                original_length,
                processed_length,
                was_redacted,
                pattern_id,
            ),
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    app = %app_name,
                    has_embedding = has_embedding,
                    "Failed to store pattern"
                );
                IngestItemResult::error(original_length, &e.to_string())
            }
        }
    }

    /// Search for patterns using the query router.
    pub async fn search(&self, params: &QueryParams) -> Result<Vec<SearchResult>> {
        self.query_router.route(params).await
    }

    /// Get the PII gate.
    pub fn pii_gate(&self) -> &PiiGate {
        &self.pii_gate
    }

    /// Get mutable reference to dedup engine.
    pub fn dedup_mut(&mut self) -> &mut SlidingWindowDedup {
        &mut self.dedup
    }

    /// Get the configuration.
    pub fn config(&self) -> &OSpipeConfig {
        &self.config
    }

    /// Get dedup statistics.
    pub fn dedup_stats(&self) -> super::dedup::DedupStats {
        self.dedup.stats()
    }
}

// Helper functions (copied from helpers.rs to avoid circular dependency)

fn categorize_app(app_name: &str) -> &'static str {
    let s = app_name.to_lowercase();
    if s.contains("code") || s.contains("vim") || s.contains("neovim")
        || s.contains("intellij") || s.contains("xcode") || s.contains("cursor")
    {
        "coding"
    } else if s.contains("chrome") || s.contains("firefox") || s.contains("safari")
        || s.contains("arc") || s.contains("brave") || s.contains("edge")
    {
        "browsing"
    } else if s.contains("slack") || s.contains("discord") || s.contains("teams")
        || s.contains("messages") || s.contains("telegram")
    {
        "communication"
    } else if s.contains("zoom") || s.contains("meet") || s.contains("facetime") {
        "meetings"
    } else if s.contains("terminal") || s.contains("iterm") || s.contains("warp")
        || s.contains("alacritty") || s.contains("kitty")
    {
        "terminal"
    } else {
        "other"
    }
}

fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        // Find a valid UTF-8 boundary
        let mut end = max_len.saturating_sub(3);
        while end > 0 && !s.is_char_boundary(end) {
            end -= 1;
        }
        if end == 0 {
            "...".to_string()
        } else {
            format!("{}...", &s[..end])
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ingest_item_result_rejected() {
        let result = IngestItemResult::rejected(100, "Contains PII");
        assert!(result.was_rejected);
        assert!(!result.was_duplicate);
        assert_eq!(result.pattern_id, None);
    }

    #[test]
    fn test_ingest_item_result_duplicate() {
        let result = IngestItemResult::duplicate(100, 0.95);
        assert!(result.was_duplicate);
        assert!(!result.was_rejected);
        assert_eq!(result.dedup_similarity, Some(0.95));
    }

    #[test]
    fn test_ingest_item_result_success() {
        let result = IngestItemResult::success(100, 95, true, "pattern-123".to_string());
        assert!(!result.was_rejected);
        assert!(!result.was_duplicate);
        assert!(result.was_redacted);
        assert_eq!(result.pattern_id, Some("pattern-123".to_string()));
    }

    #[test]
    fn test_ingest_result_add() {
        let mut summary = IngestResult::new();

        summary.add(&IngestItemResult::success(100, 100, false, "id1".to_string()));
        summary.add(&IngestItemResult::success(100, 90, true, "id2".to_string()));
        summary.add(&IngestItemResult::rejected(100, "PII"));
        summary.add(&IngestItemResult::duplicate(100, 0.95));

        assert_eq!(summary.total_processed, 4);
        assert_eq!(summary.stored_count, 2);
        assert_eq!(summary.rejected_count, 1);
        assert_eq!(summary.duplicate_count, 1);
        assert_eq!(summary.redacted_count, 1);
    }

    #[test]
    fn test_categorize_app() {
        assert_eq!(categorize_app("Visual Studio Code"), "coding");
        assert_eq!(categorize_app("Google Chrome"), "browsing");
        assert_eq!(categorize_app("Slack"), "communication");
        assert_eq!(categorize_app("Zoom"), "meetings");
        assert_eq!(categorize_app("iTerm2"), "terminal");
        assert_eq!(categorize_app("RandomApp"), "other");
    }

    #[test]
    fn test_truncate() {
        assert_eq!(truncate("short", 10), "short");
        assert_eq!(truncate("this is a longer string", 10), "this is...");
    }
}
