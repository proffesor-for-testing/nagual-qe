//! Configuration types for the OSpipe pipeline.
#![allow(dead_code)]

use std::path::PathBuf;
use std::time::Duration;

use serde::{Deserialize, Serialize};

/// Embedding dimension options.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum EmbeddingDim {
    /// 128-dimensional embeddings (nagual native, optimized for storage)
    #[default]
    Dim128,
    /// 384-dimensional embeddings (OSpipe native, MiniLM original)
    Dim384,
}

impl EmbeddingDim {
    /// Get the numeric dimension value.
    pub fn as_usize(&self) -> usize {
        match self {
            EmbeddingDim::Dim128 => 128,
            EmbeddingDim::Dim384 => 384,
        }
    }

    /// Parse from string.
    pub fn from_str(s: &str) -> Option<Self> {
        match s.trim().to_lowercase().as_str() {
            "128" | "dim128" => Some(EmbeddingDim::Dim128),
            "384" | "dim384" => Some(EmbeddingDim::Dim384),
            _ => None,
        }
    }
}

impl std::fmt::Display for EmbeddingDim {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EmbeddingDim::Dim128 => write!(f, "128"),
            EmbeddingDim::Dim384 => write!(f, "384"),
        }
    }
}

/// Configuration for the OSpipe pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OSpipeConfig {
    /// Enable PII detection and redaction.
    pub pii_enabled: bool,

    /// PII policy: "reject", "redact", "warn", "allow"
    pub pii_policy: String,

    /// Enable deduplication.
    pub dedup_enabled: bool,

    /// Cosine similarity threshold for deduplication (0.0-1.0).
    pub dedup_threshold: f32,

    /// Time window for deduplication.
    pub dedup_window: Duration,

    /// Embedding dimension to use.
    pub embedding_dim: EmbeddingDim,

    /// Whether to generate embeddings.
    pub generate_embeddings: bool,

    /// Path to the ONNX model file (optional, uses default if not set).
    pub model_path: Option<PathBuf>,

    /// Path to the tokenizer JSON file (optional, uses default if not set).
    pub tokenizer_path: Option<PathBuf>,
}

impl Default for OSpipeConfig {
    fn default() -> Self {
        Self {
            pii_enabled: true,
            pii_policy: "redact".to_string(),
            dedup_enabled: true,
            dedup_threshold: 0.9,
            dedup_window: Duration::from_secs(5 * 60), // 5 minutes
            embedding_dim: EmbeddingDim::Dim128, // nagual native dimension
            generate_embeddings: true,
            model_path: None,
            tokenizer_path: None,
        }
    }
}

impl OSpipeConfig {
    /// Create a minimal config with PII disabled and no dedup.
    pub fn minimal() -> Self {
        Self {
            pii_enabled: false,
            dedup_enabled: false,
            generate_embeddings: false,
            ..Default::default()
        }
    }

    /// Builder-style method to set PII policy.
    pub fn with_pii_policy(mut self, policy: &str) -> Self {
        self.pii_policy = policy.to_string();
        self
    }

    /// Builder-style method to set dedup threshold.
    pub fn with_dedup_threshold(mut self, threshold: f32) -> Self {
        self.dedup_threshold = threshold;
        self
    }

    /// Builder-style method to set dedup window.
    pub fn with_dedup_window(mut self, window: Duration) -> Self {
        self.dedup_window = window;
        self
    }

    /// Builder-style method to set embedding dimension.
    pub fn with_embedding_dim(mut self, dim: EmbeddingDim) -> Self {
        self.embedding_dim = dim;
        self
    }

    /// Builder-style method to enable/disable PII detection.
    pub fn with_pii_enabled(mut self, enabled: bool) -> Self {
        self.pii_enabled = enabled;
        self
    }

    /// Builder-style method to enable/disable deduplication.
    pub fn with_dedup_enabled(mut self, enabled: bool) -> Self {
        self.dedup_enabled = enabled;
        self
    }

    /// Builder-style method to enable/disable embedding generation.
    pub fn with_generate_embeddings(mut self, enabled: bool) -> Self {
        self.generate_embeddings = enabled;
        self
    }
}

/// Configuration for a single ingest operation.
#[derive(Debug, Clone)]
pub struct IngestConfig {
    /// Time range start.
    pub since: chrono::DateTime<chrono::Utc>,

    /// Content type filter: "ocr", "audio", "input", "all".
    pub content_type: String,

    /// Filter by application name.
    pub app_name: Option<String>,

    /// Only ingest focused (active) window content.
    pub focused_only: bool,

    /// Minimum text length to include.
    pub min_length: usize,

    /// Path to SQLite database.
    pub db_path: PathBuf,

    /// PostgreSQL connection URL.
    pub postgres_url: Option<String>,

    /// Screenpipe API URL.
    pub screenpipe_url: String,

    /// OSpipe-specific configuration.
    pub ospipe: OSpipeConfig,
}

impl IngestConfig {
    /// Create a new ingest config with default OSpipe settings.
    pub fn new(
        since: chrono::DateTime<chrono::Utc>,
        db_path: PathBuf,
        screenpipe_url: String,
    ) -> Self {
        Self {
            since,
            content_type: "all".to_string(),
            app_name: None,
            focused_only: false,
            min_length: 50,
            db_path,
            postgres_url: None,
            screenpipe_url,
            ospipe: OSpipeConfig::default(),
        }
    }

    /// Builder-style method to set content type.
    pub fn with_content_type(mut self, content_type: &str) -> Self {
        self.content_type = content_type.to_string();
        self
    }

    /// Builder-style method to set app name filter.
    pub fn with_app_name(mut self, app_name: Option<String>) -> Self {
        self.app_name = app_name;
        self
    }

    /// Builder-style method to set focused only.
    pub fn with_focused_only(mut self, focused_only: bool) -> Self {
        self.focused_only = focused_only;
        self
    }

    /// Builder-style method to set minimum length.
    pub fn with_min_length(mut self, min_length: usize) -> Self {
        self.min_length = min_length;
        self
    }

    /// Builder-style method to set PostgreSQL URL.
    pub fn with_postgres_url(mut self, url: Option<String>) -> Self {
        self.postgres_url = url;
        self
    }

    /// Builder-style method to set OSpipe config.
    pub fn with_ospipe_config(mut self, ospipe: OSpipeConfig) -> Self {
        self.ospipe = ospipe;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_dim_as_usize() {
        assert_eq!(EmbeddingDim::Dim128.as_usize(), 128);
        assert_eq!(EmbeddingDim::Dim384.as_usize(), 384);
    }

    #[test]
    fn test_embedding_dim_from_str() {
        assert_eq!(EmbeddingDim::from_str("128"), Some(EmbeddingDim::Dim128));
        assert_eq!(EmbeddingDim::from_str("384"), Some(EmbeddingDim::Dim384));
        assert_eq!(EmbeddingDim::from_str("dim128"), Some(EmbeddingDim::Dim128));
        assert_eq!(EmbeddingDim::from_str("invalid"), None);
    }

    #[test]
    fn test_ospipe_config_default() {
        let config = OSpipeConfig::default();
        assert!(config.pii_enabled);
        assert_eq!(config.pii_policy, "redact");
        assert!(config.dedup_enabled);
        assert_eq!(config.dedup_threshold, 0.9);
        assert_eq!(config.embedding_dim, EmbeddingDim::Dim128);
    }

    #[test]
    fn test_ospipe_config_minimal() {
        let config = OSpipeConfig::minimal();
        assert!(!config.pii_enabled);
        assert!(!config.dedup_enabled);
        assert!(!config.generate_embeddings);
    }

    #[test]
    fn test_ospipe_config_builder() {
        let config = OSpipeConfig::default()
            .with_pii_policy("reject")
            .with_dedup_threshold(0.85)
            .with_embedding_dim(EmbeddingDim::Dim128);

        assert_eq!(config.pii_policy, "reject");
        assert_eq!(config.dedup_threshold, 0.85);
        assert_eq!(config.embedding_dim, EmbeddingDim::Dim128);
    }
}
