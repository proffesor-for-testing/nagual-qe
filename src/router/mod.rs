//! FastGRNN-Based Vendor Router for LLM Selection
//!
//! This module provides intelligent routing between local and cloud LLM vendors
//! based on query complexity estimation using a lightweight FastGRNN neural network.
//!
//! # Architecture
//!
//! ```text
//! Query -> ComplexityEstimator -> FastGRNN -> complexity score [0.0, 1.0]
//!                                       |
//!                                       v
//!                               VendorSelector -> Vendor selection
//!                                       |
//!                                       v
//!                               Fallback chain: local -> claude -> gpt -> local-large
//! ```
//!
//! # Features
//!
//! - **FastGRNN**: Lightweight recurrent neural network for fast inference (<5ms)
//! - **Complexity Estimation**: Multi-factor query analysis
//! - **Vendor Selection**: Intelligent routing with fallback chains
//! - **Metrics Integration**: Observability for routing decisions
//!
//! # Example
//!
//! ```rust,ignore
//! use nagual::router::{VendorRouter, RouterConfig, Vendor};
//!
//! let router = VendorRouter::new(RouterConfig::default())?;
//!
//! // Route a query
//! let query = "How do I implement a binary search tree?";
//! let embedding = embedder.embed(query)?;
//! let vendor = router.route(query, &embedding.embedding)?;
//!
//! match vendor {
//!     Vendor::LocalSmall => println!("Using local small model"),
//!     Vendor::LocalLarge => println!("Using local large model"),
//!     Vendor::Claude => println!("Using Claude API"),
//!     Vendor::GPT => println!("Using GPT API"),
//! }
//! ```

pub mod complexity_estimator;
pub mod fastgrnn;
pub mod vendor_selector;

// KOS P10: Compute Routing Ladder (Reflex/Retrieval/Heavy/Human lanes)
pub mod ladder;

pub use complexity_estimator::{
    ComplexityEstimator, ComplexityFeatures, ComplexityLevel, ComplexityScore, EstimatorConfig,
};
pub use fastgrnn::{
    FastGRNN, FastGRNNBackend, FastGRNNConfig, FastGRNNWeights, GRNNCell,
};
#[cfg(feature = "onnx-embed")]
pub use fastgrnn::{OnnxFastGRNN, OnnxFastGRNNConfig};
pub use vendor_selector::{
    FallbackChain, RoutingDecision, RoutingMetrics, Vendor, VendorConfig, VendorRouter,
    VendorSelector, VendorStatus,
};

use thiserror::Error;

/// Errors specific to router operations.
#[derive(Error, Debug)]
pub enum RouterError {
    /// Feature extraction error
    #[error("Feature extraction failed: {0}")]
    FeatureExtraction(String),

    /// Model inference error
    #[error("Model inference failed: {0}")]
    Inference(String),

    /// No available vendors
    #[error("No available vendors: all vendors in fallback chain are unavailable")]
    NoAvailableVendors,

    /// Invalid configuration
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    /// Model loading error
    #[error("Failed to load model from '{path}': {reason}")]
    ModelLoad { path: String, reason: String },

    /// Embedding dimension mismatch
    #[error("Embedding dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    /// Timeout error
    #[error("Routing timeout: took {actual_ms}ms, limit was {limit_ms}ms")]
    Timeout { actual_ms: u64, limit_ms: u64 },
}

/// Result type for router operations.
pub type RouterResult<T> = std::result::Result<T, RouterError>;

/// Configuration for the complete vendor router.
#[derive(Debug, Clone)]
pub struct RouterConfig {
    /// FastGRNN configuration.
    pub fastgrnn: FastGRNNConfig,

    /// Complexity estimator configuration.
    pub estimator: EstimatorConfig,

    /// Vendor selector configuration.
    pub selector: VendorConfig,

    /// Maximum routing latency in milliseconds (default: 5ms).
    pub max_latency_ms: u64,

    /// Whether to use cached routing decisions.
    pub enable_caching: bool,

    /// Cache size for routing decisions.
    pub cache_size: usize,

    /// Enable debug logging for routing decisions.
    pub debug_logging: bool,
}

impl Default for RouterConfig {
    fn default() -> Self {
        Self {
            fastgrnn: FastGRNNConfig::default(),
            estimator: EstimatorConfig::default(),
            selector: VendorConfig::default(),
            max_latency_ms: 5,
            enable_caching: true,
            cache_size: 1000,
            debug_logging: false,
        }
    }
}

impl RouterConfig {
    /// Create a new router configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set FastGRNN configuration.
    pub fn with_fastgrnn(mut self, config: FastGRNNConfig) -> Self {
        self.fastgrnn = config;
        self
    }

    /// Set estimator configuration.
    pub fn with_estimator(mut self, config: EstimatorConfig) -> Self {
        self.estimator = config;
        self
    }

    /// Set vendor selector configuration.
    pub fn with_selector(mut self, config: VendorConfig) -> Self {
        self.selector = config;
        self
    }

    /// Set maximum routing latency.
    pub fn with_max_latency(mut self, max_ms: u64) -> Self {
        self.max_latency_ms = max_ms;
        self
    }

    /// Enable or disable caching.
    pub fn with_caching(mut self, enabled: bool) -> Self {
        self.enable_caching = enabled;
        self
    }

    /// Set cache size.
    pub fn with_cache_size(mut self, size: usize) -> Self {
        self.cache_size = size;
        self
    }

    /// Enable debug logging.
    pub fn with_debug_logging(mut self, enabled: bool) -> Self {
        self.debug_logging = enabled;
        self
    }

    /// Create a configuration optimized for low latency.
    pub fn low_latency() -> Self {
        Self {
            fastgrnn: FastGRNNConfig::compact(),
            estimator: EstimatorConfig::fast(),
            selector: VendorConfig::default(),
            max_latency_ms: 2,
            enable_caching: true,
            cache_size: 5000,
            debug_logging: false,
        }
    }

    /// Create a configuration optimized for accuracy.
    pub fn high_accuracy() -> Self {
        Self {
            fastgrnn: FastGRNNConfig::default(),
            estimator: EstimatorConfig::default(),
            selector: VendorConfig::default(),
            max_latency_ms: 10,
            enable_caching: true,
            cache_size: 1000,
            debug_logging: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_router_config_default() {
        let config = RouterConfig::default();
        assert_eq!(config.max_latency_ms, 5);
        assert!(config.enable_caching);
        assert_eq!(config.cache_size, 1000);
        assert!(!config.debug_logging);
    }

    #[test]
    fn test_router_config_low_latency() {
        let config = RouterConfig::low_latency();
        assert_eq!(config.max_latency_ms, 2);
        assert_eq!(config.cache_size, 5000);
    }

    #[test]
    fn test_router_config_high_accuracy() {
        let config = RouterConfig::high_accuracy();
        assert_eq!(config.max_latency_ms, 10);
    }

    #[test]
    fn test_router_config_builder() {
        let config = RouterConfig::new()
            .with_max_latency(3)
            .with_caching(false)
            .with_cache_size(500)
            .with_debug_logging(true);

        assert_eq!(config.max_latency_ms, 3);
        assert!(!config.enable_caching);
        assert_eq!(config.cache_size, 500);
        assert!(config.debug_logging);
    }

    #[test]
    fn test_router_error_display() {
        let err = RouterError::NoAvailableVendors;
        assert!(err.to_string().contains("No available vendors"));

        let err = RouterError::Timeout {
            actual_ms: 10,
            limit_ms: 5,
        };
        assert!(err.to_string().contains("10ms"));
        assert!(err.to_string().contains("5ms"));
    }
}
