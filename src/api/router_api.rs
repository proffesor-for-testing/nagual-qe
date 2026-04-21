//! Router namespace API for vendor routing operations.
//!
//! The Router API provides methods for intelligent LLM vendor routing using
//! a FastGRNN neural network to estimate query complexity and select the
//! optimal vendor (local, Claude, GPT).
//!
//! # Example
//!
//! ```rust,ignore
//! // Estimate complexity from query + embedding
//! let score = nagual.router.estimate_complexity("How to cache?", &embedding)?;
//! println!("Complexity: {:.2}", score.score);
//!
//! // Route a query
//! let decision = nagual.router.route("How to implement caching?", &embedding)?;
//! println!("Vendor: {:?}", decision.vendor);
//! ```

use tracing::{debug, instrument};

use super::NagualState;
use crate::error::{NagualError, Result};
use crate::router::{
    ComplexityScore, RouterConfig, RoutingDecision, VendorRouter,
};

/// API for FastGRNN-based vendor routing.
///
/// This API provides access to the FastGRNN complexity estimator and
/// the full vendor routing pipeline, enabling intelligent selection of
/// LLM vendors based on query characteristics.
pub struct RouterApi {
    router: VendorRouter,
}

impl RouterApi {
    /// Create a new RouterApi from shared Nagual state.
    pub(crate) fn new(_state: NagualState) -> Result<Self> {
        let router = VendorRouter::new(RouterConfig::default())
            .map_err(|e| NagualError::internal(e.to_string()))?;
        Ok(Self { router })
    }

    /// Estimate query complexity using FastGRNN.
    ///
    /// Extracts features from the query text and embedding, then runs
    /// them through the FastGRNN model to produce a complexity score.
    ///
    /// # Arguments
    ///
    /// * `query` - The user query text
    /// * `embedding` - The query embedding vector
    ///
    /// # Returns
    ///
    /// A `ComplexityScore` containing the score, features, and confidence.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let score = nagual.router.estimate_complexity("How to cache?", &embedding)?;
    /// println!("Complexity: {:.2}, Confidence: {:.2}", score.score, score.confidence);
    /// ```
    #[instrument(skip(self, embedding))]
    pub fn estimate_complexity(&self, query: &str, embedding: &[f32]) -> Result<ComplexityScore> {
        self.router
            .estimate_complexity(query, embedding)
            .map_err(|e| NagualError::internal(e.to_string()))
    }

    /// Route a query to the optimal vendor.
    ///
    /// Extracts features from the query and embedding, runs the FastGRNN
    /// complexity estimator, and selects the best vendor with a fallback chain.
    ///
    /// # Arguments
    ///
    /// * `query` - The user query text
    /// * `embedding` - The query embedding vector
    ///
    /// # Returns
    ///
    /// A `RoutingDecision` containing the selected vendor, complexity score,
    /// confidence, and fallback chain.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let decision = nagual.router.route("How to implement caching?", &embedding)?;
    /// println!("Vendor: {:?}, Complexity: {:.2}", decision.vendor, decision.complexity);
    /// ```
    #[instrument(skip(self, embedding))]
    pub fn route(&self, query: &str, embedding: &[f32]) -> Result<RoutingDecision> {
        let decision = self
            .router
            .route(query, embedding)
            .map_err(|e| NagualError::internal(e.to_string()))?;

        debug!(
            vendor = ?decision.vendor,
            complexity = decision.complexity,
            confidence = decision.confidence,
            "Query routed"
        );

        Ok(decision)
    }

    /// Get the underlying VendorRouter for advanced operations.
    pub fn vendor_router(&self) -> &VendorRouter {
        &self.router
    }
}

#[cfg(test)]
mod tests {
    use crate::router::RouterConfig;

    #[test]
    fn test_router_config_defaults() {
        let config = RouterConfig::default();
        assert_eq!(config.max_latency_ms, 5);
        assert!(config.enable_caching);
    }
}
