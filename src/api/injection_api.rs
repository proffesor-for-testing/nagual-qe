//! Injection namespace API for E_nagual attention bias operations.
//!
//! The Injection API provides methods for computing E_nagual attention biases
//! and building provider-specific context for injecting learned knowledge
//! into LLM prompts.
//!
//! # Example
//!
//! ```rust,ignore
//! use nagual::injection::{Provider, ContextBuilder};
//!
//! // Build E_nagual context for a query
//! let builder = nagual.injection.build_context("How to handle database timeouts?");
//! let e_nagual = builder.build();
//!
//! // Format for a specific provider
//! let context = ContextBuilder::new(Provider::Anthropic)
//!     .with_e_nagual(&e_nagual)
//!     .as_system_message()
//!     .build();
//! ```

use super::NagualState;
use crate::injection::{ENagualBuilder, ENagualConfig, InjectionContext};

/// API for E_nagual attention bias injection.
///
/// This API provides methods for computing and formatting learned knowledge
/// biases that can be injected into vendor LLM prompts (Claude, GPT, Gemini,
/// or local models).
pub struct InjectionApi {
    config: ENagualConfig,
}

impl InjectionApi {
    /// Create a new InjectionApi from shared Nagual state.
    pub(crate) fn new(_state: NagualState) -> Self {
        Self {
            config: ENagualConfig::default(),
        }
    }

    /// Build an E_nagual context for a query.
    ///
    /// Returns a builder that allows adding patterns, trajectory hints,
    /// and HNSW neighbors before computing the final E_nagual bias.
    ///
    /// # Arguments
    ///
    /// * `query` - The query to compute bias for
    ///
    /// # Returns
    ///
    /// An `ENagualBuilder` configured with the current settings.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let builder = nagual.injection.build_context("How to optimize queries?");
    /// let e_nagual = builder
    ///     .with_patterns(patterns)
    ///     .build();
    /// ```
    pub fn build_context(&self, query: &str) -> ENagualBuilder {
        ENagualBuilder::new(query).config(self.config.clone())
    }

    /// Get the current E_nagual configuration.
    pub fn config(&self) -> &ENagualConfig {
        &self.config
    }

    /// Create a new injection context with default settings.
    ///
    /// The injection context provides filtering options for controlling
    /// which patterns and trajectory hints are included in the bias.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let ctx = nagual.injection.create_context()
    ///     .with_min_reward(0.8)
    ///     .with_max_patterns(10);
    /// ```
    pub fn create_context(&self) -> InjectionContext {
        InjectionContext::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_injection_api_config_defaults() {
        let config = ENagualConfig::default();
        assert!(config.max_patterns > 0);
        assert!(config.min_pattern_reward >= 0.0);
    }

    #[test]
    fn test_injection_context_creation() {
        let ctx = InjectionContext::new()
            .with_min_reward(0.8)
            .with_max_patterns(10);
        assert_eq!(ctx.min_reward, 0.8);
        assert_eq!(ctx.max_patterns, 10);
    }
}
