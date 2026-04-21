//! Injection Module - Attention Bias Injection for Vendor LLMs
//!
//! This module provides E_nagual, a mechanism for injecting learned knowledge
//! from ReasoningBank into closed-source LLM prompts. It enables knowledge
//! transfer to vendor models (Claude, GPT, Gemini) without requiring fine-tuning
//! or access to model weights.
//!
//! # Overview
//!
//! The injection system works by:
//!
//! 1. **Computing E_nagual**: Gathering relevant patterns, HNSW neighbors, and
//!    trajectory history based on the current query
//! 2. **Formatting for Provider**: Converting the bias into provider-specific
//!    formats (XML for Claude, Markdown for GPT, templates for local models)
//! 3. **Injecting into Prompt**: Adding the formatted context to system prompts,
//!    few-shot examples, or user messages
//!
//! # Key Components
//!
//! - [`ENagual`]: The core attention bias computation
//! - [`ENagualBuilder`]: Builder for constructing E_nagual from various sources
//! - [`ContextBuilder`]: Provider-specific formatting and output generation
//! - [`Provider`]: Supported LLM providers (Anthropic, OpenAI, Google, Local)
//!
//! # Example
//!
//! ```ignore
//! use nagual::injection::{ENagual, ENagualBuilder, ContextBuilder, Provider};
//!
//! // 1. Compute E_nagual from patterns and trajectories
//! let e_nagual = ENagualBuilder::new("How to handle database timeouts?")
//!     .config(ENagualConfig::default())
//!     .with_patterns(patterns)
//!     .with_trajectories(&trajectories)
//!     .with_hnsw_neighbors(&neighbors)
//!     .build();
//!
//! // 2. Format for specific provider
//! let context = ContextBuilder::new(Provider::Anthropic)
//!     .with_e_nagual(&e_nagual)
//!     .as_system_message()
//!     .build();
//!
//! // 3. Use in your LLM call
//! let response = claude_client
//!     .system(context.system_message.unwrap())
//!     .user("How should I handle database timeouts in my service?")
//!     .send()
//!     .await?;
//! ```
//!
//! # Performance Characteristics
//!
//! - E_nagual computation: O(k) where k is the number of patterns/neighbors
//! - HNSW search: O(log n) with 150x-12,500x speedup over brute force
//! - Context formatting: O(p) where p is number of patterns in output
//!
//! # Supported Providers
//!
//! | Provider | Format | Best For |
//! |----------|--------|----------|
//! | Anthropic | XML tags | Claude models (excellent XML parsing) |
//! | OpenAI | Markdown | GPT-4, o1 models |
//! | Google | Markdown | Gemini models |
//! | Local | Templates | Llama, Mistral, Qwen |
//! | Generic | Plain text | Any model |
//!
//! # Open-Weight Model Support (Attention Surgery)
//!
//! For open-weight models (GGUF via llama.cpp, candle), direct attention
//! weight modification is available via [`attention_surgery`] and
//! [`model_hooks`]:
//!
//! ```ignore
//! use nagual::injection::{AttentionSurgery, AttentionSurgeryConfig, ModelConfig};
//! use nagual::injection::{HookRegistry, ENagualHook};
//!
//! let surgery = AttentionSurgery::new(AttentionSurgeryConfig::default());
//! let biases = surgery.prepare_biases(&e_nagual, &ModelConfig::llama_7b());
//!
//! let hook = ENagualHook::new(surgery, biases);
//! let mut registry = HookRegistry::new();
//! registry.register(Box::new(hook));
//! ```

pub mod attention_surgery;
mod context_builder;
mod e_nagual;
pub mod model_hooks;

pub use attention_surgery::{
    AttentionBias, AttentionSurgery, AttentionSurgeryConfig, BiasMethod, ModelConfig,
};
pub use context_builder::{
    BuiltContext, ContextBuilder, ContextConfig, FewShotMessage, Provider,
    format_for_anthropic, format_for_local, format_for_openai,
};
pub use e_nagual::{
    ENagual, ENagualBuilder, ENagualConfig, Example, TrajectoryHint,
};
pub use model_hooks::{
    AttentionState, ENagualHook, HookRegistry, ModelHook,
};

/// Injection context combining storage references for E_nagual computation.
///
/// This struct provides a convenient way to pass all necessary components
/// to the E_nagual computation.
#[derive(Clone)]
pub struct InjectionContext {
    /// Minimum pattern reward for inclusion.
    pub min_reward: f32,

    /// Maximum patterns to include.
    pub max_patterns: usize,

    /// Maximum trajectory hints to include.
    pub max_hints: usize,

    /// Whether to include negative examples.
    pub include_negative: bool,

    /// Session ID for filtering (optional).
    pub session_id: Option<String>,

    /// Agent ID for filtering (optional).
    pub agent_id: Option<String>,
}

impl Default for InjectionContext {
    fn default() -> Self {
        Self {
            min_reward: 0.6,
            max_patterns: 5,
            max_hints: 3,
            include_negative: true,
            session_id: None,
            agent_id: None,
        }
    }
}

impl InjectionContext {
    /// Create a new injection context.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the minimum reward.
    pub fn with_min_reward(mut self, min_reward: f32) -> Self {
        self.min_reward = min_reward.clamp(0.0, 1.0);
        self
    }

    /// Set the maximum patterns.
    pub fn with_max_patterns(mut self, max: usize) -> Self {
        self.max_patterns = max;
        self
    }

    /// Set the session ID filter.
    pub fn with_session_id(mut self, session_id: impl Into<String>) -> Self {
        self.session_id = Some(session_id.into());
        self
    }

    /// Set the agent ID filter.
    pub fn with_agent_id(mut self, agent_id: impl Into<String>) -> Self {
        self.agent_id = Some(agent_id.into());
        self
    }

    /// Convert to ENagualConfig.
    pub fn to_e_nagual_config(&self) -> ENagualConfig {
        ENagualConfig {
            max_patterns: self.max_patterns,
            max_trajectory_steps: self.max_hints,
            min_pattern_reward: self.min_reward,
            include_negative_examples: self.include_negative,
            ..ENagualConfig::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_injection_context_default() {
        let ctx = InjectionContext::default();
        assert_eq!(ctx.min_reward, 0.6);
        assert_eq!(ctx.max_patterns, 5);
        assert!(ctx.include_negative);
    }

    #[test]
    fn test_injection_context_builder() {
        let ctx = InjectionContext::new()
            .with_min_reward(0.8)
            .with_max_patterns(10)
            .with_session_id("session-123")
            .with_agent_id("agent-456");

        assert_eq!(ctx.min_reward, 0.8);
        assert_eq!(ctx.max_patterns, 10);
        assert_eq!(ctx.session_id, Some("session-123".to_string()));
        assert_eq!(ctx.agent_id, Some("agent-456".to_string()));
    }

    #[test]
    fn test_to_e_nagual_config() {
        let ctx = InjectionContext::new()
            .with_min_reward(0.7)
            .with_max_patterns(8);

        let config = ctx.to_e_nagual_config();
        assert_eq!(config.min_pattern_reward, 0.7);
        assert_eq!(config.max_patterns, 8);
    }
}
