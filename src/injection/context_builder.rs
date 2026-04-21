//! Context Builder - LLM Provider-specific Formatting
//!
//! This module provides formatters for different LLM providers, converting
//! E_nagual attention bias into provider-specific prompt formats.
//!
//! # Supported Providers
//!
//! - **Anthropic (Claude)**: System prompt format with XML-style tags
//! - **OpenAI (GPT)**: System/user message format with structured content
//! - **Local Models**: Template-based prompts (Llama, Mistral, etc.)
//!
//! # Example
//!
//! ```ignore
//! use nagual::injection::{ENagual, ContextBuilder, Provider};
//!
//! let e_nagual = compute_e_nagual(...).await?;
//!
//! // Format for Claude
//! let claude_context = ContextBuilder::new(Provider::Anthropic)
//!     .with_e_nagual(&e_nagual)
//!     .build();
//!
//! // Format for GPT-4
//! let gpt_context = ContextBuilder::new(Provider::OpenAI)
//!     .with_e_nagual(&e_nagual)
//!     .as_system_message()
//!     .build();
//! ```

use serde::{Deserialize, Serialize};

use super::e_nagual::ENagual;

/// Supported LLM providers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Provider {
    /// Anthropic Claude models
    Anthropic,
    /// OpenAI GPT models
    OpenAI,
    /// Google Gemini models
    Google,
    /// Local models (Llama, Mistral, etc.)
    Local,
    /// Generic format (works with most models)
    Generic,
}

impl Provider {
    /// Get the provider from a model name hint.
    pub fn from_model_name(model: &str) -> Self {
        let lower = model.to_lowercase();
        if lower.contains("claude") || lower.contains("anthropic") {
            Provider::Anthropic
        } else if lower.contains("gpt") || lower.contains("openai") || lower.contains("o1") {
            Provider::OpenAI
        } else if lower.contains("gemini") || lower.contains("palm") || lower.contains("google") {
            Provider::Google
        } else if lower.contains("llama") || lower.contains("mistral") || lower.contains("qwen") {
            Provider::Local
        } else {
            Provider::Generic
        }
    }
}

impl std::fmt::Display for Provider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Provider::Anthropic => write!(f, "anthropic"),
            Provider::OpenAI => write!(f, "openai"),
            Provider::Google => write!(f, "google"),
            Provider::Local => write!(f, "local"),
            Provider::Generic => write!(f, "generic"),
        }
    }
}

/// Output format type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputFormat {
    /// System prompt content
    SystemPrompt,
    /// User message content
    UserMessage,
    /// Separate system and user messages
    SplitMessages,
    /// Few-shot conversation format
    FewShot,
}

/// Configuration for context building.
#[derive(Debug, Clone)]
pub struct ContextConfig {
    /// Maximum token budget for the context.
    pub max_tokens: usize,

    /// Whether to include confidence indicators.
    pub include_confidence: bool,

    /// Whether to include trajectory hints.
    pub include_hints: bool,

    /// Whether to include negative examples.
    pub include_negative: bool,

    /// Custom prefix to add before the context.
    pub custom_prefix: Option<String>,

    /// Custom suffix to add after the context.
    pub custom_suffix: Option<String>,

    /// Estimated characters per token (for budget calculation).
    pub chars_per_token: f32,
}

impl Default for ContextConfig {
    fn default() -> Self {
        Self {
            max_tokens: 2000,
            include_confidence: true,
            include_hints: true,
            include_negative: true,
            custom_prefix: None,
            custom_suffix: None,
            chars_per_token: 4.0,
        }
    }
}

impl ContextConfig {
    /// Create a minimal configuration.
    pub fn minimal() -> Self {
        Self {
            max_tokens: 500,
            include_confidence: false,
            include_hints: false,
            include_negative: false,
            ..Default::default()
        }
    }

    /// Set the maximum tokens.
    pub fn with_max_tokens(mut self, max: usize) -> Self {
        self.max_tokens = max;
        self
    }

    /// Set a custom prefix.
    pub fn with_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.custom_prefix = Some(prefix.into());
        self
    }

    /// Set a custom suffix.
    pub fn with_suffix(mut self, suffix: impl Into<String>) -> Self {
        self.custom_suffix = Some(suffix.into());
        self
    }

    /// Get the character budget.
    fn char_budget(&self) -> usize {
        (self.max_tokens as f32 * self.chars_per_token) as usize
    }
}

/// Result of context building.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuiltContext {
    /// The formatted context content.
    pub content: String,

    /// System message (if applicable).
    pub system_message: Option<String>,

    /// User message (if applicable).
    pub user_message: Option<String>,

    /// Few-shot examples (if applicable).
    pub few_shot_examples: Option<Vec<FewShotMessage>>,

    /// Estimated token count.
    pub estimated_tokens: usize,

    /// The provider this was formatted for.
    pub provider: Provider,

    /// The output format used.
    pub format: String,
}

/// A message in few-shot format.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FewShotMessage {
    /// Role (system, user, assistant).
    pub role: String,

    /// Message content.
    pub content: String,
}

/// Builder for constructing provider-specific context.
pub struct ContextBuilder<'a> {
    provider: Provider,
    e_nagual: Option<&'a ENagual>,
    config: ContextConfig,
    output_format: OutputFormat,
    user_query: Option<String>,
}

impl<'a> ContextBuilder<'a> {
    /// Create a new context builder for the specified provider.
    pub fn new(provider: Provider) -> Self {
        Self {
            provider,
            e_nagual: None,
            config: ContextConfig::default(),
            output_format: OutputFormat::SystemPrompt,
            user_query: None,
        }
    }

    /// Set the E_nagual to format.
    pub fn with_e_nagual(mut self, e_nagual: &'a ENagual) -> Self {
        self.e_nagual = Some(e_nagual);
        self
    }

    /// Set the configuration.
    pub fn with_config(mut self, config: ContextConfig) -> Self {
        self.config = config;
        self
    }

    /// Set the output format to system message.
    pub fn as_system_message(mut self) -> Self {
        self.output_format = OutputFormat::SystemPrompt;
        self
    }

    /// Set the output format to user message.
    pub fn as_user_message(mut self) -> Self {
        self.output_format = OutputFormat::UserMessage;
        self
    }

    /// Set the output format to split messages.
    pub fn as_split_messages(mut self) -> Self {
        self.output_format = OutputFormat::SplitMessages;
        self
    }

    /// Set the output format to few-shot.
    pub fn as_few_shot(mut self) -> Self {
        self.output_format = OutputFormat::FewShot;
        self
    }

    /// Set the user query (for few-shot format).
    pub fn with_user_query(mut self, query: impl Into<String>) -> Self {
        self.user_query = Some(query.into());
        self
    }

    /// Build the context.
    pub fn build(self) -> BuiltContext {
        let e_nagual = match self.e_nagual {
            Some(e) => e,
            None => {
                return BuiltContext {
                    content: String::new(),
                    system_message: None,
                    user_message: None,
                    few_shot_examples: None,
                    estimated_tokens: 0,
                    provider: self.provider,
                    format: format!("{:?}", self.output_format),
                };
            }
        };

        match self.provider {
            Provider::Anthropic => self.build_anthropic(e_nagual),
            Provider::OpenAI => self.build_openai(e_nagual),
            Provider::Google => self.build_google(e_nagual),
            Provider::Local => self.build_local(e_nagual),
            Provider::Generic => self.build_generic(e_nagual),
        }
    }

    /// Build context for Anthropic Claude.
    fn build_anthropic(&self, e_nagual: &ENagual) -> BuiltContext {
        match self.output_format {
            OutputFormat::SystemPrompt => self.build_anthropic_system(e_nagual),
            OutputFormat::FewShot => self.build_anthropic_few_shot(e_nagual),
            _ => self.build_anthropic_system(e_nagual),
        }
    }

    /// Build Anthropic system prompt format.
    fn build_anthropic_system(&self, e_nagual: &ENagual) -> BuiltContext {
        let mut content = String::new();

        // Add custom prefix
        if let Some(ref prefix) = self.config.custom_prefix {
            content.push_str(prefix);
            content.push_str("\n\n");
        }

        // Use XML format for Claude (it handles XML tags well)
        content.push_str(&e_nagual.to_xml_context());

        // Add custom suffix
        if let Some(ref suffix) = self.config.custom_suffix {
            content.push_str("\n\n");
            content.push_str(suffix);
        }

        // Truncate if needed
        let char_budget = self.config.char_budget();
        if content.len() > char_budget {
            content = truncate_at_boundary(&content, char_budget);
        }

        let estimated_tokens = (content.len() as f32 / self.config.chars_per_token) as usize;

        BuiltContext {
            content: content.clone(),
            system_message: Some(content),
            user_message: None,
            few_shot_examples: None,
            estimated_tokens,
            provider: Provider::Anthropic,
            format: "system_prompt".to_string(),
        }
    }

    /// Build Anthropic few-shot format.
    fn build_anthropic_few_shot(&self, e_nagual: &ENagual) -> BuiltContext {
        let examples = e_nagual.to_few_shot_examples();
        let mut messages: Vec<FewShotMessage> = Vec::new();

        // Add system context as first message
        if e_nagual.has_content() && self.config.include_confidence {
            messages.push(FewShotMessage {
                role: "system".to_string(),
                content: format!(
                    "You have access to learned patterns with {:.0}% overall confidence.",
                    e_nagual.overall_confidence() * 100.0
                ),
            });
        }

        // Add examples as user/assistant pairs
        for example in examples.iter().take(3) {
            messages.push(FewShotMessage {
                role: "user".to_string(),
                content: example.input.clone(),
            });
            messages.push(FewShotMessage {
                role: "assistant".to_string(),
                content: example.output.clone(),
            });
        }

        // Add the actual user query
        if let Some(ref query) = self.user_query {
            messages.push(FewShotMessage {
                role: "user".to_string(),
                content: query.clone(),
            });
        }

        let content = messages
            .iter()
            .map(|m| format!("[{}] {}", m.role, m.content))
            .collect::<Vec<_>>()
            .join("\n\n");

        let estimated_tokens = (content.len() as f32 / self.config.chars_per_token) as usize;

        BuiltContext {
            content,
            system_message: None,
            user_message: None,
            few_shot_examples: Some(messages),
            estimated_tokens,
            provider: Provider::Anthropic,
            format: "few_shot".to_string(),
        }
    }

    /// Build context for OpenAI GPT.
    fn build_openai(&self, e_nagual: &ENagual) -> BuiltContext {
        match self.output_format {
            OutputFormat::SystemPrompt => self.build_openai_system(e_nagual),
            OutputFormat::SplitMessages => self.build_openai_split(e_nagual),
            OutputFormat::FewShot => self.build_openai_few_shot(e_nagual),
            _ => self.build_openai_system(e_nagual),
        }
    }

    /// Build OpenAI system message format.
    fn build_openai_system(&self, e_nagual: &ENagual) -> BuiltContext {
        let mut content = String::new();

        // Add custom prefix
        if let Some(ref prefix) = self.config.custom_prefix {
            content.push_str(prefix);
            content.push_str("\n\n");
        }

        // Build structured system prompt
        content.push_str("## Learned Context\n\n");

        if self.config.include_confidence {
            content.push_str(&format!(
                "Overall confidence in learned patterns: {:.0}%\n\n",
                e_nagual.overall_confidence() * 100.0
            ));
        }

        // Add patterns
        if !e_nagual.relevant_patterns.is_empty() {
            content.push_str("### Relevant Patterns\n\n");
            for (i, scored) in e_nagual.relevant_patterns.iter().enumerate() {
                let pattern = &scored.pattern;
                content.push_str(&format!(
                    "**Pattern {}** ({}% match)\n",
                    i + 1,
                    (scored.similarity * 100.0) as i32
                ));
                content.push_str(&format!("- Problem: {}\n", pattern.problem));
                content.push_str(&format!("- Solution: {}\n", pattern.solution));
                if let Some(ref ctx) = pattern.context {
                    if !ctx.is_empty() {
                        content.push_str(&format!("- Context: {}\n", ctx));
                    }
                }
                content.push('\n');
            }
        }

        // Add hints
        if self.config.include_hints && !e_nagual.trajectory_hints.is_empty() {
            content.push_str("### Reasoning Hints\n\n");
            for hint in &e_nagual.trajectory_hints {
                content.push_str(&format!(
                    "- {} (confidence: {}%)\n",
                    hint.decision,
                    (hint.confidence * 100.0) as i32
                ));
            }
            content.push('\n');
        }

        // Add negative examples
        if self.config.include_negative && !e_nagual.negative_examples.is_empty() {
            content.push_str("### Approaches to Avoid\n\n");
            for pattern in &e_nagual.negative_examples {
                content.push_str(&format!(
                    "- {} (failed approach: {})\n",
                    pattern.problem,
                    pattern.solution
                ));
            }
            content.push('\n');
        }

        // Add custom suffix
        if let Some(ref suffix) = self.config.custom_suffix {
            content.push_str(suffix);
        }

        let estimated_tokens = (content.len() as f32 / self.config.chars_per_token) as usize;

        BuiltContext {
            content: content.clone(),
            system_message: Some(content),
            user_message: None,
            few_shot_examples: None,
            estimated_tokens,
            provider: Provider::OpenAI,
            format: "system_prompt".to_string(),
        }
    }

    /// Build OpenAI split messages format.
    fn build_openai_split(&self, e_nagual: &ENagual) -> BuiltContext {
        // System message with context
        let system = format!(
            "You have access to learned patterns with {:.0}% confidence. \
            Use these patterns to inform your responses:\n\n{}",
            e_nagual.overall_confidence() * 100.0,
            e_nagual.to_prompt_prefix()
        );

        // User message (if provided)
        let user = self.user_query.clone();

        let content = format!(
            "[System]\n{}\n\n[User]\n{}",
            system,
            user.as_deref().unwrap_or("(no query)")
        );

        let estimated_tokens = (content.len() as f32 / self.config.chars_per_token) as usize;

        BuiltContext {
            content,
            system_message: Some(system),
            user_message: user,
            few_shot_examples: None,
            estimated_tokens,
            provider: Provider::OpenAI,
            format: "split_messages".to_string(),
        }
    }

    /// Build OpenAI few-shot format.
    fn build_openai_few_shot(&self, e_nagual: &ENagual) -> BuiltContext {
        let examples = e_nagual.to_few_shot_examples();
        let mut messages: Vec<FewShotMessage> = Vec::new();

        // System message
        messages.push(FewShotMessage {
            role: "system".to_string(),
            content: "You are a helpful assistant that learns from past successful patterns.".to_string(),
        });

        // Add examples
        for example in examples.iter().take(3) {
            messages.push(FewShotMessage {
                role: "user".to_string(),
                content: example.input.clone(),
            });
            messages.push(FewShotMessage {
                role: "assistant".to_string(),
                content: example.output.clone(),
            });
        }

        // Add actual query
        if let Some(ref query) = self.user_query {
            messages.push(FewShotMessage {
                role: "user".to_string(),
                content: query.clone(),
            });
        }

        let content = serde_json::to_string_pretty(&messages).unwrap_or_default();
        let estimated_tokens = (content.len() as f32 / self.config.chars_per_token) as usize;

        BuiltContext {
            content,
            system_message: None,
            user_message: None,
            few_shot_examples: Some(messages),
            estimated_tokens,
            provider: Provider::OpenAI,
            format: "few_shot".to_string(),
        }
    }

    /// Build context for Google Gemini.
    fn build_google(&self, e_nagual: &ENagual) -> BuiltContext {
        // Gemini uses a similar format to OpenAI
        let mut content = String::new();

        content.push_str("# Context from Learned Patterns\n\n");

        // Patterns as markdown
        if !e_nagual.relevant_patterns.is_empty() {
            content.push_str("## Relevant Knowledge\n\n");
            for scored in &e_nagual.relevant_patterns {
                let pattern = &scored.pattern;
                content.push_str(&format!("### {}\n", pattern.problem));
                content.push_str(&format!("**Solution:** {}\n\n", pattern.solution));
            }
        }

        // Hints
        if self.config.include_hints && !e_nagual.trajectory_hints.is_empty() {
            content.push_str("## Reasoning Guidelines\n\n");
            for hint in &e_nagual.trajectory_hints {
                content.push_str(&format!("- {}\n", hint.decision));
            }
            content.push('\n');
        }

        let estimated_tokens = (content.len() as f32 / self.config.chars_per_token) as usize;

        BuiltContext {
            content: content.clone(),
            system_message: Some(content),
            user_message: self.user_query.clone(),
            few_shot_examples: None,
            estimated_tokens,
            provider: Provider::Google,
            format: "system_prompt".to_string(),
        }
    }

    /// Build context for local models.
    fn build_local(&self, e_nagual: &ENagual) -> BuiltContext {
        // Local models often work best with simple, direct prompts
        let mut content = String::new();

        // Use a simple template format
        content.push_str("### Instruction\n\n");
        content.push_str("Use the following learned patterns to help answer questions:\n\n");

        // Add patterns in simple format
        for (i, scored) in e_nagual.relevant_patterns.iter().enumerate() {
            let pattern = &scored.pattern;
            content.push_str(&format!(
                "Pattern {}: When asked about \"{}\", the solution is: {}\n\n",
                i + 1,
                pattern.problem,
                pattern.solution
            ));
        }

        // Add the query if provided
        if let Some(ref query) = self.user_query {
            content.push_str("### Question\n\n");
            content.push_str(query);
            content.push_str("\n\n### Response\n\n");
        }

        let estimated_tokens = (content.len() as f32 / self.config.chars_per_token) as usize;

        BuiltContext {
            content: content.clone(),
            system_message: None,
            user_message: Some(content),
            few_shot_examples: None,
            estimated_tokens,
            provider: Provider::Local,
            format: "template".to_string(),
        }
    }

    /// Build generic context.
    fn build_generic(&self, e_nagual: &ENagual) -> BuiltContext {
        // Use the standard prompt prefix format
        let content = e_nagual.to_prompt_prefix();
        let estimated_tokens = (content.len() as f32 / self.config.chars_per_token) as usize;

        BuiltContext {
            content: content.clone(),
            system_message: Some(content),
            user_message: self.user_query.clone(),
            few_shot_examples: None,
            estimated_tokens,
            provider: Provider::Generic,
            format: "prompt_prefix".to_string(),
        }
    }
}

/// Format E_nagual for Anthropic Claude.
///
/// This is a convenience function for the most common use case.
pub fn format_for_anthropic(e_nagual: &ENagual) -> String {
    ContextBuilder::new(Provider::Anthropic)
        .with_e_nagual(e_nagual)
        .build()
        .content
}

/// Format E_nagual for OpenAI GPT.
///
/// This is a convenience function for the most common use case.
pub fn format_for_openai(e_nagual: &ENagual) -> String {
    ContextBuilder::new(Provider::OpenAI)
        .with_e_nagual(e_nagual)
        .build()
        .content
}

/// Format E_nagual for local models.
///
/// This is a convenience function for the most common use case.
pub fn format_for_local(e_nagual: &ENagual) -> String {
    ContextBuilder::new(Provider::Local)
        .with_e_nagual(e_nagual)
        .build()
        .content
}

/// Truncate text at a reasonable boundary (sentence or word).
fn truncate_at_boundary(text: &str, max_len: usize) -> String {
    if text.len() <= max_len {
        return text.to_string();
    }

    // Find a good break point
    let target = max_len.saturating_sub(3); // Leave room for "..."

    // Try to break at sentence boundary
    if let Some(pos) = text[..target].rfind(". ") {
        return format!("{}...", &text[..pos + 1]);
    }

    // Try to break at word boundary
    if let Some(pos) = text[..target].rfind(' ') {
        return format!("{}...", &text[..pos]);
    }

    // Hard truncate
    format!("{}...", &text[..target])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::injection::e_nagual::ENagual;
    use crate::reasoning_bank::{FactorScores, Pattern, ScoredPattern};

    fn create_test_e_nagual() -> ENagual {
        let pattern = Pattern::new(
            "How to handle errors?",
            "Use Result type with proper error handling",
            "rust.error_handling"
        )
        .with_context("Rust best practices")
        .with_confidence(0.9)
        .with_reward(0.85);

        let scored = ScoredPattern {
            pattern,
            similarity: 0.92,
            final_score: 0.88,
            factor_scores: FactorScores::default(),
        };

        ENagual::new("test query").with_patterns(vec![scored])
    }

    #[test]
    fn test_provider_from_model_name() {
        assert_eq!(Provider::from_model_name("claude-3-opus"), Provider::Anthropic);
        assert_eq!(Provider::from_model_name("gpt-4"), Provider::OpenAI);
        assert_eq!(Provider::from_model_name("gemini-pro"), Provider::Google);
        assert_eq!(Provider::from_model_name("llama-2"), Provider::Local);
        assert_eq!(Provider::from_model_name("unknown"), Provider::Generic);
    }

    #[test]
    fn test_context_config_default() {
        let config = ContextConfig::default();
        assert_eq!(config.max_tokens, 2000);
        assert!(config.include_confidence);
        assert!(config.include_hints);
    }

    #[test]
    fn test_context_config_minimal() {
        let config = ContextConfig::minimal();
        assert_eq!(config.max_tokens, 500);
        assert!(!config.include_confidence);
    }

    #[test]
    fn test_build_anthropic_system() {
        let e_nagual = create_test_e_nagual();
        let context = ContextBuilder::new(Provider::Anthropic)
            .with_e_nagual(&e_nagual)
            .build();

        assert!(context.content.contains("<learned_context>"));
        assert!(context.content.contains("<patterns>"));
        assert!(context.system_message.is_some());
        assert_eq!(context.provider, Provider::Anthropic);
    }

    #[test]
    fn test_build_openai_system() {
        let e_nagual = create_test_e_nagual();
        let context = ContextBuilder::new(Provider::OpenAI)
            .with_e_nagual(&e_nagual)
            .build();

        assert!(context.content.contains("## Learned Context"));
        assert!(context.content.contains("Relevant Patterns"));
        assert!(context.system_message.is_some());
        assert_eq!(context.provider, Provider::OpenAI);
    }

    #[test]
    fn test_build_few_shot() {
        let e_nagual = create_test_e_nagual();
        let context = ContextBuilder::new(Provider::OpenAI)
            .with_e_nagual(&e_nagual)
            .with_user_query("How do I handle timeouts?")
            .as_few_shot()
            .build();

        assert!(context.few_shot_examples.is_some());
        let examples = context.few_shot_examples.unwrap();
        assert!(!examples.is_empty());
    }

    #[test]
    fn test_build_local() {
        let e_nagual = create_test_e_nagual();
        let context = ContextBuilder::new(Provider::Local)
            .with_e_nagual(&e_nagual)
            .with_user_query("What about error handling?")
            .build();

        assert!(context.content.contains("### Instruction"));
        assert!(context.content.contains("### Question"));
        assert!(context.content.contains("### Response"));
    }

    #[test]
    fn test_build_with_config() {
        let e_nagual = create_test_e_nagual();
        let config = ContextConfig::default()
            .with_max_tokens(500)
            .with_prefix("Custom prefix:")
            .with_suffix("Custom suffix.");

        let context = ContextBuilder::new(Provider::Anthropic)
            .with_e_nagual(&e_nagual)
            .with_config(config)
            .build();

        assert!(context.content.contains("Custom prefix:"));
        assert!(context.content.contains("Custom suffix."));
    }

    #[test]
    fn test_format_convenience_functions() {
        let e_nagual = create_test_e_nagual();

        let anthropic = format_for_anthropic(&e_nagual);
        assert!(anthropic.contains("<learned_context>"));

        let openai = format_for_openai(&e_nagual);
        assert!(openai.contains("## Learned Context"));

        let local = format_for_local(&e_nagual);
        assert!(local.contains("### Instruction"));
    }

    #[test]
    fn test_truncate_at_boundary() {
        let text = "This is a sentence. This is another sentence. And one more.";
        let truncated = truncate_at_boundary(text, 30);

        assert!(truncated.len() <= 33); // Allow for "..."
        assert!(truncated.ends_with("..."));
    }

    #[test]
    fn test_empty_e_nagual() {
        let e_nagual = ENagual::new("test");
        let context = ContextBuilder::new(Provider::Generic)
            .with_e_nagual(&e_nagual)
            .build();

        // Should still produce valid output
        assert!(context.content.contains("Learned Context"));
    }
}
