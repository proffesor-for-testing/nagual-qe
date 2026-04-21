//! Prompt formatting for patterns.
//!
//! This module converts patterns into LLM-ready format with:
//! - Structured problem/solution/context sections
//! - Confidence and reliability indicators
//! - Token-aware truncation
//! - Multiple formatting styles
//!
//! # Example
//!
//! ```ignore
//! use nagual::reasoning_bank::{format_for_prompt, FormatConfig};
//!
//! let formatted = format_for_prompt(&patterns, &FormatConfig::default())?;
//! // Use in system prompt:
//! // "Here are relevant patterns:\n{formatted}"
//! ```

use serde::{Deserialize, Serialize};

use super::{Pattern, ReasoningBankResult, ScoredPattern};

/// Configuration for prompt formatting.
#[derive(Debug, Clone)]
pub struct FormatConfig {
    /// Maximum number of tokens/characters to output.
    pub max_tokens: usize,

    /// Whether to include confidence indicators.
    pub include_confidence: bool,

    /// Whether to include reliability scores.
    pub include_reliability: bool,

    /// Whether to include context.
    pub include_context: bool,

    /// Whether to include domain information.
    pub include_domain: bool,

    /// Whether to include similarity scores.
    pub include_similarity: bool,

    /// Whether to include critiques.
    pub include_critique: bool,

    /// Truncation strategy when token limit is exceeded.
    pub truncation: TruncationStrategy,

    /// Format style for output.
    pub style: FormatStyle,

    /// Separator between patterns.
    pub pattern_separator: String,

    /// Characters per token estimate (for rough token counting).
    pub chars_per_token: f32,
}

impl Default for FormatConfig {
    fn default() -> Self {
        Self {
            max_tokens: 2000,
            include_confidence: true,
            include_reliability: true,
            include_context: true,
            include_domain: true,
            include_similarity: false,
            include_critique: true,
            truncation: TruncationStrategy::default(),
            style: FormatStyle::default(),
            pattern_separator: "\n---\n".to_string(),
            chars_per_token: 4.0, // Rough estimate for English text
        }
    }
}

impl FormatConfig {
    /// Create a minimal config (problem + solution only).
    pub fn minimal() -> Self {
        Self {
            include_confidence: false,
            include_reliability: false,
            include_context: false,
            include_domain: false,
            include_similarity: false,
            include_critique: false,
            ..Default::default()
        }
    }

    /// Create a verbose config (all information).
    pub fn verbose() -> Self {
        Self {
            include_similarity: true,
            ..Default::default()
        }
    }

    /// Set the maximum tokens.
    pub fn with_max_tokens(mut self, max_tokens: usize) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    /// Set the truncation strategy.
    pub fn with_truncation(mut self, truncation: TruncationStrategy) -> Self {
        self.truncation = truncation;
        self
    }

    /// Set the format style.
    pub fn with_style(mut self, style: FormatStyle) -> Self {
        self.style = style;
        self
    }

    /// Estimate token count from character count.
    fn estimate_tokens(&self, chars: usize) -> usize {
        (chars as f32 / self.chars_per_token).ceil() as usize
    }

    /// Estimate character budget from token limit.
    fn char_budget(&self) -> usize {
        (self.max_tokens as f32 * self.chars_per_token) as usize
    }
}

/// Strategy for handling token limit overflow.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub enum TruncationStrategy {
    /// Truncate individual patterns to fit more
    #[default]
    TruncatePatterns,

    /// Drop lower-ranked patterns to stay within limit
    DropPatterns,

    /// Hard truncate the entire output
    HardTruncate,

    /// No truncation (may exceed limit)
    None,
}

/// Output format style.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub enum FormatStyle {
    /// Markdown format with headers
    #[default]
    Markdown,

    /// Plain text format
    PlainText,

    /// XML-style tags for structured parsing
    Xml,

    /// JSON format
    Json,

    /// Compact single-line per pattern
    Compact,
}

/// A formatted pattern ready for prompt injection.
#[derive(Debug, Clone)]
pub struct FormattedPattern {
    /// The formatted text.
    pub text: String,

    /// Estimated token count.
    pub estimated_tokens: usize,

    /// Whether the pattern was truncated.
    pub truncated: bool,

    /// Original pattern ID.
    pub pattern_id: String,
}

/// Prompt formatter for converting patterns to LLM-ready format.
pub struct PromptFormatter {
    config: FormatConfig,
}

impl PromptFormatter {
    /// Create a new formatter with the given configuration.
    pub fn new(config: FormatConfig) -> Self {
        Self { config }
    }

    /// Create a formatter with default configuration.
    pub fn default_formatter() -> Self {
        Self::new(FormatConfig::default())
    }

    /// Format a single pattern.
    pub fn format_pattern(&self, pattern: &Pattern) -> FormattedPattern {
        self.format_pattern_with_score(pattern, None)
    }

    /// Format a scored pattern.
    pub fn format_scored_pattern(&self, scored: &ScoredPattern) -> FormattedPattern {
        self.format_pattern_with_score(&scored.pattern, Some(scored.similarity))
    }

    /// Format a pattern with optional similarity score.
    fn format_pattern_with_score(
        &self,
        pattern: &Pattern,
        similarity: Option<f32>,
    ) -> FormattedPattern {
        let text = match self.config.style {
            FormatStyle::Markdown => self.format_markdown(pattern, similarity),
            FormatStyle::PlainText => self.format_plain_text(pattern, similarity),
            FormatStyle::Xml => self.format_xml(pattern, similarity),
            FormatStyle::Json => self.format_json(pattern, similarity),
            FormatStyle::Compact => self.format_compact(pattern, similarity),
        };

        let estimated_tokens = self.config.estimate_tokens(text.len());

        FormattedPattern {
            text,
            estimated_tokens,
            truncated: false,
            pattern_id: pattern.id.clone(),
        }
    }

    /// Format pattern in Markdown style.
    fn format_markdown(&self, pattern: &Pattern, similarity: Option<f32>) -> String {
        let mut parts = Vec::new();

        // Header with optional indicators
        let mut header = String::from("### Pattern");
        if self.config.include_domain {
            header.push_str(&format!(" [{}]", pattern.domain));
        }
        parts.push(header);

        // Indicators line
        let mut indicators = Vec::new();
        if self.config.include_confidence {
            indicators.push(format!(
                "Confidence: {}",
                confidence_indicator(pattern.confidence)
            ));
        }
        if self.config.include_reliability {
            indicators.push(format!(
                "Reliability: {}",
                confidence_indicator(pattern.reliability_score())
            ));
        }
        if self.config.include_similarity {
            if let Some(sim) = similarity {
                indicators.push(format!("Match: {:.0}%", sim * 100.0));
            }
        }
        if !indicators.is_empty() {
            parts.push(format!("*{}*", indicators.join(" | ")));
        }

        // Problem
        parts.push(format!("**Problem:** {}", pattern.problem));

        // Solution
        parts.push(format!("**Solution:** {}", pattern.solution));

        // Context
        if self.config.include_context {
            if let Some(ref ctx) = pattern.context {
                parts.push(format!("**Context:** {}", ctx));
            }
        }

        // Critique
        if self.config.include_critique {
            if let Some(ref critique) = pattern.critique {
                parts.push(format!("**Notes:** {}", critique));
            }
        }

        parts.join("\n")
    }

    /// Format pattern in plain text style.
    fn format_plain_text(&self, pattern: &Pattern, similarity: Option<f32>) -> String {
        let mut parts = Vec::new();

        // Header
        let mut header = String::from("PATTERN");
        if self.config.include_domain {
            header.push_str(&format!(" [Domain: {}]", pattern.domain));
        }
        parts.push(header);

        // Indicators
        if self.config.include_confidence {
            parts.push(format!(
                "  Confidence: {}",
                confidence_indicator(pattern.confidence)
            ));
        }
        if self.config.include_reliability {
            parts.push(format!(
                "  Reliability: {}",
                confidence_indicator(pattern.reliability_score())
            ));
        }
        if self.config.include_similarity {
            if let Some(sim) = similarity {
                parts.push(format!("  Match: {:.0}%", sim * 100.0));
            }
        }

        // Problem
        parts.push(format!("  Problem: {}", pattern.problem));

        // Solution
        parts.push(format!("  Solution: {}", pattern.solution));

        // Context
        if self.config.include_context {
            if let Some(ref ctx) = pattern.context {
                parts.push(format!("  Context: {}", ctx));
            }
        }

        // Critique
        if self.config.include_critique {
            if let Some(ref critique) = pattern.critique {
                parts.push(format!("  Notes: {}", critique));
            }
        }

        parts.join("\n")
    }

    /// Format pattern in XML style.
    fn format_xml(&self, pattern: &Pattern, similarity: Option<f32>) -> String {
        let mut parts = Vec::new();

        parts.push("<pattern>".to_string());

        if self.config.include_domain {
            parts.push(format!("  <domain>{}</domain>", escape_xml(&pattern.domain)));
        }

        if self.config.include_confidence {
            parts.push(format!("  <confidence>{:.2}</confidence>", pattern.confidence));
        }

        if self.config.include_reliability {
            parts.push(format!(
                "  <reliability>{:.2}</reliability>",
                pattern.reliability_score()
            ));
        }

        if self.config.include_similarity {
            if let Some(sim) = similarity {
                parts.push(format!("  <similarity>{:.2}</similarity>", sim));
            }
        }

        parts.push(format!(
            "  <problem>{}</problem>",
            escape_xml(&pattern.problem)
        ));
        parts.push(format!(
            "  <solution>{}</solution>",
            escape_xml(&pattern.solution)
        ));

        if self.config.include_context {
            if let Some(ref ctx) = pattern.context {
                parts.push(format!("  <context>{}</context>", escape_xml(ctx)));
            }
        }

        if self.config.include_critique {
            if let Some(ref critique) = pattern.critique {
                parts.push(format!("  <notes>{}</notes>", escape_xml(critique)));
            }
        }

        parts.push("</pattern>".to_string());

        parts.join("\n")
    }

    /// Format pattern in JSON style.
    fn format_json(&self, pattern: &Pattern, similarity: Option<f32>) -> String {
        let mut obj = serde_json::Map::new();

        if self.config.include_domain {
            obj.insert(
                "domain".to_string(),
                serde_json::Value::String(pattern.domain.clone()),
            );
        }

        if self.config.include_confidence {
            obj.insert(
                "confidence".to_string(),
                serde_json::Value::Number(
                    serde_json::Number::from_f64(pattern.confidence as f64).unwrap(),
                ),
            );
        }

        if self.config.include_reliability {
            obj.insert(
                "reliability".to_string(),
                serde_json::Value::Number(
                    serde_json::Number::from_f64(pattern.reliability_score() as f64).unwrap(),
                ),
            );
        }

        if self.config.include_similarity {
            if let Some(sim) = similarity {
                obj.insert(
                    "similarity".to_string(),
                    serde_json::Value::Number(serde_json::Number::from_f64(sim as f64).unwrap()),
                );
            }
        }

        obj.insert(
            "problem".to_string(),
            serde_json::Value::String(pattern.problem.clone()),
        );
        obj.insert(
            "solution".to_string(),
            serde_json::Value::String(pattern.solution.clone()),
        );

        if self.config.include_context {
            if let Some(ref ctx) = pattern.context {
                obj.insert(
                    "context".to_string(),
                    serde_json::Value::String(ctx.clone()),
                );
            }
        }

        if self.config.include_critique {
            if let Some(ref critique) = pattern.critique {
                obj.insert(
                    "notes".to_string(),
                    serde_json::Value::String(critique.clone()),
                );
            }
        }

        serde_json::to_string_pretty(&obj).unwrap_or_default()
    }

    /// Format pattern in compact style.
    fn format_compact(&self, pattern: &Pattern, similarity: Option<f32>) -> String {
        let mut parts = Vec::new();

        // Domain if included
        if self.config.include_domain {
            parts.push(format!("[{}]", pattern.domain));
        }

        // Indicators
        if self.config.include_confidence {
            parts.push(format!("({})", confidence_indicator_short(pattern.confidence)));
        }

        if self.config.include_similarity {
            if let Some(sim) = similarity {
                parts.push(format!("{:.0}%", sim * 100.0));
            }
        }

        // Problem and solution
        parts.push(format!("Q: {} | A: {}", pattern.problem, pattern.solution));

        parts.join(" ")
    }

    /// Format multiple patterns.
    pub fn format_patterns(&self, patterns: &[Pattern]) -> ReasoningBankResult<String> {
        let scored: Vec<ScoredPattern> = patterns
            .iter()
            .map(|p| ScoredPattern {
                pattern: p.clone(),
                similarity: 0.0,
                final_score: 0.0,
                factor_scores: super::retrieval::FactorScores::default(),
            })
            .collect();
        self.format_scored_patterns(&scored)
    }

    /// Format multiple scored patterns with token limit handling.
    pub fn format_scored_patterns(
        &self,
        scored_patterns: &[ScoredPattern],
    ) -> ReasoningBankResult<String> {
        if scored_patterns.is_empty() {
            return Ok(String::new());
        }

        let char_budget = self.config.char_budget();
        let separator_len = self.config.pattern_separator.len();
        let mut formatted_parts: Vec<String> = Vec::new();
        let mut total_chars = 0;

        for scored in scored_patterns {
            let formatted = self.format_scored_pattern(scored);

            match self.config.truncation {
                TruncationStrategy::TruncatePatterns => {
                    let available = char_budget.saturating_sub(total_chars + separator_len);
                    if available == 0 {
                        break;
                    }
                    let text = if formatted.text.len() > available {
                        truncate_text(&formatted.text, available)
                    } else {
                        formatted.text
                    };
                    total_chars += text.len() + separator_len;
                    formatted_parts.push(text);
                }
                TruncationStrategy::DropPatterns => {
                    let new_total = total_chars + formatted.text.len() + separator_len;
                    if new_total > char_budget {
                        break;
                    }
                    total_chars = new_total;
                    formatted_parts.push(formatted.text);
                }
                TruncationStrategy::HardTruncate | TruncationStrategy::None => {
                    total_chars += formatted.text.len() + separator_len;
                    formatted_parts.push(formatted.text);
                }
            }
        }

        let mut result = formatted_parts.join(&self.config.pattern_separator);

        // Apply hard truncation if needed
        if matches!(self.config.truncation, TruncationStrategy::HardTruncate) {
            if result.len() > char_budget {
                result = truncate_text(&result, char_budget);
            }
        }

        Ok(result)
    }
}

/// Convenience function to format patterns with default config.
pub fn format_for_prompt(
    patterns: &[ScoredPattern],
    config: &FormatConfig,
) -> ReasoningBankResult<String> {
    let formatter = PromptFormatter::new(config.clone());
    formatter.format_scored_patterns(patterns)
}

/// Convert a confidence score to a human-readable indicator.
fn confidence_indicator(score: f32) -> &'static str {
    if score >= 0.9 {
        "Very High"
    } else if score >= 0.75 {
        "High"
    } else if score >= 0.5 {
        "Medium"
    } else if score >= 0.25 {
        "Low"
    } else {
        "Very Low"
    }
}

/// Convert a confidence score to a short indicator.
fn confidence_indicator_short(score: f32) -> &'static str {
    if score >= 0.9 {
        "+++"
    } else if score >= 0.75 {
        "++"
    } else if score >= 0.5 {
        "+"
    } else if score >= 0.25 {
        "-"
    } else {
        "--"
    }
}

/// Escape special characters for XML.
fn escape_xml(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

/// Truncate text to a maximum length, adding ellipsis.
fn truncate_text(text: &str, max_len: usize) -> String {
    if text.len() <= max_len {
        return text.to_string();
    }

    let ellipsis = "...";
    if max_len <= ellipsis.len() {
        return ellipsis[..max_len].to_string();
    }

    // Find a good break point (prefer word boundary)
    let target_len = max_len - ellipsis.len();
    let truncated = &text[..target_len];

    // Try to break at last space
    if let Some(last_space) = truncated.rfind(' ') {
        if last_space > target_len / 2 {
            return format!("{}{}", &text[..last_space], ellipsis);
        }
    }

    format!("{}{}", truncated, ellipsis)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reasoning_bank::retrieval::FactorScores;

    fn create_test_pattern() -> Pattern {
        Pattern::new(
            "How to handle database connection timeouts?",
            "Use connection pooling with retry logic and circuit breaker pattern",
            "database.resilience",
        )
        .with_context("Common in microservices architectures")
        .with_confidence(0.85)
        .with_critique("Works well for transient failures")
    }

    fn create_scored_pattern() -> ScoredPattern {
        ScoredPattern {
            pattern: create_test_pattern(),
            similarity: 0.92,
            final_score: 0.88,
            factor_scores: FactorScores::default(),
        }
    }

    #[test]
    fn test_format_config_default() {
        let config = FormatConfig::default();
        assert_eq!(config.max_tokens, 2000);
        assert!(config.include_confidence);
        assert!(config.include_context);
    }

    #[test]
    fn test_format_config_minimal() {
        let config = FormatConfig::minimal();
        assert!(!config.include_confidence);
        assert!(!config.include_context);
        assert!(!config.include_domain);
    }

    #[test]
    fn test_format_markdown() {
        let formatter = PromptFormatter::new(FormatConfig::default());
        let pattern = create_test_pattern();
        let formatted = formatter.format_pattern(&pattern);

        assert!(formatted.text.contains("### Pattern"));
        assert!(formatted.text.contains("[database.resilience]"));
        assert!(formatted.text.contains("**Problem:**"));
        assert!(formatted.text.contains("**Solution:**"));
        assert!(formatted.text.contains("Confidence: High"));
    }

    #[test]
    fn test_format_plain_text() {
        let config = FormatConfig::default().with_style(FormatStyle::PlainText);
        let formatter = PromptFormatter::new(config);
        let pattern = create_test_pattern();
        let formatted = formatter.format_pattern(&pattern);

        assert!(formatted.text.contains("PATTERN"));
        assert!(formatted.text.contains("Problem:"));
        assert!(formatted.text.contains("Solution:"));
    }

    #[test]
    fn test_format_xml() {
        let config = FormatConfig::default().with_style(FormatStyle::Xml);
        let formatter = PromptFormatter::new(config);
        let pattern = create_test_pattern();
        let formatted = formatter.format_pattern(&pattern);

        assert!(formatted.text.contains("<pattern>"));
        assert!(formatted.text.contains("</pattern>"));
        assert!(formatted.text.contains("<problem>"));
        assert!(formatted.text.contains("<solution>"));
    }

    #[test]
    fn test_format_json() {
        let config = FormatConfig::default().with_style(FormatStyle::Json);
        let formatter = PromptFormatter::new(config);
        let pattern = create_test_pattern();
        let formatted = formatter.format_pattern(&pattern);

        // Should be valid JSON
        let parsed: serde_json::Value = serde_json::from_str(&formatted.text).unwrap();
        assert!(parsed.get("problem").is_some());
        assert!(parsed.get("solution").is_some());
    }

    #[test]
    fn test_format_compact() {
        let config = FormatConfig::default().with_style(FormatStyle::Compact);
        let formatter = PromptFormatter::new(config);
        let pattern = create_test_pattern();
        let formatted = formatter.format_pattern(&pattern);

        assert!(formatted.text.contains("[database.resilience]"));
        assert!(formatted.text.contains("Q:"));
        assert!(formatted.text.contains("| A:"));
    }

    #[test]
    fn test_format_with_similarity() {
        let config = FormatConfig::verbose();
        let formatter = PromptFormatter::new(config);
        let scored = create_scored_pattern();
        let formatted = formatter.format_scored_pattern(&scored);

        assert!(formatted.text.contains("Match: 92%"));
    }

    #[test]
    fn test_format_multiple_patterns() {
        let formatter = PromptFormatter::default_formatter();
        let patterns = vec![create_scored_pattern(), create_scored_pattern()];

        let result = formatter.format_scored_patterns(&patterns).unwrap();

        // Should contain separator
        assert!(result.contains("---"));
        // Should have two patterns
        assert_eq!(result.matches("### Pattern").count(), 2);
    }

    #[test]
    fn test_truncation_drop_patterns() {
        let config = FormatConfig::default()
            .with_max_tokens(100) // Very small limit
            .with_truncation(TruncationStrategy::DropPatterns);
        let formatter = PromptFormatter::new(config);

        let patterns = vec![
            create_scored_pattern(),
            create_scored_pattern(),
            create_scored_pattern(),
        ];

        let result = formatter.format_scored_patterns(&patterns).unwrap();

        // Should have fewer patterns due to limit
        let pattern_count = result.matches("### Pattern").count();
        assert!(pattern_count < 3);
    }

    #[test]
    fn test_confidence_indicator() {
        assert_eq!(confidence_indicator(0.95), "Very High");
        assert_eq!(confidence_indicator(0.80), "High");
        assert_eq!(confidence_indicator(0.60), "Medium");
        assert_eq!(confidence_indicator(0.30), "Low");
        assert_eq!(confidence_indicator(0.10), "Very Low");
    }

    #[test]
    fn test_escape_xml() {
        assert_eq!(escape_xml("a < b"), "a &lt; b");
        assert_eq!(escape_xml("a > b"), "a &gt; b");
        assert_eq!(escape_xml("a & b"), "a &amp; b");
        assert_eq!(escape_xml("\"quoted\""), "&quot;quoted&quot;");
    }

    #[test]
    fn test_truncate_text() {
        let text = "This is a long text that needs truncation";
        let truncated = truncate_text(text, 20);

        assert!(truncated.len() <= 20);
        assert!(truncated.ends_with("..."));
    }

    #[test]
    fn test_truncate_text_at_word_boundary() {
        let text = "Hello world this is a test";
        let truncated = truncate_text(text, 15);

        // Should break at word boundary
        assert!(truncated.ends_with("..."));
        assert!(!truncated.contains("thi")); // Should not cut mid-word
    }

    #[test]
    fn test_format_for_prompt_convenience() {
        let patterns = vec![create_scored_pattern()];
        let config = FormatConfig::default();

        let result = format_for_prompt(&patterns, &config).unwrap();

        assert!(!result.is_empty());
        assert!(result.contains("### Pattern"));
    }

    #[test]
    fn test_estimated_tokens() {
        let config = FormatConfig::default();
        let chars = 400;
        let estimated = config.estimate_tokens(chars);

        // With 4 chars per token, 400 chars = 100 tokens
        assert_eq!(estimated, 100);
    }

    #[test]
    fn test_empty_patterns() {
        let formatter = PromptFormatter::default_formatter();
        let patterns: Vec<ScoredPattern> = vec![];

        let result = formatter.format_scored_patterns(&patterns).unwrap();

        assert!(result.is_empty());
    }
}
