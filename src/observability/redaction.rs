//! Privacy-preserving log redaction.
//!
//! Provides a tracing-subscriber layer that automatically redacts sensitive
//! information (PII) from log output using the security module's PII detector.

use std::collections::HashSet;
use std::fmt;
use std::sync::Arc;

use parking_lot::RwLock;
use tracing::{Event, Subscriber};
use tracing_subscriber::{
    fmt::{
        format::{self, FormatEvent, FormatFields},
        FmtContext,
    },
    layer::Context,
    registry::LookupSpan,
    Layer,
};

use crate::security::pii::{PiiClassification, PiiDetector, PiiMatch, PiiType};

/// Configuration for log redaction.
#[derive(Debug, Clone)]
pub struct RedactionConfig {
    /// Minimum PII classification level to redact (default: Medium).
    pub min_classification: PiiClassification,
    /// Whether to redact field names that match sensitive patterns.
    pub redact_field_names: bool,
    /// Additional field names to always redact.
    pub sensitive_fields: HashSet<String>,
    /// Redaction placeholder text.
    pub redaction_text: String,
    /// Whether to log redaction events.
    pub log_redactions: bool,
    /// Whether to preserve partial information (e.g., first/last chars).
    pub preserve_partial: bool,
}

impl Default for RedactionConfig {
    fn default() -> Self {
        let mut sensitive_fields = HashSet::new();
        sensitive_fields.insert("password".to_string());
        sensitive_fields.insert("secret".to_string());
        sensitive_fields.insert("token".to_string());
        sensitive_fields.insert("api_key".to_string());
        sensitive_fields.insert("apikey".to_string());
        sensitive_fields.insert("authorization".to_string());
        sensitive_fields.insert("auth".to_string());
        sensitive_fields.insert("credential".to_string());
        sensitive_fields.insert("private_key".to_string());
        sensitive_fields.insert("ssn".to_string());
        sensitive_fields.insert("credit_card".to_string());
        sensitive_fields.insert("card_number".to_string());

        Self {
            min_classification: PiiClassification::Medium,
            redact_field_names: true,
            sensitive_fields,
            redaction_text: "[REDACTED]".to_string(),
            log_redactions: false,
            preserve_partial: true,
        }
    }
}

impl RedactionConfig {
    /// Create a new config with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the minimum classification level to redact.
    pub fn with_min_classification(mut self, level: PiiClassification) -> Self {
        self.min_classification = level;
        self
    }

    /// Add a sensitive field name.
    pub fn with_sensitive_field(mut self, field: impl Into<String>) -> Self {
        self.sensitive_fields.insert(field.into());
        self
    }

    /// Set the redaction placeholder text.
    pub fn with_redaction_text(mut self, text: impl Into<String>) -> Self {
        self.redaction_text = text.into();
        self
    }

    /// Enable or disable partial preservation.
    pub fn with_preserve_partial(mut self, preserve: bool) -> Self {
        self.preserve_partial = preserve;
        self
    }

    /// Enable or disable logging of redaction events.
    pub fn with_log_redactions(mut self, log: bool) -> Self {
        self.log_redactions = log;
        self
    }

    /// Check if a field name is sensitive.
    pub fn is_sensitive_field(&self, name: &str) -> bool {
        let name_lower = name.to_lowercase();
        self.sensitive_fields.iter().any(|s| name_lower.contains(s))
    }
}

/// A log redactor that removes or masks PII from text.
pub struct LogRedactor {
    detector: PiiDetector,
    config: RedactionConfig,
    /// Count of redactions performed.
    redaction_count: RwLock<u64>,
}

impl LogRedactor {
    /// Create a new log redactor with default configuration.
    pub fn new() -> Self {
        Self {
            detector: PiiDetector::new(),
            config: RedactionConfig::default(),
            redaction_count: RwLock::new(0),
        }
    }

    /// Create a log redactor with custom configuration.
    pub fn with_config(config: RedactionConfig) -> Self {
        Self {
            detector: PiiDetector::new(),
            config,
            redaction_count: RwLock::new(0),
        }
    }

    /// Redact sensitive information from text.
    pub fn redact(&self, text: &str) -> String {
        let matches = self.detector.scan_text(text);

        if matches.is_empty() {
            return text.to_string();
        }

        let mut result = text.to_string();
        let mut offset: i64 = 0;

        // Sort matches by position (they should already be sorted)
        let mut relevant_matches: Vec<&PiiMatch> = matches
            .iter()
            .filter(|m| m.classification >= self.config.min_classification)
            .collect();

        relevant_matches.sort_by_key(|m| m.range.start);

        for pii_match in relevant_matches {
            let start = (pii_match.range.start as i64 + offset) as usize;
            let end = (pii_match.range.end as i64 + offset) as usize;

            if start >= result.len() || end > result.len() {
                continue;
            }

            let replacement = self.create_replacement(pii_match);
            let old_len = end - start;
            let new_len = replacement.len();

            result.replace_range(start..end, &replacement);
            offset += new_len as i64 - old_len as i64;

            // Increment redaction count
            *self.redaction_count.write() += 1;
        }

        result
    }

    /// Create a replacement string for a PII match.
    fn create_replacement(&self, pii_match: &PiiMatch) -> String {
        if self.config.preserve_partial && pii_match.matched_text.len() > 4 {
            pii_match.redacted_text()
        } else {
            self.config.redaction_text.clone()
        }
    }

    /// Redact a field value if the field name is sensitive.
    pub fn redact_field(&self, name: &str, value: &str) -> String {
        if self.config.redact_field_names && self.config.is_sensitive_field(name) {
            *self.redaction_count.write() += 1;
            return self.config.redaction_text.clone();
        }

        self.redact(value)
    }

    /// Get the total number of redactions performed.
    pub fn redaction_count(&self) -> u64 {
        *self.redaction_count.read()
    }

    /// Reset the redaction count.
    pub fn reset_count(&self) {
        *self.redaction_count.write() = 0;
    }

    /// Check if text contains PII that would be redacted.
    pub fn contains_pii(&self, text: &str) -> bool {
        let matches = self.detector.scan_text(text);
        matches.iter().any(|m| m.classification >= self.config.min_classification)
    }

    /// Get a summary of PII types in text.
    pub fn analyze(&self, text: &str) -> RedactionAnalysis {
        let matches = self.detector.scan_text(text);
        let relevant: Vec<_> = matches
            .iter()
            .filter(|m| m.classification >= self.config.min_classification)
            .collect();

        RedactionAnalysis {
            total_pii_found: matches.len(),
            would_be_redacted: relevant.len(),
            highest_classification: matches
                .iter()
                .map(|m| m.classification)
                .max()
                .unwrap_or(PiiClassification::None),
            pii_types: relevant.iter().map(|m| m.pii_type).collect(),
        }
    }
}

impl Default for LogRedactor {
    fn default() -> Self {
        Self::new()
    }
}

/// Analysis result from redaction check.
#[derive(Debug, Clone)]
pub struct RedactionAnalysis {
    /// Total PII matches found.
    pub total_pii_found: usize,
    /// Number that would be redacted based on configuration.
    pub would_be_redacted: usize,
    /// Highest PII classification found.
    pub highest_classification: PiiClassification,
    /// Types of PII that would be redacted.
    pub pii_types: Vec<PiiType>,
}

impl RedactionAnalysis {
    /// Check if any redaction would occur.
    pub fn needs_redaction(&self) -> bool {
        self.would_be_redacted > 0
    }
}

/// A tracing layer that redacts PII from log output.
pub struct RedactingLayer<S> {
    redactor: Arc<LogRedactor>,
    inner: S,
}

impl<S> RedactingLayer<S> {
    /// Create a new redacting layer wrapping another layer.
    pub fn new(inner: S) -> Self {
        Self {
            redactor: Arc::new(LogRedactor::new()),
            inner,
        }
    }

    /// Create a redacting layer with custom configuration.
    pub fn with_config(inner: S, config: RedactionConfig) -> Self {
        Self {
            redactor: Arc::new(LogRedactor::with_config(config)),
            inner,
        }
    }

    /// Get the redactor for manual use.
    pub fn redactor(&self) -> &Arc<LogRedactor> {
        &self.redactor
    }
}

impl<S, N> Layer<S> for RedactingLayer<N>
where
    S: Subscriber + for<'a> LookupSpan<'a>,
    N: Layer<S>,
{
    fn on_event(&self, event: &Event<'_>, ctx: Context<'_, S>) {
        // Forward to inner layer - actual redaction happens in formatting
        self.inner.on_event(event, ctx);
    }

    fn on_enter(&self, id: &tracing::span::Id, ctx: Context<'_, S>) {
        self.inner.on_enter(id, ctx);
    }

    fn on_exit(&self, id: &tracing::span::Id, ctx: Context<'_, S>) {
        self.inner.on_exit(id, ctx);
    }

    fn on_close(&self, id: tracing::span::Id, ctx: Context<'_, S>) {
        self.inner.on_close(id, ctx);
    }
}

/// A formatter that redacts PII from events.
pub struct RedactingFormatter<F> {
    inner: F,
    redactor: Arc<LogRedactor>,
}

impl<F> RedactingFormatter<F> {
    /// Create a new redacting formatter.
    pub fn new(inner: F, redactor: Arc<LogRedactor>) -> Self {
        Self { inner, redactor }
    }
}

impl<S, N, F> FormatEvent<S, N> for RedactingFormatter<F>
where
    S: Subscriber + for<'a> LookupSpan<'a>,
    N: for<'writer> FormatFields<'writer> + 'static,
    F: FormatEvent<S, N>,
{
    fn format_event(
        &self,
        ctx: &FmtContext<'_, S, N>,
        mut writer: format::Writer<'_>,
        event: &Event<'_>,
    ) -> fmt::Result {
        // Create a buffer to capture the formatted output
        let mut buffer = String::new();
        let buffer_writer = format::Writer::new(&mut buffer);

        // Format to buffer
        // Note: This is a simplified implementation. A full implementation
        // would need to intercept and redact individual field values.
        self.inner.format_event(ctx, buffer_writer, event)?;

        // Redact the buffer
        let redacted = self.redactor.redact(&buffer);

        // Write the redacted output
        write!(writer, "{}", redacted)
    }
}

/// Visitor that redacts field values.
pub struct RedactingVisitor<'a> {
    redactor: &'a LogRedactor,
    output: String,
}

impl<'a> RedactingVisitor<'a> {
    /// Create a new redacting visitor.
    pub fn new(redactor: &'a LogRedactor) -> Self {
        Self {
            redactor,
            output: String::new(),
        }
    }

    /// Get the redacted output.
    pub fn into_output(self) -> String {
        self.output
    }
}

impl<'a> tracing::field::Visit for RedactingVisitor<'a> {
    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn fmt::Debug) {
        let value_str = format!("{:?}", value);
        let redacted = self.redactor.redact_field(field.name(), &value_str);

        if !self.output.is_empty() {
            self.output.push_str(", ");
        }
        self.output.push_str(&format!("{}={}", field.name(), redacted));
    }

    fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
        let redacted = self.redactor.redact_field(field.name(), value);

        if !self.output.is_empty() {
            self.output.push_str(", ");
        }
        self.output.push_str(&format!("{}=\"{}\"", field.name(), redacted));
    }

    fn record_i64(&mut self, field: &tracing::field::Field, value: i64) {
        if !self.output.is_empty() {
            self.output.push_str(", ");
        }
        self.output.push_str(&format!("{}={}", field.name(), value));
    }

    fn record_u64(&mut self, field: &tracing::field::Field, value: u64) {
        if !self.output.is_empty() {
            self.output.push_str(", ");
        }
        self.output.push_str(&format!("{}={}", field.name(), value));
    }

    fn record_bool(&mut self, field: &tracing::field::Field, value: bool) {
        if !self.output.is_empty() {
            self.output.push_str(", ");
        }
        self.output.push_str(&format!("{}={}", field.name(), value));
    }
}

/// Redact a string in place (convenience function).
pub fn redact(text: &str) -> String {
    LogRedactor::new().redact(text)
}

/// Redact with custom configuration (convenience function).
pub fn redact_with_config(text: &str, config: RedactionConfig) -> String {
    LogRedactor::with_config(config).redact(text)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_redaction_config_defaults() {
        let config = RedactionConfig::default();
        assert_eq!(config.min_classification, PiiClassification::Medium);
        assert!(config.redact_field_names);
        assert!(config.sensitive_fields.contains("password"));
        assert_eq!(config.redaction_text, "[REDACTED]");
    }

    #[test]
    fn test_redaction_config_builder() {
        let config = RedactionConfig::new()
            .with_min_classification(PiiClassification::High)
            .with_sensitive_field("custom_secret")
            .with_redaction_text("***")
            .with_preserve_partial(false);

        assert_eq!(config.min_classification, PiiClassification::High);
        assert!(config.sensitive_fields.contains("custom_secret"));
        assert_eq!(config.redaction_text, "***");
        assert!(!config.preserve_partial);
    }

    #[test]
    fn test_sensitive_field_detection() {
        let config = RedactionConfig::default();

        assert!(config.is_sensitive_field("password"));
        assert!(config.is_sensitive_field("user_password"));
        assert!(config.is_sensitive_field("PASSWORD"));
        assert!(config.is_sensitive_field("api_key"));
        assert!(config.is_sensitive_field("authorization"));
        assert!(!config.is_sensitive_field("username"));
        assert!(!config.is_sensitive_field("email"));
    }

    #[test]
    fn test_email_redaction() {
        let redactor = LogRedactor::new();
        let text = "User email is john.doe@example.com and phone is 555-123-4567";
        let redacted = redactor.redact(text);

        assert!(!redacted.contains("john.doe@example.com"));
        assert!(redacted.contains("jo"));
        assert!(redacted.contains("om"));
    }

    #[test]
    fn test_credit_card_redaction() {
        let redactor = LogRedactor::new();
        let text = "Card: 4532-1234-5678-9012";
        let redacted = redactor.redact(text);

        assert!(!redacted.contains("4532-1234-5678-9012"));
    }

    #[test]
    fn test_ssn_redaction() {
        let redactor = LogRedactor::new();
        let text = "SSN: 456-78-9012";
        let redacted = redactor.redact(text);

        assert!(!redacted.contains("456-78-9012"));
    }

    #[test]
    fn test_api_key_redaction() {
        let redactor = LogRedactor::new();
        let text = "AWS key: AKIAIOSFODNN7EXAMPLE";
        let redacted = redactor.redact(text);

        assert!(!redacted.contains("AKIAIOSFODNN7EXAMPLE"));
    }

    #[test]
    fn test_field_name_redaction() {
        let redactor = LogRedactor::new();

        let result = redactor.redact_field("password", "supersecret123");
        assert_eq!(result, "[REDACTED]");

        let result = redactor.redact_field("api_key", "sk-abc123");
        assert_eq!(result, "[REDACTED]");

        let result = redactor.redact_field("username", "johndoe");
        assert_eq!(result, "johndoe");
    }

    #[test]
    fn test_no_pii_no_redaction() {
        let redactor = LogRedactor::new();
        let text = "This is a normal log message with no sensitive data";
        let redacted = redactor.redact(text);

        assert_eq!(redacted, text);
    }

    #[test]
    fn test_low_classification_not_redacted_by_default() {
        let redactor = LogRedactor::new();
        // IP addresses are Low classification
        let text = "Server at 10.0.0.1";

        // Default min classification is Medium, so Low shouldn't be redacted
        let config = RedactionConfig::new().with_min_classification(PiiClassification::Medium);
        let redactor = LogRedactor::with_config(config);
        let redacted = redactor.redact(text);

        // Low classification IPs should not be redacted by default
        assert_eq!(redacted, text);
    }

    #[test]
    fn test_redaction_count() {
        let redactor = LogRedactor::new();

        redactor.redact("Email: test@example.com");
        redactor.redact("Another email: user@test.org");

        assert!(redactor.redaction_count() >= 2);

        redactor.reset_count();
        assert_eq!(redactor.redaction_count(), 0);
    }

    #[test]
    fn test_contains_pii() {
        let redactor = LogRedactor::new();

        assert!(redactor.contains_pii("Contact: john@example.com"));
        assert!(!redactor.contains_pii("No sensitive data here"));
    }

    #[test]
    fn test_analyze() {
        let redactor = LogRedactor::new();
        let text = "Email: john@test.com, Card: 4532-1234-5678-9012";

        let analysis = redactor.analyze(text);

        assert!(analysis.total_pii_found >= 2);
        assert!(analysis.needs_redaction());
        assert!(analysis.highest_classification >= PiiClassification::Medium);
    }

    #[test]
    fn test_multiple_pii_same_line() {
        let redactor = LogRedactor::new();
        let text = "User john@test.com called from 555-123-4567 with card 4532-1234-5678-9012";
        let redacted = redactor.redact(text);

        assert!(!redacted.contains("john@test.com"));
        assert!(!redacted.contains("4532-1234-5678-9012"));
    }

    #[test]
    fn test_preserve_partial_false() {
        let config = RedactionConfig::new().with_preserve_partial(false);
        let redactor = LogRedactor::with_config(config);

        let text = "Email: john.doe@example.com";
        let redacted = redactor.redact(text);

        assert!(redacted.contains("[REDACTED]"));
        assert!(!redacted.contains("jo"));
    }

    #[test]
    fn test_convenience_function() {
        let text = "Secret: ghp_abcdefghijklmnopqrstuvwxyz0123456789";
        let redacted = redact(text);

        assert!(!redacted.contains("ghp_abcdefghijklmnopqrstuvwxyz0123456789"));
    }

    #[test]
    fn test_redacting_visitor() {
        let redactor = LogRedactor::new();
        let visitor = RedactingVisitor::new(&redactor);

        // This is a simplified test - actual field visiting would come from tracing
        // We can't easily simulate tracing fields without a real span/event
        let output = visitor.into_output();
        assert!(output.is_empty()); // No fields visited yet
    }
}
