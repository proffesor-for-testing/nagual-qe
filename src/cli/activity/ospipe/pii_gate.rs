//! PII Safety Gate for the OSpipe pipeline.
//!
//! Provides a configurable gate that can reject, redact, warn, or allow
//! content based on PII detection results.

use serde::{Deserialize, Serialize};

use crate::security::pii::{PiiClassification, PiiDetector, PiiMatch, RedactionResult};

/// Policy for handling detected PII.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum PiiPolicy {
    /// Reject content containing PII above a threshold.
    Reject,
    /// Redact PII with tokens (e.g., [EMAIL], [SSN]).
    #[default]
    Redact,
    /// Log a warning but allow content through unchanged.
    Warn,
    /// Allow all content through without modification.
    Allow,
}

impl PiiPolicy {
    /// Parse a policy from a string.
    pub fn from_str(s: &str) -> Option<Self> {
        match s.trim().to_lowercase().as_str() {
            "reject" => Some(PiiPolicy::Reject),
            "redact" => Some(PiiPolicy::Redact),
            "warn" => Some(PiiPolicy::Warn),
            "allow" => Some(PiiPolicy::Allow),
            _ => None,
        }
    }
}

impl std::fmt::Display for PiiPolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PiiPolicy::Reject => write!(f, "reject"),
            PiiPolicy::Redact => write!(f, "redact"),
            PiiPolicy::Warn => write!(f, "warn"),
            PiiPolicy::Allow => write!(f, "allow"),
        }
    }
}

/// Result of processing content through the PII gate.
#[derive(Debug, Clone)]
pub enum PiiGateResult {
    /// Content was allowed through unchanged (no PII or policy is Allow).
    Allowed {
        content: String,
        had_pii: bool,
        classification: PiiClassification,
    },
    /// Content was redacted (PII replaced with tokens).
    Redacted {
        content: String,
        redaction_result: RedactionResult,
    },
    /// A warning was issued but content passed through unchanged.
    Warned {
        content: String,
        matches: Vec<PiiMatch>,
        classification: PiiClassification,
    },
    /// Content was rejected due to PII above threshold.
    Rejected {
        reason: String,
        matches: Vec<PiiMatch>,
        classification: PiiClassification,
    },
}

impl PiiGateResult {
    /// Check if the content was accepted (not rejected).
    pub fn is_accepted(&self) -> bool {
        !matches!(self, PiiGateResult::Rejected { .. })
    }

    /// Get the processed content, if any.
    pub fn content(&self) -> Option<&str> {
        match self {
            PiiGateResult::Allowed { content, .. } => Some(content),
            PiiGateResult::Redacted { content, .. } => Some(content),
            PiiGateResult::Warned { content, .. } => Some(content),
            PiiGateResult::Rejected { .. } => None,
        }
    }

    /// Get the highest PII classification found.
    pub fn classification(&self) -> PiiClassification {
        match self {
            PiiGateResult::Allowed { classification, .. } => *classification,
            PiiGateResult::Redacted { redaction_result, .. } => {
                redaction_result.highest_classification
            }
            PiiGateResult::Warned { classification, .. } => *classification,
            PiiGateResult::Rejected { classification, .. } => *classification,
        }
    }

    /// Check if the result indicates PII was found.
    pub fn had_pii(&self) -> bool {
        match self {
            PiiGateResult::Allowed { had_pii, .. } => *had_pii,
            PiiGateResult::Redacted { redaction_result, .. } => redaction_result.was_redacted(),
            PiiGateResult::Warned { matches, .. } => !matches.is_empty(),
            PiiGateResult::Rejected { matches, .. } => !matches.is_empty(),
        }
    }
}

/// PII Safety Gate that applies a configured policy to incoming content.
pub struct PiiGate {
    /// The PII detector instance.
    detector: PiiDetector,
    /// The policy to apply.
    policy: PiiPolicy,
    /// Minimum classification level to trigger the policy.
    /// Content with PII below this level is allowed through.
    rejection_threshold: PiiClassification,
}

impl Default for PiiGate {
    fn default() -> Self {
        Self::new(PiiPolicy::Redact)
    }
}

impl PiiGate {
    /// Create a new PII gate with the specified policy.
    pub fn new(policy: PiiPolicy) -> Self {
        Self {
            detector: PiiDetector::new(),
            policy,
            rejection_threshold: PiiClassification::Medium,
        }
    }

    /// Create a new PII gate with a custom rejection threshold.
    pub fn with_threshold(policy: PiiPolicy, threshold: PiiClassification) -> Self {
        Self {
            detector: PiiDetector::new(),
            policy,
            rejection_threshold: threshold,
        }
    }

    /// Set the rejection threshold.
    pub fn set_threshold(&mut self, threshold: PiiClassification) {
        self.rejection_threshold = threshold;
    }

    /// Get the current policy.
    pub fn policy(&self) -> PiiPolicy {
        self.policy
    }

    /// Set the policy.
    pub fn set_policy(&mut self, policy: PiiPolicy) {
        self.policy = policy;
    }

    /// Process content through the PII gate.
    ///
    /// Returns a `PiiGateResult` indicating what action was taken.
    pub fn process(&self, content: &str) -> PiiGateResult {
        // Fast path: if policy is Allow, skip detection
        if self.policy == PiiPolicy::Allow {
            return PiiGateResult::Allowed {
                content: content.to_string(),
                had_pii: false,
                classification: PiiClassification::None,
            };
        }

        // Scan for PII
        let matches = self.detector.scan_text(content);

        // If no PII found, allow through
        if matches.is_empty() {
            return PiiGateResult::Allowed {
                content: content.to_string(),
                had_pii: false,
                classification: PiiClassification::None,
            };
        }

        // Determine highest classification
        let classification = matches
            .iter()
            .map(|m| m.classification)
            .max()
            .unwrap_or(PiiClassification::None);

        // If below threshold, allow through
        if classification < self.rejection_threshold {
            return PiiGateResult::Allowed {
                content: content.to_string(),
                had_pii: true,
                classification,
            };
        }

        // Apply policy
        match self.policy {
            PiiPolicy::Allow => PiiGateResult::Allowed {
                content: content.to_string(),
                had_pii: true,
                classification,
            },
            PiiPolicy::Warn => PiiGateResult::Warned {
                content: content.to_string(),
                matches,
                classification,
            },
            PiiPolicy::Redact => {
                let redaction_result = self.detector.redact(content);
                PiiGateResult::Redacted {
                    content: redaction_result.redacted_text.clone(),
                    redaction_result,
                }
            }
            PiiPolicy::Reject => PiiGateResult::Rejected {
                reason: format!(
                    "Content contains {} PII ({} matches)",
                    classification,
                    matches.len()
                ),
                matches,
                classification,
            },
        }
    }

    /// Quick check if content contains PII above threshold.
    pub fn would_reject(&self, content: &str) -> bool {
        if self.policy != PiiPolicy::Reject {
            return false;
        }

        let classification = self.detector.classify(content);
        classification >= self.rejection_threshold
    }

    /// Get a reference to the underlying detector.
    pub fn detector(&self) -> &PiiDetector {
        &self.detector
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pii_policy_from_str() {
        assert_eq!(PiiPolicy::from_str("reject"), Some(PiiPolicy::Reject));
        assert_eq!(PiiPolicy::from_str("redact"), Some(PiiPolicy::Redact));
        assert_eq!(PiiPolicy::from_str("warn"), Some(PiiPolicy::Warn));
        assert_eq!(PiiPolicy::from_str("allow"), Some(PiiPolicy::Allow));
        assert_eq!(PiiPolicy::from_str("REDACT"), Some(PiiPolicy::Redact));
        assert_eq!(PiiPolicy::from_str("invalid"), None);
    }

    #[test]
    fn test_gate_allow_no_pii() {
        let gate = PiiGate::new(PiiPolicy::Reject);
        let result = gate.process("Hello, this is a normal message.");

        assert!(result.is_accepted());
        assert!(!result.had_pii());
        assert_eq!(result.classification(), PiiClassification::None);
    }

    #[test]
    fn test_gate_allow_policy() {
        let gate = PiiGate::new(PiiPolicy::Allow);
        let result = gate.process("Email: test@example.com SSN: 456-78-9012");

        assert!(result.is_accepted());
        assert!(matches!(result, PiiGateResult::Allowed { .. }));
    }

    #[test]
    fn test_gate_redact_policy() {
        let gate = PiiGate::new(PiiPolicy::Redact);
        let result = gate.process("Contact: test@example.com");

        assert!(result.is_accepted());
        assert!(result.had_pii());

        if let PiiGateResult::Redacted { content, .. } = result {
            assert!(content.contains("[EMAIL]"));
            assert!(!content.contains("test@example.com"));
        } else {
            panic!("Expected Redacted result");
        }
    }

    #[test]
    fn test_gate_warn_policy() {
        let gate = PiiGate::new(PiiPolicy::Warn);
        let result = gate.process("SSN: 456-78-9012");

        assert!(result.is_accepted());
        assert!(result.had_pii());

        if let PiiGateResult::Warned { content, matches, .. } = result {
            assert!(content.contains("456-78-9012")); // Not redacted
            assert!(!matches.is_empty());
        } else {
            panic!("Expected Warned result");
        }
    }

    #[test]
    fn test_gate_reject_policy() {
        let gate = PiiGate::new(PiiPolicy::Reject);
        let result = gate.process("SSN: 456-78-9012");

        assert!(!result.is_accepted());
        assert!(result.had_pii());
        assert_eq!(result.classification(), PiiClassification::Critical);

        if let PiiGateResult::Rejected { reason, .. } = result {
            assert!(reason.contains("CRITICAL"));
        } else {
            panic!("Expected Rejected result");
        }
    }

    #[test]
    fn test_gate_threshold_allows_low_pii() {
        let gate = PiiGate::with_threshold(PiiPolicy::Reject, PiiClassification::Critical);
        // IP addresses are Low classification
        let result = gate.process("Server at 192.168.1.100");

        assert!(result.is_accepted());
        assert!(result.had_pii());
        assert_eq!(result.classification(), PiiClassification::Low);
    }

    #[test]
    fn test_gate_threshold_rejects_high_pii() {
        let gate = PiiGate::with_threshold(PiiPolicy::Reject, PiiClassification::Medium);
        // Email is Medium classification
        let result = gate.process("Email: test@example.com");

        assert!(!result.is_accepted());
        assert_eq!(result.classification(), PiiClassification::Medium);
    }

    #[test]
    fn test_would_reject() {
        let gate = PiiGate::new(PiiPolicy::Reject);

        assert!(gate.would_reject("SSN: 456-78-9012"));
        assert!(!gate.would_reject("Hello, world!"));

        let gate_allow = PiiGate::new(PiiPolicy::Allow);
        assert!(!gate_allow.would_reject("SSN: 456-78-9012"));
    }

    #[test]
    fn test_gate_content_method() {
        let gate = PiiGate::new(PiiPolicy::Redact);

        let result = gate.process("Email: test@example.com");
        assert!(result.content().is_some());
        assert!(result.content().unwrap().contains("[EMAIL]"));

        let gate_reject = PiiGate::new(PiiPolicy::Reject);
        let result_reject = gate_reject.process("SSN: 456-78-9012");
        assert!(result_reject.content().is_none());
    }
}
