//! Privacy management and data classification.
//!
//! Implements data classification per ADR-008, retention policies,
//! redaction and anonymization functions for PII protection.

use std::collections::HashMap;
use std::time::Duration;

use regex::Regex;
use serde::{Deserialize, Serialize};

use crate::security::pii::{PiiDetector, PiiMatch, PiiType};

/// Data classification levels per ADR-008.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum DataClassification {
    /// Public data - can be shared freely
    Public = 0,
    /// Internal data - keep within system boundaries
    Internal = 1,
    /// Confidential data - client-related, redact before external storage
    Confidential = 2,
    /// Restricted data - PII, must be encrypted at rest
    Restricted = 3,
}

impl DataClassification {
    /// Returns whether this classification requires encryption at rest.
    pub fn requires_encryption(&self) -> bool {
        *self >= DataClassification::Restricted
    }

    /// Returns whether this classification allows external sync.
    pub fn allows_external_sync(&self) -> bool {
        *self <= DataClassification::Internal
    }

    /// Returns whether this classification requires redaction for logging.
    pub fn requires_redaction(&self) -> bool {
        *self >= DataClassification::Confidential
    }

    /// Returns the default retention period for this classification.
    pub fn default_retention(&self) -> Option<Duration> {
        match self {
            DataClassification::Public => None, // No automatic deletion
            DataClassification::Internal => Some(Duration::from_secs(365 * 24 * 60 * 60)), // 1 year
            DataClassification::Confidential => Some(Duration::from_secs(90 * 24 * 60 * 60)), // 90 days
            DataClassification::Restricted => Some(Duration::from_secs(30 * 24 * 60 * 60)), // 30 days
        }
    }

    /// Returns a human-readable description.
    pub fn description(&self) -> &'static str {
        match self {
            DataClassification::Public => "Public - can be shared freely",
            DataClassification::Internal => "Internal - system use only",
            DataClassification::Confidential => "Confidential - client data, requires redaction",
            DataClassification::Restricted => "Restricted - PII, requires encryption",
        }
    }
}

impl Default for DataClassification {
    fn default() -> Self {
        DataClassification::Internal
    }
}

impl std::fmt::Display for DataClassification {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DataClassification::Public => write!(f, "PUBLIC"),
            DataClassification::Internal => write!(f, "INTERNAL"),
            DataClassification::Confidential => write!(f, "CONFIDENTIAL"),
            DataClassification::Restricted => write!(f, "RESTRICTED"),
        }
    }
}

/// Retention rule configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetentionRule {
    /// Data classification this rule applies to
    pub classification: DataClassification,
    /// Retention period (None = keep forever)
    pub retention_period: Option<Duration>,
    /// Whether to archive before deletion
    pub archive_before_delete: bool,
    /// Whether to require approval for deletion
    pub require_approval: bool,
}

impl RetentionRule {
    /// Create a new retention rule.
    pub fn new(classification: DataClassification) -> Self {
        Self {
            classification,
            retention_period: classification.default_retention(),
            archive_before_delete: classification >= DataClassification::Confidential,
            require_approval: classification == DataClassification::Restricted,
        }
    }

    /// Set a custom retention period.
    pub fn with_retention(mut self, period: Duration) -> Self {
        self.retention_period = Some(period);
        self
    }

    /// Remove retention period (keep forever).
    pub fn keep_forever(mut self) -> Self {
        self.retention_period = None;
        self
    }
}

/// Privacy policy configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrivacyPolicy {
    /// Retention rules by classification
    pub retention_rules: HashMap<DataClassification, RetentionRule>,
    /// Custom redaction patterns (regex -> replacement)
    pub custom_redaction_patterns: Vec<(String, String)>,
    /// Keywords that trigger confidential classification
    pub confidential_keywords: Vec<String>,
    /// Whether to auto-classify incoming data
    pub auto_classify: bool,
    /// Default classification for unclassified data
    pub default_classification: DataClassification,
}

impl Default for PrivacyPolicy {
    fn default() -> Self {
        let mut retention_rules = HashMap::new();
        retention_rules.insert(
            DataClassification::Public,
            RetentionRule::new(DataClassification::Public),
        );
        retention_rules.insert(
            DataClassification::Internal,
            RetentionRule::new(DataClassification::Internal),
        );
        retention_rules.insert(
            DataClassification::Confidential,
            RetentionRule::new(DataClassification::Confidential),
        );
        retention_rules.insert(
            DataClassification::Restricted,
            RetentionRule::new(DataClassification::Restricted),
        );

        Self {
            retention_rules,
            custom_redaction_patterns: Vec::new(),
            confidential_keywords: vec![
                "nda".to_string(),
                "client".to_string(),
                "confidential".to_string(),
                "proprietary".to_string(),
                "internal only".to_string(),
                "do not share".to_string(),
            ],
            auto_classify: true,
            default_classification: DataClassification::Internal,
        }
    }
}

impl PrivacyPolicy {
    /// Create a new privacy policy with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a strict privacy policy (shorter retention, more redaction).
    pub fn strict() -> Self {
        let mut policy = Self::default();

        // Shorter retention periods
        if let Some(rule) = policy.retention_rules.get_mut(&DataClassification::Internal) {
            rule.retention_period = Some(Duration::from_secs(180 * 24 * 60 * 60)); // 180 days
        }
        if let Some(rule) = policy.retention_rules.get_mut(&DataClassification::Confidential) {
            rule.retention_period = Some(Duration::from_secs(30 * 24 * 60 * 60)); // 30 days
            rule.require_approval = true;
        }
        if let Some(rule) = policy.retention_rules.get_mut(&DataClassification::Restricted) {
            rule.retention_period = Some(Duration::from_secs(7 * 24 * 60 * 60)); // 7 days
        }

        // Default to higher classification
        policy.default_classification = DataClassification::Confidential;

        policy
    }

    /// Add a custom redaction pattern.
    pub fn add_redaction_pattern(mut self, pattern: &str, replacement: &str) -> Self {
        self.custom_redaction_patterns
            .push((pattern.to_string(), replacement.to_string()));
        self
    }

    /// Add confidential keyword.
    pub fn add_confidential_keyword(mut self, keyword: &str) -> Self {
        self.confidential_keywords.push(keyword.to_lowercase());
        self
    }

    /// Get retention rule for a classification.
    pub fn get_retention_rule(&self, classification: DataClassification) -> Option<&RetentionRule> {
        self.retention_rules.get(&classification)
    }
}

/// Redaction style configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RedactionStyle {
    /// Replace with fixed text: [REDACTED]
    Fixed,
    /// Replace with type-specific text: [REDACTED_EMAIL]
    TypeLabeled,
    /// Replace with asterisks: jo**@ex*****.com
    Masked,
    /// Replace entire match with asterisks: **********************
    FullMask,
    /// Hash the content: [HASH:a1b2c3d4]
    Hashed,
}

impl Default for RedactionStyle {
    fn default() -> Self {
        RedactionStyle::TypeLabeled
    }
}

/// Privacy manager for data classification, redaction, and anonymization.
pub struct PrivacyManager {
    policy: PrivacyPolicy,
    pii_detector: PiiDetector,
    confidential_regex: Regex,
}

impl PrivacyManager {
    /// Create a new privacy manager with the given policy.
    pub fn new(policy: PrivacyPolicy) -> Self {
        // Build regex from confidential keywords
        let pattern = policy
            .confidential_keywords
            .iter()
            .map(|k| regex::escape(k))
            .collect::<Vec<_>>()
            .join("|");
        let confidential_regex =
            Regex::new(&format!(r"(?i)\b({})\b", pattern)).unwrap_or_else(|_| Regex::new(r"^$").unwrap());

        Self {
            policy,
            pii_detector: PiiDetector::new(),
            confidential_regex,
        }
    }

    /// Create a privacy manager with default policy.
    pub fn with_default_policy() -> Self {
        Self::new(PrivacyPolicy::default())
    }

    /// Classify data based on content analysis.
    pub fn classify(&self, data: &str) -> DataClassification {
        // Check for PII first (highest priority)
        if self.pii_detector.contains_critical_pii(data) {
            return DataClassification::Restricted;
        }

        if self.pii_detector.contains_pii(data) {
            return DataClassification::Restricted;
        }

        // Check for confidential keywords
        if self.confidential_regex.is_match(data) {
            return DataClassification::Confidential;
        }

        self.policy.default_classification
    }

    /// Redact PII from text using the specified style.
    pub fn redact(&self, text: &str, style: RedactionStyle) -> RedactedText {
        let matches = self.pii_detector.scan_text(text);
        let mut redacted = text.to_string();
        let mut redactions = Vec::new();

        // Process matches in reverse order to maintain positions
        for pii_match in matches.iter().rev() {
            let replacement = self.create_replacement(pii_match, style);
            redacted.replace_range(pii_match.range.clone(), &replacement);
            redactions.push(RedactionRecord {
                pii_type: pii_match.pii_type,
                original_length: pii_match.matched_text.len(),
                position: pii_match.range.start,
            });
        }

        // Apply custom redaction patterns
        for (pattern, replacement) in &self.policy.custom_redaction_patterns {
            if let Ok(re) = Regex::new(pattern) {
                redacted = re.replace_all(&redacted, replacement.as_str()).to_string();
            }
        }

        RedactedText {
            text: redacted,
            redactions_count: redactions.len(),
            redactions,
            original_classification: self.classify(text),
        }
    }

    /// Create replacement text based on redaction style.
    fn create_replacement(&self, pii_match: &PiiMatch, style: RedactionStyle) -> String {
        match style {
            RedactionStyle::Fixed => "[REDACTED]".to_string(),
            RedactionStyle::TypeLabeled => {
                format!("[REDACTED_{}]", pii_match.pii_type.label().to_uppercase().replace(' ', "_"))
            }
            RedactionStyle::Masked => pii_match.redacted_text(),
            RedactionStyle::FullMask => "*".repeat(pii_match.matched_text.len()),
            RedactionStyle::Hashed => {
                use ring::digest::{digest, SHA256};
                let hash = digest(&SHA256, pii_match.matched_text.as_bytes());
                let hex = hex::encode(&hash.as_ref()[..4]);
                format!("[HASH:{}]", hex)
            }
        }
    }

    /// Anonymize data by applying irreversible transformations.
    ///
    /// Unlike redaction, anonymization cannot be reversed and removes
    /// all potentially identifying information.
    pub fn anonymize(&self, text: &str) -> AnonymizedText {
        let mut result = text.to_string();
        let matches = self.pii_detector.scan_text(text);
        let mut transformations = Vec::new();

        // Process in reverse order
        for pii_match in matches.iter().rev() {
            let anonymized = self.anonymize_value(&pii_match.pii_type, &pii_match.matched_text);
            result.replace_range(pii_match.range.clone(), &anonymized);
            transformations.push(AnonymizationRecord {
                pii_type: pii_match.pii_type,
                method: self.get_anonymization_method(&pii_match.pii_type),
            });
        }

        // Remove confidential keywords
        result = self.confidential_regex.replace_all(&result, "[FILTERED]").to_string();

        AnonymizedText {
            text: result,
            transformations_count: transformations.len(),
            transformations,
            is_fully_anonymized: true,
        }
    }

    /// Anonymize a specific value based on its PII type.
    fn anonymize_value(&self, pii_type: &PiiType, _value: &str) -> String {
        match pii_type {
            PiiType::Email => "anonymous@example.com".to_string(),
            PiiType::PhoneNumber => "(000) 000-0000".to_string(),
            PiiType::Ssn => "000-00-0000".to_string(),
            PiiType::CreditCard => "0000-0000-0000-0000".to_string(),
            PiiType::Ipv4Address => "0.0.0.0".to_string(),
            PiiType::Ipv6Address => "::".to_string(),
            PiiType::ApiKey | PiiType::Password | PiiType::AwsAccessKey | PiiType::GithubToken => {
                "[CREDENTIAL_REMOVED]".to_string()
            }
            PiiType::JwtToken => "[TOKEN_REMOVED]".to_string(),
            PiiType::PrivateKey => "[KEY_REMOVED]".to_string(),
            PiiType::Name => "Anonymous User".to_string(),
            PiiType::Address => "123 Anonymous St".to_string(),
            PiiType::DateOfBirth => "1900-01-01".to_string(),
            PiiType::Passport | PiiType::DriversLicense => "[ID_REMOVED]".to_string(),
            PiiType::BankAccount => "[ACCOUNT_REMOVED]".to_string(),
        }
    }

    /// Get the anonymization method used for a PII type.
    fn get_anonymization_method(&self, pii_type: &PiiType) -> AnonymizationMethod {
        match pii_type {
            PiiType::Email | PiiType::PhoneNumber | PiiType::Name | PiiType::Address => {
                AnonymizationMethod::Generalization
            }
            PiiType::Ssn | PiiType::CreditCard | PiiType::Ipv4Address | PiiType::Ipv6Address => {
                AnonymizationMethod::Nullification
            }
            _ => AnonymizationMethod::Suppression,
        }
    }

    /// Get the privacy policy.
    pub fn policy(&self) -> &PrivacyPolicy {
        &self.policy
    }

    /// Check if data should be encrypted based on classification.
    pub fn should_encrypt(&self, data: &str) -> bool {
        self.classify(data).requires_encryption()
    }

    /// Check if data can be synced externally.
    pub fn can_sync_external(&self, data: &str) -> bool {
        self.classify(data).allows_external_sync()
    }

    /// Process data before storage, applying appropriate transformations.
    pub fn process_for_storage(&self, data: &str) -> ProcessedData {
        let classification = self.classify(data);

        let (processed_text, needs_encryption) = match classification {
            DataClassification::Public | DataClassification::Internal => {
                (data.to_string(), false)
            }
            DataClassification::Confidential => {
                let redacted = self.redact(data, RedactionStyle::TypeLabeled);
                (redacted.text, false)
            }
            DataClassification::Restricted => {
                let redacted = self.redact(data, RedactionStyle::TypeLabeled);
                (redacted.text, true)
            }
        };

        ProcessedData {
            text: processed_text,
            classification,
            needs_encryption,
            retention_period: self
                .policy
                .get_retention_rule(classification)
                .and_then(|r| r.retention_period),
        }
    }
}

/// Result of a redaction operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RedactedText {
    /// The redacted text
    pub text: String,
    /// Number of redactions performed
    pub redactions_count: usize,
    /// Details of each redaction
    pub redactions: Vec<RedactionRecord>,
    /// Original data classification
    pub original_classification: DataClassification,
}

/// Record of a single redaction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RedactionRecord {
    /// Type of PII that was redacted
    pub pii_type: PiiType,
    /// Original length of the redacted content
    pub original_length: usize,
    /// Position in the original text
    pub position: usize,
}

/// Result of an anonymization operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnonymizedText {
    /// The anonymized text
    pub text: String,
    /// Number of transformations performed
    pub transformations_count: usize,
    /// Details of each transformation
    pub transformations: Vec<AnonymizationRecord>,
    /// Whether the text is fully anonymized
    pub is_fully_anonymized: bool,
}

/// Record of a single anonymization transformation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnonymizationRecord {
    /// Type of PII that was anonymized
    pub pii_type: PiiType,
    /// Method used for anonymization
    pub method: AnonymizationMethod,
}

/// Anonymization methods.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AnonymizationMethod {
    /// Replace with generic value
    Generalization,
    /// Replace with null/zero value
    Nullification,
    /// Remove entirely
    Suppression,
    /// Add noise to numeric values
    Perturbation,
    /// Replace with pseudonym
    Pseudonymization,
}

/// Processed data ready for storage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessedData {
    /// The processed text
    pub text: String,
    /// Data classification
    pub classification: DataClassification,
    /// Whether encryption is required
    pub needs_encryption: bool,
    /// Recommended retention period
    pub retention_period: Option<Duration>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classification_public() {
        let manager = PrivacyManager::with_default_policy();
        let classification = manager.classify("Hello, world!");
        assert_eq!(classification, DataClassification::Internal); // Default
    }

    #[test]
    fn test_classification_confidential() {
        let manager = PrivacyManager::with_default_policy();
        let classification = manager.classify("This is NDA protected client information");
        assert_eq!(classification, DataClassification::Confidential);
    }

    #[test]
    fn test_classification_restricted() {
        let manager = PrivacyManager::with_default_policy();
        let classification = manager.classify("User email: john@example.com");
        assert_eq!(classification, DataClassification::Restricted);
    }

    #[test]
    fn test_redaction_fixed() {
        let manager = PrivacyManager::with_default_policy();
        let redacted = manager.redact("Contact: john@example.com", RedactionStyle::Fixed);
        assert!(redacted.text.contains("[REDACTED]"));
        assert!(!redacted.text.contains("john@example.com"));
        assert_eq!(redacted.redactions_count, 1);
    }

    #[test]
    fn test_redaction_type_labeled() {
        let manager = PrivacyManager::with_default_policy();
        let redacted = manager.redact("Contact: john@example.com", RedactionStyle::TypeLabeled);
        assert!(redacted.text.contains("[REDACTED_EMAIL_ADDRESS]"));
    }

    #[test]
    fn test_redaction_masked() {
        let manager = PrivacyManager::with_default_policy();
        let redacted = manager.redact("Contact: john@example.com", RedactionStyle::Masked);
        // Masked version preserves first and last 2 chars
        assert!(redacted.text.contains("jo"));
        assert!(redacted.text.contains("*"));
    }

    #[test]
    fn test_anonymization() {
        let manager = PrivacyManager::with_default_policy();
        let anonymized = manager.anonymize("Contact john@example.com or call (555) 123-4567");

        assert!(anonymized.text.contains("anonymous@example.com"));
        assert!(anonymized.text.contains("(000) 000-0000"));
        assert!(!anonymized.text.contains("john@example.com"));
        assert!(!anonymized.text.contains("555"));
        assert!(anonymized.transformations_count >= 2);
    }

    #[test]
    fn test_process_for_storage() {
        let manager = PrivacyManager::with_default_policy();

        // Internal data
        let processed = manager.process_for_storage("Regular internal data");
        assert_eq!(processed.classification, DataClassification::Internal);
        assert!(!processed.needs_encryption);

        // Restricted data (PII)
        let processed = manager.process_for_storage("SSN: 456-78-9012");
        assert_eq!(processed.classification, DataClassification::Restricted);
        assert!(processed.needs_encryption);
        assert!(processed.text.contains("[REDACTED"));
    }

    #[test]
    fn test_retention_periods() {
        let policy = PrivacyPolicy::default();

        assert!(policy
            .get_retention_rule(DataClassification::Public)
            .unwrap()
            .retention_period
            .is_none());

        assert!(policy
            .get_retention_rule(DataClassification::Restricted)
            .unwrap()
            .retention_period
            .is_some());
    }

    #[test]
    fn test_strict_policy() {
        let policy = PrivacyPolicy::strict();
        assert_eq!(
            policy.default_classification,
            DataClassification::Confidential
        );

        let restricted_rule = policy.get_retention_rule(DataClassification::Restricted).unwrap();
        assert_eq!(
            restricted_rule.retention_period,
            Some(Duration::from_secs(7 * 24 * 60 * 60))
        );
    }

    #[test]
    fn test_custom_redaction_patterns() {
        let policy = PrivacyPolicy::default()
            .add_redaction_pattern(r"PROJECT-\d+", "[PROJECT_ID]");
        let manager = PrivacyManager::new(policy);

        let redacted = manager.redact("Working on PROJECT-12345", RedactionStyle::Fixed);
        assert!(redacted.text.contains("[PROJECT_ID]"));
    }

    #[test]
    fn test_should_encrypt() {
        let manager = PrivacyManager::with_default_policy();

        assert!(!manager.should_encrypt("Regular text"));
        assert!(manager.should_encrypt("Email: user@test.com"));
    }

    #[test]
    fn test_can_sync_external() {
        let manager = PrivacyManager::with_default_policy();

        assert!(manager.can_sync_external("Public announcement"));
        assert!(!manager.can_sync_external("Client NDA details"));
        assert!(!manager.can_sync_external("SSN: 456-78-9012"));
    }
}
