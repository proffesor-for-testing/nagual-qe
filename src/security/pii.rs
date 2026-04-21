//! PII (Personally Identifiable Information) detection and classification.
//!
//! Provides comprehensive PII detection using regex patterns for common
//! sensitive data types including emails, phone numbers, SSNs, credit cards,
//! and IP addresses.

use std::ops::Range;

use regex::Regex;
use serde::{Deserialize, Serialize};

/// Classification level for detected PII.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum PiiClassification {
    /// No PII detected
    None = 0,
    /// Low sensitivity: IP addresses, device identifiers
    Low = 1,
    /// Medium sensitivity: Email addresses, phone numbers
    Medium = 2,
    /// High sensitivity: Names, addresses, dates of birth
    High = 3,
    /// Critical sensitivity: SSN, credit cards, passwords, API keys
    Critical = 4,
}

impl PiiClassification {
    /// Returns a human-readable description of the classification level.
    pub fn description(&self) -> &'static str {
        match self {
            PiiClassification::None => "No PII detected",
            PiiClassification::Low => "Low sensitivity (IP addresses, device IDs)",
            PiiClassification::Medium => "Medium sensitivity (emails, phone numbers)",
            PiiClassification::High => "High sensitivity (names, addresses, DOB)",
            PiiClassification::Critical => "Critical sensitivity (SSN, credit cards, credentials)",
        }
    }

    /// Returns whether this classification requires encryption at rest.
    pub fn requires_encryption(&self) -> bool {
        matches!(self, PiiClassification::High | PiiClassification::Critical)
    }

    /// Returns whether this classification requires redaction before logging.
    pub fn requires_redaction(&self) -> bool {
        *self >= PiiClassification::Medium
    }
}

impl Default for PiiClassification {
    fn default() -> Self {
        PiiClassification::None
    }
}

impl std::fmt::Display for PiiClassification {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PiiClassification::None => write!(f, "NONE"),
            PiiClassification::Low => write!(f, "LOW"),
            PiiClassification::Medium => write!(f, "MEDIUM"),
            PiiClassification::High => write!(f, "HIGH"),
            PiiClassification::Critical => write!(f, "CRITICAL"),
        }
    }
}

/// Type of PII detected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PiiType {
    /// Email address
    Email,
    /// Phone number (various formats)
    PhoneNumber,
    /// Social Security Number
    Ssn,
    /// Credit card number
    CreditCard,
    /// IPv4 address
    Ipv4Address,
    /// IPv6 address
    Ipv6Address,
    /// API key or token
    ApiKey,
    /// Password or secret
    Password,
    /// Date of birth
    DateOfBirth,
    /// Physical address
    Address,
    /// Personal name
    Name,
    /// Passport number
    Passport,
    /// Driver's license
    DriversLicense,
    /// Bank account number
    BankAccount,
    /// AWS access key
    AwsAccessKey,
    /// GitHub token
    GithubToken,
    /// JWT token
    JwtToken,
    /// Private key
    PrivateKey,
}

impl PiiType {
    /// Returns the classification level for this PII type.
    pub fn classification(&self) -> PiiClassification {
        match self {
            PiiType::Ipv4Address | PiiType::Ipv6Address => PiiClassification::Low,
            PiiType::Email | PiiType::PhoneNumber => PiiClassification::Medium,
            PiiType::Name | PiiType::Address | PiiType::DateOfBirth => PiiClassification::High,
            PiiType::Ssn
            | PiiType::CreditCard
            | PiiType::ApiKey
            | PiiType::Password
            | PiiType::Passport
            | PiiType::DriversLicense
            | PiiType::BankAccount
            | PiiType::AwsAccessKey
            | PiiType::GithubToken
            | PiiType::JwtToken
            | PiiType::PrivateKey => PiiClassification::Critical,
        }
    }

    /// Returns a human-readable label for the PII type.
    pub fn label(&self) -> &'static str {
        match self {
            PiiType::Email => "Email Address",
            PiiType::PhoneNumber => "Phone Number",
            PiiType::Ssn => "Social Security Number",
            PiiType::CreditCard => "Credit Card Number",
            PiiType::Ipv4Address => "IPv4 Address",
            PiiType::Ipv6Address => "IPv6 Address",
            PiiType::ApiKey => "API Key",
            PiiType::Password => "Password",
            PiiType::DateOfBirth => "Date of Birth",
            PiiType::Address => "Physical Address",
            PiiType::Name => "Personal Name",
            PiiType::Passport => "Passport Number",
            PiiType::DriversLicense => "Driver's License",
            PiiType::BankAccount => "Bank Account Number",
            PiiType::AwsAccessKey => "AWS Access Key",
            PiiType::GithubToken => "GitHub Token",
            PiiType::JwtToken => "JWT Token",
            PiiType::PrivateKey => "Private Key",
        }
    }
}

impl std::fmt::Display for PiiType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.label())
    }
}

/// A detected PII occurrence with position information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PiiMatch {
    /// Type of PII detected
    pub pii_type: PiiType,
    /// Classification level
    pub classification: PiiClassification,
    /// The matched text (for logging, may be partially redacted)
    pub matched_text: String,
    /// Byte range in the original text
    pub range: Range<usize>,
    /// Confidence score (0.0 - 1.0)
    pub confidence: f64,
}

impl PiiMatch {
    /// Returns a redacted version of the matched text suitable for logging.
    pub fn redacted_text(&self) -> String {
        let len = self.matched_text.len();
        if len <= 4 {
            "*".repeat(len)
        } else {
            format!(
                "{}{}{}",
                &self.matched_text[..2],
                "*".repeat(len - 4),
                &self.matched_text[len - 2..]
            )
        }
    }
}

/// Pattern definition for PII detection.
struct PiiPattern {
    pii_type: PiiType,
    regex: Regex,
    confidence: f64,
}

/// PII detector with configurable patterns.
pub struct PiiDetector {
    patterns: Vec<PiiPattern>,
}

impl Default for PiiDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl PiiDetector {
    /// Create a new PII detector with default patterns.
    pub fn new() -> Self {
        let patterns = vec![
            // Email addresses (RFC 5322 simplified)
            PiiPattern {
                pii_type: PiiType::Email,
                regex: Regex::new(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}").unwrap(),
                confidence: 0.95,
            },
            // US Phone numbers (various formats)
            PiiPattern {
                pii_type: PiiType::PhoneNumber,
                regex: Regex::new(
                    r"(?:\+1[-.\s]?)?\(?[2-9]\d{2}\)?[-.\s]?\d{3}[-.\s]?\d{4}",
                )
                .unwrap(),
                confidence: 0.85,
            },
            // International phone numbers
            PiiPattern {
                pii_type: PiiType::PhoneNumber,
                regex: Regex::new(r"\+[1-9]\d{1,14}").unwrap(),
                confidence: 0.80,
            },
            // Social Security Number (XXX-XX-XXXX)
            PiiPattern {
                pii_type: PiiType::Ssn,
                regex: Regex::new(r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b").unwrap(),
                confidence: 0.90,
            },
            // Credit card numbers (major brands)
            // Visa: 4XXX
            PiiPattern {
                pii_type: PiiType::CreditCard,
                regex: Regex::new(r"\b4\d{3}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b").unwrap(),
                confidence: 0.95,
            },
            // Mastercard: 5XXX or 2XXX
            PiiPattern {
                pii_type: PiiType::CreditCard,
                regex: Regex::new(r"\b[52]\d{3}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b").unwrap(),
                confidence: 0.95,
            },
            // American Express: 34XX or 37XX
            PiiPattern {
                pii_type: PiiType::CreditCard,
                regex: Regex::new(r"\b3[47]\d{2}[-\s]?\d{6}[-\s]?\d{5}\b").unwrap(),
                confidence: 0.95,
            },
            // Generic credit card (13-19 digits)
            PiiPattern {
                pii_type: PiiType::CreditCard,
                regex: Regex::new(r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{1,7}\b").unwrap(),
                confidence: 0.75,
            },
            // IPv4 addresses
            PiiPattern {
                pii_type: PiiType::Ipv4Address,
                regex: Regex::new(
                    r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b",
                )
                .unwrap(),
                confidence: 0.90,
            },
            // IPv6 addresses (simplified)
            PiiPattern {
                pii_type: PiiType::Ipv6Address,
                regex: Regex::new(
                    r"(?i)\b(?:[0-9a-f]{1,4}:){7}[0-9a-f]{1,4}\b",
                )
                .unwrap(),
                confidence: 0.90,
            },
            // AWS Access Key
            PiiPattern {
                pii_type: PiiType::AwsAccessKey,
                regex: Regex::new(r"(?:AKIA|ABIA|ACCA|ASIA)[0-9A-Z]{16}").unwrap(),
                confidence: 0.99,
            },
            // GitHub Personal Access Token
            PiiPattern {
                pii_type: PiiType::GithubToken,
                regex: Regex::new(r"ghp_[a-zA-Z0-9]{36}").unwrap(),
                confidence: 0.99,
            },
            // GitHub fine-grained token
            PiiPattern {
                pii_type: PiiType::GithubToken,
                regex: Regex::new(r"github_pat_[a-zA-Z0-9]{22}_[a-zA-Z0-9]{59}").unwrap(),
                confidence: 0.99,
            },
            // JWT tokens
            PiiPattern {
                pii_type: PiiType::JwtToken,
                regex: Regex::new(r"eyJ[a-zA-Z0-9_-]*\.eyJ[a-zA-Z0-9_-]*\.[a-zA-Z0-9_-]*").unwrap(),
                confidence: 0.95,
            },
            // Generic API keys (key=value patterns)
            PiiPattern {
                pii_type: PiiType::ApiKey,
                regex: Regex::new(
                    r#"(?i)(?:api[_-]?key|secret|token|password|auth)\s*[:=]\s*['"][^'"]{8,}['"]"#,
                )
                .unwrap(),
                confidence: 0.85,
            },
            // Generic API keys (standalone)
            PiiPattern {
                pii_type: PiiType::ApiKey,
                regex: Regex::new(r"(?i)(?:sk|pk|api)[-_][a-zA-Z0-9]{20,}").unwrap(),
                confidence: 0.80,
            },
            // Private key headers
            PiiPattern {
                pii_type: PiiType::PrivateKey,
                regex: Regex::new(r"-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----").unwrap(),
                confidence: 0.99,
            },
            // Password assignments
            PiiPattern {
                pii_type: PiiType::Password,
                regex: Regex::new(
                    r#"(?i)password\s*[:=]\s*['"]?[^'"<>\s]{4,}['"]?"#,
                )
                .unwrap(),
                confidence: 0.75,
            },
        ];

        Self { patterns }
    }

    /// Create a PII detector with custom patterns.
    pub fn with_patterns(patterns: Vec<(PiiType, &str, f64)>) -> Result<Self, regex::Error> {
        let patterns = patterns
            .into_iter()
            .map(|(pii_type, pattern, confidence)| {
                Ok(PiiPattern {
                    pii_type,
                    regex: Regex::new(pattern)?,
                    confidence,
                })
            })
            .collect::<Result<Vec<_>, regex::Error>>()?;

        Ok(Self { patterns })
    }

    /// Classify the overall PII sensitivity level of the text.
    ///
    /// Returns the highest classification found in the text.
    pub fn classify(&self, text: &str) -> PiiClassification {
        self.scan_text(text)
            .iter()
            .map(|m| m.classification)
            .max()
            .unwrap_or(PiiClassification::None)
    }

    /// Scan text for all PII occurrences with position information.
    pub fn scan_text(&self, text: &str) -> Vec<PiiMatch> {
        let mut matches = Vec::new();

        for pattern in &self.patterns {
            for regex_match in pattern.regex.find_iter(text) {
                let matched_text = regex_match.as_str().to_string();

                // Skip common false positives
                if self.is_false_positive(&pattern.pii_type, &matched_text) {
                    continue;
                }

                matches.push(PiiMatch {
                    pii_type: pattern.pii_type,
                    classification: pattern.pii_type.classification(),
                    matched_text,
                    range: regex_match.start()..regex_match.end(),
                    confidence: pattern.confidence,
                });
            }
        }

        // Sort by position
        matches.sort_by_key(|m| m.range.start);

        // Remove overlapping matches (keep higher classification)
        self.deduplicate_matches(matches)
    }

    /// Check if the match is likely a false positive.
    fn is_false_positive(&self, pii_type: &PiiType, text: &str) -> bool {
        match pii_type {
            PiiType::Ipv4Address => {
                // Skip localhost and common internal IPs that aren't really PII
                matches!(
                    text,
                    "127.0.0.1" | "0.0.0.0" | "255.255.255.255" | "255.255.255.0"
                )
            }
            PiiType::Ssn => {
                // Skip obvious non-SSN patterns like 123-45-6789
                text == "123-45-6789" || text == "000-00-0000"
            }
            PiiType::CreditCard => {
                // Skip test card numbers
                text.replace(['-', ' '], "").starts_with("4111111111111111")
            }
            _ => false,
        }
    }

    /// Remove overlapping matches, keeping the one with higher classification.
    fn deduplicate_matches(&self, matches: Vec<PiiMatch>) -> Vec<PiiMatch> {
        if matches.len() <= 1 {
            return matches;
        }

        let mut result: Vec<PiiMatch> = Vec::new();

        for current in matches {
            // Check if this match overlaps with any existing result
            let overlaps_with = result
                .iter()
                .position(|existing| {
                    // Check for overlap
                    current.range.start < existing.range.end
                        && current.range.end > existing.range.start
                });

            match overlaps_with {
                Some(idx) => {
                    // If current has higher classification, replace
                    if current.classification > result[idx].classification {
                        result[idx] = current;
                    }
                    // Otherwise, keep existing (do nothing)
                }
                None => {
                    // No overlap, add to results
                    result.push(current);
                }
            }
        }

        result
    }

    /// Check if text contains any PII.
    pub fn contains_pii(&self, text: &str) -> bool {
        self.classify(text) != PiiClassification::None
    }

    /// Check if text contains critical PII (SSN, credit cards, etc.).
    pub fn contains_critical_pii(&self, text: &str) -> bool {
        self.classify(text) == PiiClassification::Critical
    }

    /// Get a summary of all PII types found in the text.
    pub fn summarize(&self, text: &str) -> PiiSummary {
        let matches = self.scan_text(text);
        let mut type_counts = std::collections::HashMap::new();

        for m in &matches {
            *type_counts.entry(m.pii_type).or_insert(0) += 1;
        }

        PiiSummary {
            total_matches: matches.len(),
            highest_classification: matches
                .iter()
                .map(|m| m.classification)
                .max()
                .unwrap_or(PiiClassification::None),
            type_counts,
            matches,
        }
    }

    /// Redact all PII in the text, replacing matches with tokens like [EMAIL], [SSN], etc.
    ///
    /// Returns a `RedactionResult` containing the redacted text and details about
    /// what was redacted.
    ///
    /// # Example
    ///
    /// ```
    /// use nagual::security::pii::PiiDetector;
    ///
    /// let detector = PiiDetector::new();
    /// let result = detector.redact("Contact me at john@example.com");
    /// assert!(result.redacted_text.contains("[EMAIL]"));
    /// assert_eq!(result.redaction_count, 1);
    /// ```
    pub fn redact(&self, text: &str) -> RedactionResult {
        let matches = self.scan_text(text);

        if matches.is_empty() {
            return RedactionResult {
                redacted_text: text.to_string(),
                original_text: text.to_string(),
                redaction_count: 0,
                redactions: Vec::new(),
                highest_classification: PiiClassification::None,
            };
        }

        // Sort matches by position (reverse order for safe replacement)
        let mut sorted_matches = matches.clone();
        sorted_matches.sort_by(|a, b| b.range.start.cmp(&a.range.start));

        let mut redacted = text.to_string();
        let mut redactions = Vec::new();

        for m in &sorted_matches {
            let token = Self::redaction_token(m.pii_type);

            // Ensure we don't go out of bounds
            if m.range.start < redacted.len() && m.range.end <= redacted.len() {
                redacted.replace_range(m.range.clone(), token);
                redactions.push(Redaction {
                    pii_type: m.pii_type,
                    token: token.to_string(),
                    original_range: m.range.clone(),
                    classification: m.classification,
                });
            }
        }

        RedactionResult {
            redacted_text: redacted,
            original_text: text.to_string(),
            redaction_count: redactions.len(),
            redactions,
            highest_classification: sorted_matches
                .iter()
                .map(|m| m.classification)
                .max()
                .unwrap_or(PiiClassification::None),
        }
    }

    /// Get the redaction token for a PII type.
    fn redaction_token(pii_type: PiiType) -> &'static str {
        match pii_type {
            PiiType::Email => "[EMAIL]",
            PiiType::PhoneNumber => "[PHONE]",
            PiiType::Ssn => "[SSN]",
            PiiType::CreditCard => "[CC]",
            PiiType::Ipv4Address => "[IP]",
            PiiType::Ipv6Address => "[IP]",
            PiiType::ApiKey => "[API_KEY]",
            PiiType::Password => "[PASSWORD]",
            PiiType::DateOfBirth => "[DOB]",
            PiiType::Address => "[ADDRESS]",
            PiiType::Name => "[NAME]",
            PiiType::Passport => "[PASSPORT]",
            PiiType::DriversLicense => "[DL]",
            PiiType::BankAccount => "[BANK_ACCT]",
            PiiType::AwsAccessKey => "[AWS_KEY]",
            PiiType::GithubToken => "[GH_TOKEN]",
            PiiType::JwtToken => "[JWT]",
            PiiType::PrivateKey => "[PRIVATE_KEY]",
        }
    }
}

/// Result of a redaction operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RedactionResult {
    /// The text with PII replaced by tokens.
    pub redacted_text: String,
    /// The original text (for reference).
    pub original_text: String,
    /// Number of redactions made.
    pub redaction_count: usize,
    /// Details of each redaction.
    pub redactions: Vec<Redaction>,
    /// Highest PII classification found.
    pub highest_classification: PiiClassification,
}

impl RedactionResult {
    /// Check if any redactions were made.
    pub fn was_redacted(&self) -> bool {
        self.redaction_count > 0
    }

    /// Check if critical PII was found and redacted.
    pub fn had_critical_pii(&self) -> bool {
        self.highest_classification == PiiClassification::Critical
    }
}

/// Details of a single redaction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Redaction {
    /// Type of PII that was redacted.
    pub pii_type: PiiType,
    /// The token used to replace the PII.
    pub token: String,
    /// Original byte range in the text.
    pub original_range: Range<usize>,
    /// Classification level of the redacted PII.
    pub classification: PiiClassification,
}

/// Summary of PII detection results.
#[derive(Debug, Serialize, Deserialize)]
pub struct PiiSummary {
    /// Total number of PII matches found
    pub total_matches: usize,
    /// Highest classification level found
    pub highest_classification: PiiClassification,
    /// Count of each PII type found
    pub type_counts: std::collections::HashMap<PiiType, usize>,
    /// All matches found
    pub matches: Vec<PiiMatch>,
}

impl PiiSummary {
    /// Check if any PII was found.
    pub fn has_pii(&self) -> bool {
        self.total_matches > 0
    }

    /// Check if critical PII was found.
    pub fn has_critical_pii(&self) -> bool {
        self.highest_classification == PiiClassification::Critical
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_email_detection() {
        let detector = PiiDetector::new();
        let text = "Contact me at john.doe@example.com for more info.";
        let matches = detector.scan_text(text);

        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].pii_type, PiiType::Email);
        assert_eq!(matches[0].classification, PiiClassification::Medium);
        assert_eq!(matches[0].matched_text, "john.doe@example.com");
    }

    #[test]
    fn test_phone_detection() {
        let detector = PiiDetector::new();
        let text = "Call me at (555) 123-4567 or +1-555-987-6543";
        let matches = detector.scan_text(text);

        assert!(matches.iter().any(|m| m.pii_type == PiiType::PhoneNumber));
        assert_eq!(
            matches
                .iter()
                .filter(|m| m.pii_type == PiiType::PhoneNumber)
                .count(),
            2
        );
    }

    #[test]
    fn test_ssn_detection() {
        let detector = PiiDetector::new();
        let text = "SSN: 123-45-6789"; // This is a known false positive
        let matches = detector.scan_text(text);
        assert!(matches.is_empty()); // Should be filtered out

        let text2 = "SSN: 456-78-9012";
        let matches2 = detector.scan_text(text2);
        assert_eq!(matches2.len(), 1);
        assert_eq!(matches2[0].pii_type, PiiType::Ssn);
        assert_eq!(matches2[0].classification, PiiClassification::Critical);
    }

    #[test]
    fn test_credit_card_detection() {
        let detector = PiiDetector::new();

        // Visa
        let visa = "Card: 4532-1234-5678-9012";
        let matches = detector.scan_text(visa);
        assert!(matches.iter().any(|m| m.pii_type == PiiType::CreditCard));

        // Mastercard
        let mc = "Card: 5234 5678 9012 3456";
        let matches = detector.scan_text(mc);
        assert!(matches.iter().any(|m| m.pii_type == PiiType::CreditCard));
    }

    #[test]
    fn test_ip_detection() {
        let detector = PiiDetector::new();
        let text = "Server at 192.168.1.100 responded. Skip 127.0.0.1 (localhost)";
        let matches = detector.scan_text(text);

        // Should detect 192.168.1.100 but skip 127.0.0.1
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].pii_type, PiiType::Ipv4Address);
        assert_eq!(matches[0].matched_text, "192.168.1.100");
    }

    #[test]
    fn test_aws_key_detection() {
        let detector = PiiDetector::new();
        let text = "AWS key: AKIAIOSFODNN7EXAMPLE";
        let matches = detector.scan_text(text);

        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].pii_type, PiiType::AwsAccessKey);
        assert_eq!(matches[0].classification, PiiClassification::Critical);
    }

    #[test]
    fn test_github_token_detection() {
        let detector = PiiDetector::new();
        let text = "Token: ghp_abcdefghijklmnopqrstuvwxyz0123456789";
        let matches = detector.scan_text(text);

        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].pii_type, PiiType::GithubToken);
    }

    #[test]
    fn test_jwt_detection() {
        let detector = PiiDetector::new();
        let text = "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U";
        let matches = detector.scan_text(text);

        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].pii_type, PiiType::JwtToken);
    }

    #[test]
    fn test_private_key_detection() {
        let detector = PiiDetector::new();
        let text = "-----BEGIN RSA PRIVATE KEY-----\nMIIEpQ...";
        let matches = detector.scan_text(text);

        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].pii_type, PiiType::PrivateKey);
        assert_eq!(matches[0].classification, PiiClassification::Critical);
    }

    #[test]
    fn test_classification_levels() {
        let detector = PiiDetector::new();

        // No PII
        assert_eq!(
            detector.classify("Hello, world!"),
            PiiClassification::None
        );

        // Low (IP)
        assert_eq!(
            detector.classify("Server at 10.0.0.1"),
            PiiClassification::Low
        );

        // Medium (email)
        assert_eq!(
            detector.classify("Email: user@test.com"),
            PiiClassification::Medium
        );

        // Critical (SSN)
        assert_eq!(
            detector.classify("SSN: 456-78-9012"),
            PiiClassification::Critical
        );
    }

    #[test]
    fn test_redacted_text() {
        let m = PiiMatch {
            pii_type: PiiType::Email,
            classification: PiiClassification::Medium,
            matched_text: "john.doe@example.com".to_string(),
            range: 0..20,
            confidence: 0.95,
        };

        let redacted = m.redacted_text();
        assert!(redacted.starts_with("jo"));
        assert!(redacted.ends_with("om"));
        assert!(redacted.contains("*"));
    }

    #[test]
    fn test_summary() {
        let detector = PiiDetector::new();
        let text = "Contact john@test.com or call (555) 123-4567. SSN: 456-78-9012";
        let summary = detector.summarize(text);

        assert!(summary.has_pii());
        assert!(summary.has_critical_pii());
        assert_eq!(summary.highest_classification, PiiClassification::Critical);
        assert!(summary.total_matches >= 3);
    }

    #[test]
    fn test_redact_email() {
        let detector = PiiDetector::new();
        let result = detector.redact("Contact me at john.doe@example.com for more info.");

        assert!(result.was_redacted());
        assert!(result.redacted_text.contains("[EMAIL]"));
        assert!(!result.redacted_text.contains("john.doe@example.com"));
        assert_eq!(result.redaction_count, 1);
    }

    #[test]
    fn test_redact_multiple_pii() {
        let detector = PiiDetector::new();
        let result = detector.redact("Email: test@example.com, SSN: 456-78-9012");

        assert!(result.was_redacted());
        assert!(result.redacted_text.contains("[EMAIL]"));
        assert!(result.redacted_text.contains("[SSN]"));
        assert!(result.had_critical_pii());
        assert_eq!(result.redaction_count, 2);
    }

    #[test]
    fn test_redact_no_pii() {
        let detector = PiiDetector::new();
        let result = detector.redact("Hello, this is a normal message with no PII.");

        assert!(!result.was_redacted());
        assert_eq!(result.redacted_text, result.original_text);
        assert_eq!(result.redaction_count, 0);
        assert_eq!(result.highest_classification, PiiClassification::None);
    }

    #[test]
    fn test_redact_phone() {
        let detector = PiiDetector::new();
        let result = detector.redact("Call me at (555) 123-4567");

        assert!(result.was_redacted());
        assert!(result.redacted_text.contains("[PHONE]"));
        assert_eq!(result.redaction_count, 1);
    }

    #[test]
    fn test_redact_credit_card() {
        let detector = PiiDetector::new();
        let result = detector.redact("Card: 4532-1234-5678-9012");

        assert!(result.was_redacted());
        assert!(result.redacted_text.contains("[CC]"));
        assert!(result.had_critical_pii());
    }

    #[test]
    fn test_redact_aws_key() {
        let detector = PiiDetector::new();
        let result = detector.redact("AWS_ACCESS_KEY=AKIAIOSFODNN7EXAMPLE");

        assert!(result.was_redacted());
        assert!(result.redacted_text.contains("[AWS_KEY]"));
        assert!(result.had_critical_pii());
    }

    #[test]
    fn test_redact_preserves_context() {
        let detector = PiiDetector::new();
        let result = detector.redact("Hello John, your email is test@example.com. Best regards!");

        assert!(result.redacted_text.starts_with("Hello John, your email is "));
        assert!(result.redacted_text.ends_with(". Best regards!"));
        assert!(result.redacted_text.contains("[EMAIL]"));
    }
}
