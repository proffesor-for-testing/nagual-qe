//! Security module for the Nagual system.
//!
//! This module provides comprehensive security features including:
//!
//! - **PII Detection** (`pii`): Detect and classify personally identifiable information
//! - **Privacy Management** (`privacy`): Data classification, redaction, and anonymization
//! - **Audit Logging** (`audit`): Append-only audit trail for security events
//! - **Credential Management** (`credentials`): Secure credential storage and rotation
//!
//! ## Security Architecture
//!
//! The security module follows defense-in-depth principles:
//!
//! 1. **Data Classification**: All data is classified by sensitivity level
//! 2. **PII Protection**: Automatic detection and redaction of sensitive data
//! 3. **Encryption**: Data at rest encryption for restricted data
//! 4. **Audit Trail**: Immutable logging of all security-relevant events
//! 5. **Credential Rotation**: Automated rotation of secrets and keys
//!
//! ## Example Usage
//!
//! ```rust,ignore
//! use nagual::security::{
//!     pii::PiiDetector,
//!     privacy::{PrivacyManager, RedactionStyle},
//!     audit::{AuditLogger, AuditLoggerConfig},
//!     credentials::{CredentialManager, CredentialType},
//! };
//!
//! // PII Detection
//! let detector = PiiDetector::new();
//! let classification = detector.classify("Email: john@example.com");
//!
//! // Privacy Management
//! let privacy = PrivacyManager::with_default_policy();
//! let redacted = privacy.redact("SSN: 456-78-9012", RedactionStyle::TypeLabeled);
//!
//! // Credential Management
//! let key = [0u8; 32]; // In practice, derive from secure source
//! let creds = CredentialManager::new(key);
//! creds.store(CredentialType::ApiKey, "my_api_key", b"secret", None).await?;
//! ```

pub mod apikey_store;
pub mod audit;
pub mod credentials;
pub mod pii;
pub mod privacy;

// Re-exports for convenience
pub use apikey_store::{ApiKeyRecord, ApiKeyStore};
pub use audit::{AuditEntry, AuditEventType, AuditLogger, AuditLoggerConfig, AuditOutcome, AuditQuery};
pub use credentials::{
    CredentialManager, CredentialMetadata, CredentialStatus, CredentialType, RotationPolicy,
    RotationResult, derive_key, generate_key, generate_password, generate_salt, hash_password,
    sha256_hash, verify_password,
};
pub use pii::{PiiClassification, PiiDetector, PiiMatch, PiiSummary, PiiType};
pub use privacy::{
    AnonymizationMethod, AnonymizedText, DataClassification, PrivacyManager, PrivacyPolicy,
    ProcessedData, RedactedText, RedactionStyle, RetentionRule,
};

/// Security configuration combining all security settings.
#[derive(Debug, Clone)]
pub struct SecurityConfig {
    /// Privacy policy configuration
    pub privacy_policy: PrivacyPolicy,
    /// Audit logger configuration
    pub audit_config: AuditLoggerConfig,
    /// Whether to enable automatic PII detection
    pub auto_pii_detection: bool,
    /// Whether to enable audit logging
    pub audit_enabled: bool,
    /// Default encryption for restricted data
    pub encrypt_restricted: bool,
}

impl Default for SecurityConfig {
    fn default() -> Self {
        Self {
            privacy_policy: PrivacyPolicy::default(),
            audit_config: AuditLoggerConfig::default(),
            auto_pii_detection: true,
            audit_enabled: true,
            encrypt_restricted: true,
        }
    }
}

impl SecurityConfig {
    /// Create a minimal security configuration (for testing).
    pub fn minimal() -> Self {
        Self {
            privacy_policy: PrivacyPolicy::default(),
            audit_config: AuditLoggerConfig {
                buffer_size: 10,
                flush_interval: std::time::Duration::from_secs(60),
                file_path: None,
                enable_chain_hashing: false,
                min_log_level: None,
            },
            auto_pii_detection: false,
            audit_enabled: false,
            encrypt_restricted: false,
        }
    }

    /// Create a strict security configuration (for production).
    pub fn strict() -> Self {
        Self {
            privacy_policy: PrivacyPolicy::strict(),
            audit_config: AuditLoggerConfig {
                buffer_size: 50,
                flush_interval: std::time::Duration::from_secs(1),
                file_path: Some("/var/log/nagual/audit.log".to_string()),
                enable_chain_hashing: true,
                min_log_level: None,
            },
            auto_pii_detection: true,
            audit_enabled: true,
            encrypt_restricted: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_security_config_default() {
        let config = SecurityConfig::default();
        assert!(config.auto_pii_detection);
        assert!(config.audit_enabled);
        assert!(config.encrypt_restricted);
    }

    #[test]
    fn test_security_config_minimal() {
        let config = SecurityConfig::minimal();
        assert!(!config.auto_pii_detection);
        assert!(!config.audit_enabled);
        assert!(!config.encrypt_restricted);
    }

    #[test]
    fn test_security_config_strict() {
        let config = SecurityConfig::strict();
        assert!(config.auto_pii_detection);
        assert!(config.audit_enabled);
        assert!(config.encrypt_restricted);
        assert!(config.audit_config.file_path.is_some());
        assert!(config.audit_config.enable_chain_hashing);
    }
}
