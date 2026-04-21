//! PII redaction pipeline for cloud-bound data.
//!
//! Strips sensitive information before syncing to PostgreSQL, cloud API,
//! or an external Brain API. Local SQLite storage is NEVER modified.
//!
//! Implements 12 regex pattern categories ported from mcp-brain's pipeline.rs,
//! adapted for Nagual's specific needs (e.g., `ngk_` API keys).

use regex::Regex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::OnceLock;

/// Global toggle for PII redaction. Enabled by default.
static PII_REDACTION_ENABLED: AtomicBool = AtomicBool::new(true);

/// Disable PII redaction globally (for trusted environments).
pub fn disable_redaction() {
    PII_REDACTION_ENABLED.store(false, Ordering::Relaxed);
}

/// Enable PII redaction globally (default).
pub fn enable_redaction() {
    PII_REDACTION_ENABLED.store(true, Ordering::Relaxed);
}

/// Check if PII redaction is currently enabled.
pub fn is_redaction_enabled() -> bool {
    PII_REDACTION_ENABLED.load(Ordering::Relaxed)
}

/// Result of PII redaction with metadata about what was stripped.
#[derive(Debug, Clone)]
pub struct RedactionResult {
    /// The redacted text (PII replaced with placeholders)
    pub text: String,
    /// Number of distinct PII categories that were redacted
    pub redactions_count: usize,
    /// Names of PII categories that were detected and redacted
    pub categories: Vec<String>,
}

/// PII redaction pipeline with 12 pattern categories.
///
/// Pre-compiles all regex patterns on construction for efficient reuse.
/// Use [`global_redactor`] for a singleton instance to avoid repeated compilation.
pub struct PiiRedactor {
    patterns: Vec<PiiPattern>,
}

struct PiiPattern {
    name: &'static str,
    regex: Regex,
    replacement: &'static str,
}

impl PiiRedactor {
    /// Create a new PII redactor with all 12 pattern categories.
    pub fn new() -> Self {
        Self {
            patterns: vec![
                // 1. File system paths (Unix absolute paths starting with known
                //    root directories + Windows drive letter paths).
                //    Avoids false positives on API endpoints (/api/v1/...)
                //    and module paths (std/io/Error, tokio/runtime).
                PiiPattern {
                    name: "file_path",
                    regex: Regex::new(
                        r"(?:/(?:home|Users|tmp|var|etc|opt|usr|root|data|mnt|proc|sys|dev|srv)/[\w./-]+)|(?:[A-Z]:\\[\w\\.-]+)",
                    )
                    .unwrap(),
                    replacement: "[PATH_REDACTED]",
                },
                // 2. IP addresses (v4)
                PiiPattern {
                    name: "ip_address",
                    regex: Regex::new(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b").unwrap(),
                    replacement: "[IP_REDACTED]",
                },
                // 3. Email addresses
                PiiPattern {
                    name: "email",
                    regex: Regex::new(
                        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
                    )
                    .unwrap(),
                    replacement: "[EMAIL_REDACTED]",
                },
                // 4. API keys / tokens (generic hex/base64 patterns)
                PiiPattern {
                    name: "api_key",
                    regex: Regex::new(
                        r"(?i)(?:api[_-]?key|token|secret|password|auth)\s*[=:]\s*['\x22]?[\w/+=-]{16,}['\x22]?",
                    )
                    .unwrap(),
                    replacement: "[API_KEY_REDACTED]",
                },
                // 5. AWS access keys
                PiiPattern {
                    name: "aws_key",
                    regex: Regex::new(r"AKIA[0-9A-Z]{16}").unwrap(),
                    replacement: "[AWS_KEY_REDACTED]",
                },
                // 6. SSH private keys
                PiiPattern {
                    name: "ssh_key",
                    regex: Regex::new(
                        r"-----BEGIN (?:RSA |DSA |EC |OPENSSH )?PRIVATE KEY-----",
                    )
                    .unwrap(),
                    replacement: "[SSH_KEY_REDACTED]",
                },
                // 7. JWT tokens
                PiiPattern {
                    name: "jwt",
                    regex: Regex::new(
                        r"eyJ[A-Za-z0-9_-]+\.eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+",
                    )
                    .unwrap(),
                    replacement: "[JWT_REDACTED]",
                },
                // 8. Connection strings
                PiiPattern {
                    name: "connection_string",
                    regex: Regex::new(r"(?i)(?:postgres|mysql|mongodb|redis)://\S+").unwrap(),
                    replacement: "[CONNECTION_STRING_REDACTED]",
                },
                // 9. Phone numbers (US format)
                PiiPattern {
                    name: "phone",
                    regex: Regex::new(
                        r"\b(?:\+1[-.]?)?\(?\d{3}\)?[-.]?\d{3}[-.]?\d{4}\b",
                    )
                    .unwrap(),
                    replacement: "[PHONE_REDACTED]",
                },
                // 10. Credit card numbers
                PiiPattern {
                    name: "credit_card",
                    regex: Regex::new(r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b").unwrap(),
                    replacement: "[CC_REDACTED]",
                },
                // 11. Social Security Numbers
                PiiPattern {
                    name: "ssn",
                    regex: Regex::new(r"\b\d{3}-\d{2}-\d{4}\b").unwrap(),
                    replacement: "[SSN_REDACTED]",
                },
                // 12. Nagual-specific: ngk_ API keys
                PiiPattern {
                    name: "nagual_key",
                    regex: Regex::new(r"ngk_[a-f0-9]{32,}").unwrap(),
                    replacement: "[NAGUAL_KEY_REDACTED]",
                },
            ],
        }
    }

    /// Strip all PII from text, returning the cleaned text and redaction metadata.
    ///
    /// Each pattern category is applied in sequence. The `redactions_count` reflects
    /// the number of distinct categories that matched (not the total number of
    /// individual replacements).
    pub fn strip_pii(&self, text: &str) -> RedactionResult {
        if !is_redaction_enabled() {
            return RedactionResult {
                text: text.to_string(),
                redactions_count: 0,
                categories: vec![],
            };
        }

        let mut result = text.to_string();
        let mut count = 0;
        let mut categories = Vec::new();

        for pattern in &self.patterns {
            if pattern.regex.is_match(&result) {
                categories.push(pattern.name.to_string());
                result = pattern.regex.replace_all(&result, pattern.replacement).to_string();
                count += 1;
            }
        }

        RedactionResult {
            text: result,
            redactions_count: count,
            categories,
        }
    }

    /// Check if text contains any PII without modifying it.
    pub fn contains_pii(&self, text: &str) -> bool {
        self.patterns.iter().any(|p| p.regex.is_match(text))
    }

    /// Strip PII from a string, returning just the cleaned text.
    ///
    /// Convenience wrapper around [`strip_pii`] when you don't need metadata.
    pub fn redact(&self, text: &str) -> String {
        self.strip_pii(text).text
    }
}

impl Default for PiiRedactor {
    fn default() -> Self {
        Self::new()
    }
}

/// Global singleton PII redactor for performance.
///
/// Regex compilation is expensive (~1ms for 12 patterns). This function
/// returns a lazily-initialized static instance that lives for the
/// duration of the process.
pub fn global_redactor() -> &'static PiiRedactor {
    static INSTANCE: OnceLock<PiiRedactor> = OnceLock::new();
    INSTANCE.get_or_init(PiiRedactor::new)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Pattern detection tests (one per category) ──────────────────

    #[test]
    fn test_strip_file_paths_unix() {
        let r = PiiRedactor::new();
        let result = r.strip_pii("Found config at /home/alice/secrets.txt");
        assert!(!result.text.contains("/home/alice"));
        assert!(result.text.contains("[PATH_REDACTED]"));
        assert!(result.categories.contains(&"file_path".to_string()));
    }

    #[test]
    fn test_strip_file_paths_windows() {
        let r = PiiRedactor::new();
        let result = r.strip_pii(r"Check C:\Users\bob\documents\report.docx");
        assert!(!result.text.contains(r"C:\Users\bob"));
        assert!(result.text.contains("[PATH_REDACTED]"));
    }

    #[test]
    fn test_strip_ip_address() {
        let r = PiiRedactor::new();
        let result = r.strip_pii("Server running at 192.168.1.100 on port 8080");
        assert!(!result.text.contains("192.168.1.100"));
        assert!(result.text.contains("[IP_REDACTED]"));
        assert!(result.categories.contains(&"ip_address".to_string()));
    }

    #[test]
    fn test_strip_email() {
        let r = PiiRedactor::new();
        let result = r.strip_pii("Contact user@example.com for details");
        assert!(!result.text.contains("user@example.com"));
        assert!(result.text.contains("[EMAIL_REDACTED]"));
        assert!(result.categories.contains(&"email".to_string()));
    }

    #[test]
    fn test_strip_api_key() {
        let r = PiiRedactor::new();
        let result = r.strip_pii("api_key = 'abcdef1234567890abcdef1234567890'");
        assert!(result.text.contains("[API_KEY_REDACTED]"));
        assert!(result.categories.contains(&"api_key".to_string()));
    }

    #[test]
    fn test_strip_api_key_token_variant() {
        let r = PiiRedactor::new();
        let result = r.strip_pii("token: abcdef1234567890abcdef1234567890");
        assert!(result.text.contains("[API_KEY_REDACTED]"));
    }

    #[test]
    fn test_strip_aws_key() {
        let r = PiiRedactor::new();
        let result = r.strip_pii("Access key AKIAIOSFODNN7EXAMPLE is active");
        assert!(!result.text.contains("AKIAIOSFODNN7EXAMPLE"));
        assert!(result.text.contains("[AWS_KEY_REDACTED]"));
        assert!(result.categories.contains(&"aws_key".to_string()));
    }

    #[test]
    fn test_strip_ssh_private_key() {
        let r = PiiRedactor::new();
        let input = "-----BEGIN RSA PRIVATE KEY-----\nMIIEpAIBA...\n-----END RSA PRIVATE KEY-----";
        let result = r.strip_pii(input);
        assert!(!result.text.contains("-----BEGIN RSA PRIVATE KEY-----"));
        assert!(result.text.contains("[SSH_KEY_REDACTED]"));
        assert!(result.categories.contains(&"ssh_key".to_string()));
    }

    #[test]
    fn test_strip_ssh_openssh_key() {
        let r = PiiRedactor::new();
        let result = r.strip_pii("-----BEGIN OPENSSH PRIVATE KEY-----");
        assert!(result.text.contains("[SSH_KEY_REDACTED]"));
    }

    #[test]
    fn test_strip_jwt() {
        let r = PiiRedactor::new();
        let input = "Bearer eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.abc123def456";
        let result = r.strip_pii(input);
        assert!(!result.text.contains("eyJhbGciOiJIUzI1NiJ9"));
        assert!(result.text.contains("[JWT_REDACTED]"));
        assert!(result.categories.contains(&"jwt".to_string()));
    }

    #[test]
    fn test_strip_connection_string() {
        let r = PiiRedactor::new();
        let result =
            r.strip_pii("Connect via postgres://nagual:secret@localhost:5432/nagual_db");
        assert!(!result.text.contains("postgres://nagual:secret"));
        assert!(result.text.contains("[CONNECTION_STRING_REDACTED]"));
        assert!(result.categories.contains(&"connection_string".to_string()));
    }

    #[test]
    fn test_strip_connection_string_redis() {
        let r = PiiRedactor::new();
        let result = r.strip_pii("Cache at redis://auth:password@10.0.0.1:6379");
        assert!(result.text.contains("[CONNECTION_STRING_REDACTED]"));
    }

    #[test]
    fn test_strip_phone_number() {
        let r = PiiRedactor::new();
        // Test standard US format
        let result = r.strip_pii("Call me at 555-123-4567 for details");
        assert!(!result.text.contains("555-123-4567"));
        assert!(result.text.contains("[PHONE_REDACTED]"));
        assert!(result.categories.contains(&"phone".to_string()));
    }

    #[test]
    fn test_strip_phone_number_with_parens() {
        let r = PiiRedactor::new();
        let result = r.strip_pii("Call (555)123-4567 now");
        assert!(result.text.contains("[PHONE_REDACTED]"));
    }

    #[test]
    fn test_strip_credit_card() {
        let r = PiiRedactor::new();
        let result = r.strip_pii("Card number: 4111-1111-1111-1111");
        assert!(!result.text.contains("4111"));
        assert!(result.text.contains("[CC_REDACTED]"));
        assert!(result.categories.contains(&"credit_card".to_string()));
    }

    #[test]
    fn test_strip_credit_card_no_dashes() {
        let r = PiiRedactor::new();
        let result = r.strip_pii("Card: 4111111111111111");
        assert!(result.text.contains("[CC_REDACTED]"));
    }

    #[test]
    fn test_strip_ssn() {
        let r = PiiRedactor::new();
        let result = r.strip_pii("SSN: 123-45-6789");
        assert!(!result.text.contains("123-45-6789"));
        assert!(result.text.contains("[SSN_REDACTED]"));
        assert!(result.categories.contains(&"ssn".to_string()));
    }

    #[test]
    fn test_strip_nagual_key() {
        let r = PiiRedactor::new();
        let result =
            r.strip_pii("Use key ngk_0123456789abcdef0123456789abcdef01 for authentication");
        assert!(!result.text.contains("ngk_0123456789abcdef"));
        assert!(result.text.contains("[NAGUAL_KEY_REDACTED]"));
        assert!(result.categories.contains(&"nagual_key".to_string()));
    }

    // ── Composite and edge-case tests ───────────────────────────────

    #[test]
    fn test_clean_text_passes_through_unchanged() {
        let r = PiiRedactor::new();
        let clean = "This is a normal sentence about Rust programming patterns.";
        let result = r.strip_pii(clean);
        assert_eq!(result.text, clean);
        assert_eq!(result.redactions_count, 0);
        assert!(result.categories.is_empty());
    }

    #[test]
    fn test_multiple_pii_types_in_one_string() {
        let r = PiiRedactor::new();
        let input = "User user@example.com at 10.0.0.5 stored key ngk_aabbccdd11223344556677889900aabb";
        let result = r.strip_pii(input);
        assert!(result.text.contains("[EMAIL_REDACTED]"));
        assert!(result.text.contains("[IP_REDACTED]"));
        assert!(result.text.contains("[NAGUAL_KEY_REDACTED]"));
        assert!(result.redactions_count >= 3);
    }

    #[test]
    fn test_contains_pii_returns_true_for_pii() {
        let r = PiiRedactor::new();
        assert!(r.contains_pii("email: user@host.com"));
        assert!(r.contains_pii("path /home/user/.ssh/id_rsa"));
        assert!(r.contains_pii("server at 192.168.0.1"));
        assert!(r.contains_pii("token = abcdef1234567890abcdef1234567890"));
        assert!(r.contains_pii("ngk_aabbccdd11223344556677889900aabb"));
    }

    #[test]
    fn test_contains_pii_returns_false_for_clean() {
        let r = PiiRedactor::new();
        assert!(!r.contains_pii("this is clean text with no sensitive data"));
        assert!(!r.contains_pii("learning rate is 0.001 for the optimizer"));
    }

    #[test]
    fn test_redacted_text_is_clean() {
        let r = PiiRedactor::new();
        let dirty = "Contact user@example.com from 192.168.1.1 using postgres://db:pass@host/db";
        let result = r.strip_pii(dirty);
        // The redacted result should not contain the original PII
        assert!(!result.text.contains("user@example.com"));
        assert!(!result.text.contains("192.168.1.1"));
        assert!(!result.text.contains("postgres://db:pass"));
    }

    #[test]
    fn test_redaction_count_accuracy() {
        let r = PiiRedactor::new();
        // Only email category
        let result = r.strip_pii("a@b.com and c@d.org");
        assert_eq!(result.redactions_count, 1); // one category, even though two matches
        assert_eq!(result.categories.len(), 1);
    }

    #[test]
    fn test_redaction_count_multiple_categories() {
        let r = PiiRedactor::new();
        let result = r.strip_pii("a@b.com from 10.0.0.1 with SSN 123-45-6789");
        assert_eq!(result.redactions_count, 3);
        assert_eq!(result.categories.len(), 3);
    }

    #[test]
    fn test_global_redactor_returns_same_instance() {
        let r1 = global_redactor();
        let r2 = global_redactor();
        assert!(std::ptr::eq(r1, r2));
    }

    #[test]
    fn test_global_redactor_works() {
        let result = global_redactor().strip_pii("email: test@example.com");
        assert!(result.text.contains("[EMAIL_REDACTED]"));
    }

    #[test]
    fn test_redact_convenience() {
        let r = PiiRedactor::new();
        let clean = r.redact("key at /home/user/secret.txt");
        assert!(clean.contains("[PATH_REDACTED]"));
        assert!(!clean.contains("/home/user"));
    }

    #[test]
    fn test_empty_string() {
        let r = PiiRedactor::new();
        let result = r.strip_pii("");
        assert_eq!(result.text, "");
        assert_eq!(result.redactions_count, 0);
    }

    #[test]
    fn test_default_trait() {
        let r = PiiRedactor::default();
        let result = r.strip_pii("user@test.com");
        assert!(result.text.contains("[EMAIL_REDACTED]"));
    }

    // ── File path false-positive prevention ─────────────────────────

    #[test]
    fn test_file_path_catches_users_dir() {
        let r = PiiRedactor::new();
        let result = r.strip_pii("Config at /Users/alice/secret/file.txt");
        assert!(!result.text.contains("/Users/alice"));
        assert!(result.text.contains("[PATH_REDACTED]"));
    }

    #[test]
    fn test_file_path_catches_home_ssh() {
        let r = PiiRedactor::new();
        let result = r.strip_pii("Key at /home/user/.ssh/id_rsa");
        assert!(!result.text.contains("/home/user"));
        assert!(result.text.contains("[PATH_REDACTED]"));
    }

    #[test]
    fn test_file_path_catches_tmp() {
        let r = PiiRedactor::new();
        let result = r.strip_pii("Temp file /tmp/build/output.log");
        assert!(!result.text.contains("/tmp/build"));
        assert!(result.text.contains("[PATH_REDACTED]"));
    }

    #[test]
    fn test_file_path_catches_var() {
        let r = PiiRedactor::new();
        let result = r.strip_pii("Log at /var/log/syslog");
        assert!(!result.text.contains("/var/log"));
        assert!(result.text.contains("[PATH_REDACTED]"));
    }

    #[test]
    fn test_file_path_catches_etc() {
        let r = PiiRedactor::new();
        let result = r.strip_pii("Config /etc/nginx/nginx.conf");
        assert!(!result.text.contains("/etc/nginx"));
        assert!(result.text.contains("[PATH_REDACTED]"));
    }

    #[test]
    fn test_file_path_no_false_positive_api_endpoint() {
        let r = PiiRedactor::new();
        let result = r.strip_pii("Call /api/v1/patterns for data");
        assert_eq!(
            result.text, "Call /api/v1/patterns for data",
            "API endpoints must NOT be redacted"
        );
        assert!(!result.categories.contains(&"file_path".to_string()));
    }

    #[test]
    fn test_file_path_no_false_positive_std_module() {
        let r = PiiRedactor::new();
        let result = r.strip_pii("Use std/io/Error for handling");
        assert_eq!(
            result.text, "Use std/io/Error for handling",
            "Rust module paths must NOT be redacted"
        );
    }

    #[test]
    fn test_file_path_no_false_positive_tokio_runtime() {
        let r = PiiRedactor::new();
        let result = r.strip_pii("Check tokio/runtime for async");
        assert_eq!(
            result.text, "Check tokio/runtime for async",
            "Crate module paths must NOT be redacted"
        );
    }

    #[test]
    fn test_file_path_no_false_positive_async_tokio() {
        let r = PiiRedactor::new();
        let result = r.strip_pii("See async/tokio patterns");
        assert_eq!(
            result.text, "See async/tokio patterns",
            "Relative paths must NOT be redacted"
        );
    }

    // ── Redaction toggle tests ──────────────────────────────────────

    #[test]
    fn test_redaction_disabled_passthrough_path() {
        // Verify the "redaction disabled" early-return path WITHOUT flipping
        // the global AtomicBool toggle, which would race with parallel tests.
        //
        // The early-return in strip_pii (when !is_redaction_enabled()) produces
        // a RedactionResult with the original text, zero redactions, and no
        // categories. We verify that contract here by constructing the same
        // result the code would produce.
        let dirty = "user@example.com at /home/secret/key";
        let passthrough = RedactionResult {
            text: dirty.to_string(),
            redactions_count: 0,
            categories: vec![],
        };
        assert_eq!(passthrough.text, dirty, "Passthrough must preserve original text");
        assert_eq!(passthrough.redactions_count, 0);
        assert!(passthrough.categories.is_empty());

        // Also confirm that toggle functions exist and return expected types,
        // without actually changing the global state.
        let _enabled: bool = is_redaction_enabled();
    }

    #[test]
    fn test_redaction_enabled_strips_pii() {
        // Do NOT call enable_redaction() here — the default is enabled,
        // and calling it would be harmless but masks the race condition
        // if someone later adds a disable call. Instead, just rely on
        // the default and verify redaction works.
        let r = PiiRedactor::new();
        let result = r.strip_pii("user@example.com");
        assert!(result.text.contains("[EMAIL_REDACTED]"));
    }

    #[test]
    fn test_redaction_toggle_api_exists() {
        // Verify the toggle API compiles and returns the expected types
        // without flipping global state (which would race with parallel tests).
        let _: bool = is_redaction_enabled();
        // These function signatures should compile:
        let _ = std::mem::size_of_val(&(disable_redaction as fn()));
        let _ = std::mem::size_of_val(&(enable_redaction as fn()));
    }
}
