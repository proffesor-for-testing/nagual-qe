//! Client for an optional collective knowledge API ("Brain").
//!
//! Provides sharing and searching of patterns with a remote brain service,
//! with automatic PII stripping on all outbound data. This module is
//! gated behind the `brain-sync` feature flag.
//!
//! No default endpoint is configured — set `BRAIN_URL` explicitly to opt in.
//!
//! # Usage
//!
//! ```rust,ignore
//! use nagual::sync::brain::BrainClient;
//!
//! let client = BrainClient::new();
//! let id = client.share("rust", "Error handling", "Use thiserror...", vec!["rust".into()]).await?;
//! let results = client.search("error handling", Some("rust"), 10).await?;
//! ```

#[cfg(feature = "brain-sync")]
pub use inner::*;

#[cfg(feature = "brain-sync")]
mod inner {
    use reqwest::Client;
    use serde::{Deserialize, Serialize};

    use crate::sync::pii::global_redactor;

    /// Request body for sharing a memory with the collective brain.
    #[derive(Debug, Serialize)]
    struct ShareRequest {
        category: String,
        title: String,
        content: String,
        tags: Vec<String>,
        contributor: String,
    }

    /// A memory retrieved from the collective brain.
    /// Quality score from Pi Brain (Beta distribution parameters)
    #[derive(Debug, Clone, Deserialize)]
    #[serde(untagged)]
    pub enum BrainQuality {
        /// Beta distribution: {"alpha": f64, "beta": f64}
        Beta { alpha: f64, beta: f64 },
        /// Simple float
        Score(f64),
    }

    impl BrainQuality {
        /// Get the mean quality score
        pub fn mean(&self) -> f64 {
            match self {
                Self::Beta { alpha, beta } => alpha / (alpha + beta),
                Self::Score(s) => *s,
            }
        }
    }

    #[derive(Debug, Clone, Deserialize)]
    pub struct BrainMemory {
        /// Unique identifier for this memory
        pub id: String,
        /// Short title
        pub title: String,
        /// Full content
        pub content: String,
        /// Category (e.g., "rust", "testing")
        pub category: String,
        /// Associated tags
        #[serde(default)]
        pub tags: Vec<String>,
        /// Quality score (Beta distribution or plain float)
        #[serde(default = "default_quality")]
        pub quality_score: BrainQuality,
        /// Creation timestamp (ISO 8601)
        pub created_at: String,
    }

    fn default_quality() -> BrainQuality {
        BrainQuality::Score(0.5)
    }

    /// Client for an optional collective knowledge API ("Brain").
    ///
    /// All outbound data is automatically PII-stripped before transmission.
    /// The client reads configuration from environment variables:
    /// - `BRAIN_URL`: Base URL — required; no default is provided
    /// - `BRAIN_API_KEY`: Optional API key for authentication
    pub struct BrainClient {
        client: Client,
        base_url: String,
        api_key: Option<String>,
    }

    impl BrainClient {
        /// Create a new client using environment variables for configuration.
        /// If `BRAIN_URL` is unset the client falls back to `http://localhost:0`
        /// so that callers without a remote configured will simply fail
        /// requests rather than send data to an unintended endpoint.
        pub fn new() -> Self {
            Self {
                client: Client::new(),
                base_url: std::env::var("BRAIN_URL")
                    .unwrap_or_else(|_| "http://localhost:0".to_string()),
                api_key: std::env::var("BRAIN_API_KEY").ok(),
            }
        }

        /// Create a client with explicit configuration.
        pub fn with_config(base_url: String, api_key: Option<String>) -> Self {
            Self {
                client: Client::new(),
                base_url,
                api_key,
            }
        }

        /// Share a pattern with the collective brain.
        ///
        /// PII is automatically stripped from title and content before sharing.
        /// The contributor is always set to "nagual".
        pub async fn share(
            &self,
            category: &str,
            title: &str,
            content: &str,
            tags: Vec<String>,
        ) -> Result<String, BrainError> {
            // Strip PII before sharing
            let clean_title = global_redactor().strip_pii(title);
            let clean_content = global_redactor().strip_pii(content);

            if clean_title.redactions_count > 0 || clean_content.redactions_count > 0 {
                tracing::debug!(
                    title_redactions = clean_title.redactions_count,
                    content_redactions = clean_content.redactions_count,
                    "PII stripped before brain share"
                );
            }

            let req = ShareRequest {
                category: category.to_string(),
                title: clean_title.text,
                content: clean_content.text,
                tags,
                contributor: "nagual".to_string(),
            };

            let mut request = self
                .client
                .post(format!("{}/v1/memories", self.base_url))
                .json(&req);

            if let Some(key) = &self.api_key {
                request = request.header("Authorization", format!("Bearer {}", key));
            }

            let response = request.send().await.map_err(BrainError::Http)?;

            if !response.status().is_success() {
                let status = response.status().as_u16();
                let msg = response.text().await.unwrap_or_default();
                return Err(BrainError::Api(format!("Status {}: {}", status, msg)));
            }

            let body: serde_json::Value = response.json().await.map_err(BrainError::Http)?;
            Ok(body["id"].as_str().unwrap_or("unknown").to_string())
        }

        /// Search the collective brain for relevant memories.
        pub async fn search(
            &self,
            query: &str,
            category: Option<&str>,
            limit: usize,
        ) -> Result<Vec<BrainMemory>, BrainError> {
            // Build URL with query parameters using reqwest's built-in support
            let mut params = vec![
                ("q", query.to_string()),
                ("limit", limit.to_string()),
            ];

            if let Some(cat) = category {
                params.push(("category", cat.to_string()));
            }

            let mut request = self
                .client
                .get(format!("{}/v1/memories/search", self.base_url))
                .query(&params);

            if let Some(key) = &self.api_key {
                request = request.header("Authorization", format!("Bearer {}", key));
            }

            let response = request.send().await.map_err(BrainError::Http)?;

            if !response.status().is_success() {
                let status = response.status().as_u16();
                let msg = response.text().await.unwrap_or_default();
                return Err(BrainError::Api(format!("Status {}: {}", status, msg)));
            }

            // Pi Brain returns a plain JSON array, not {"memories": [...]}
            let memories: Vec<BrainMemory> =
                response.json().await.map_err(BrainError::Http)?;
            Ok(memories)
        }

        /// Get system status from the brain API.
        pub async fn status(&self) -> Result<serde_json::Value, BrainError> {
            let mut request = self
                .client
                .get(format!("{}/v1/status", self.base_url));

            if let Some(key) = &self.api_key {
                request = request.header("Authorization", format!("Bearer {}", key));
            }

            let response = request.send().await.map_err(BrainError::Http)?;

            if !response.status().is_success() {
                let status = response.status().as_u16();
                let msg = response.text().await.unwrap_or_default();
                return Err(BrainError::Api(format!("Status {}: {}", status, msg)));
            }

            response.json().await.map_err(BrainError::Http)
        }

        /// Get the configured base URL.
        pub fn base_url(&self) -> &str {
            &self.base_url
        }

        /// Check if an API key is configured.
        pub fn has_api_key(&self) -> bool {
            self.api_key.is_some()
        }
    }

    impl Default for BrainClient {
        fn default() -> Self {
            Self::new()
        }
    }

    /// Errors that can occur when communicating with the brain API.
    #[derive(Debug, thiserror::Error)]
    pub enum BrainError {
        /// HTTP transport error (network, DNS, TLS, etc.)
        #[error("HTTP error: {0}")]
        Http(#[from] reqwest::Error),
        /// API-level error (non-2xx response)
        #[error("API error: {0}")]
        Api(String),
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_client_default_construction() {
            // Clear env vars that might interfere
            std::env::remove_var("BRAIN_URL");
            std::env::remove_var("BRAIN_API_KEY");

            let client = BrainClient::new();
            assert_eq!(client.base_url(), "http://localhost:0");
            assert!(!client.has_api_key());
        }

        #[test]
        fn test_client_with_config() {
            let client = BrainClient::with_config(
                "https://custom.brain.api".to_string(),
                Some("test-key-123".to_string()),
            );
            assert_eq!(client.base_url(), "https://custom.brain.api");
            assert!(client.has_api_key());
        }

        #[test]
        fn test_client_default_trait() {
            let client = BrainClient::default();
            // Should not panic
            assert!(!client.base_url().is_empty());
        }

        #[test]
        fn test_share_request_serialization() {
            let req = ShareRequest {
                category: "rust".to_string(),
                title: "Test Title".to_string(),
                content: "Test content".to_string(),
                tags: vec!["rust".to_string(), "testing".to_string()],
                contributor: "nagual".to_string(),
            };

            let json = serde_json::to_value(&req).unwrap();
            assert_eq!(json["category"], "rust");
            assert_eq!(json["title"], "Test Title");
            assert_eq!(json["contributor"], "nagual");
        }

        #[test]
        fn test_brain_memory_deserialization() {
            let json = serde_json::json!({
                "id": "mem-123",
                "title": "Error Handling",
                "content": "Use thiserror for library errors",
                "category": "rust",
                "tags": ["error", "thiserror"],
                "quality_score": 0.85,
                "created_at": "2026-03-26T10:00:00Z"
            });

            let memory: BrainMemory = serde_json::from_value(json).unwrap();
            assert_eq!(memory.id, "mem-123");
            assert_eq!(memory.category, "rust");
            assert!((memory.quality_score.mean() - 0.85).abs() < 1e-6);
            assert_eq!(memory.tags.len(), 2);
        }

        #[test]
        fn test_pii_stripped_before_share_construction() {
            // Verify that the PII redactor is accessible and works
            // (the actual share method is async and needs a server)
            let redactor = global_redactor();
            let dirty_title = "Fix bug on server 192.168.1.1";
            let dirty_content = "Contact admin@company.com for credentials";

            let clean_title = redactor.strip_pii(dirty_title);
            let clean_content = redactor.strip_pii(dirty_content);

            assert!(clean_title.text.contains("[IP_REDACTED]"));
            assert!(!clean_title.text.contains("192.168.1.1"));
            assert!(clean_content.text.contains("[EMAIL_REDACTED]"));
            assert!(!clean_content.text.contains("admin@company.com"));
        }

        #[test]
        fn test_brain_error_display() {
            let err = BrainError::Api("Not Found".to_string());
            assert_eq!(format!("{}", err), "API error: Not Found");
        }

        // ── Async tests with wiremock ───────────────────────────────

        #[tokio::test]
        async fn test_share_makes_correct_request_with_pii_stripped() {
            use wiremock::matchers::{method, path};
            use wiremock::{Mock, MockServer, ResponseTemplate};

            let mock_server = MockServer::start().await;

            // Set up a mock that captures the request
            Mock::given(method("POST"))
                .and(path("/v1/memories"))
                .respond_with(
                    ResponseTemplate::new(200)
                        .set_body_json(serde_json::json!({"id": "mem-abc123"})),
                )
                .expect(1)
                .mount(&mock_server)
                .await;

            let client = BrainClient::with_config(mock_server.uri(), None);
            let result = client
                .share(
                    "rust",
                    "Fix bug on server 192.168.1.1",
                    "Contact admin@company.com for the SSH key",
                    vec!["rust".to_string()],
                )
                .await;

            assert!(result.is_ok(), "share should succeed: {:?}", result.err());
            let id = result.unwrap();
            assert_eq!(id, "mem-abc123");

            // Verify the server received the request with PII stripped
            let received = mock_server.received_requests().await.unwrap();
            assert_eq!(received.len(), 1);
            let body: serde_json::Value =
                serde_json::from_slice(&received[0].body).unwrap();
            let title = body["title"].as_str().unwrap();
            let content = body["content"].as_str().unwrap();

            assert!(
                title.contains("[IP_REDACTED]"),
                "IP should be redacted in title, got: {}",
                title
            );
            assert!(
                !title.contains("192.168.1.1"),
                "Raw IP must not appear in title"
            );
            assert!(
                content.contains("[EMAIL_REDACTED]"),
                "Email should be redacted in content, got: {}",
                content
            );
            assert!(
                !content.contains("admin@company.com"),
                "Raw email must not appear in content"
            );
        }

        #[tokio::test]
        async fn test_search_returns_results() {
            use wiremock::matchers::{method, path};
            use wiremock::{Mock, MockServer, ResponseTemplate};

            let mock_server = MockServer::start().await;

            let response_body = serde_json::json!([
                {
                    "id": "mem-001",
                    "title": "Error Handling Pattern",
                    "content": "Use thiserror for library errors",
                    "category": "rust",
                    "tags": ["error", "thiserror"],
                    "quality_score": {"alpha": 9.2, "beta": 0.8},
                    "created_at": "2026-03-26T10:00:00Z"
                },
                {
                    "id": "mem-002",
                    "title": "Async Pattern",
                    "content": "Prefer tokio::spawn for background tasks",
                    "category": "rust",
                    "tags": ["async", "tokio"],
                    "quality_score": 0.88,
                    "created_at": "2026-03-25T08:00:00Z"
                }
            ]);

            Mock::given(method("GET"))
                .and(path("/v1/memories/search"))
                .respond_with(
                    ResponseTemplate::new(200).set_body_json(response_body),
                )
                .expect(1)
                .mount(&mock_server)
                .await;

            let client = BrainClient::with_config(mock_server.uri(), None);
            let results = client.search("error handling", Some("rust"), 10).await;

            assert!(results.is_ok(), "search should succeed: {:?}", results.err());
            let memories = results.unwrap();
            assert_eq!(memories.len(), 2);
            assert_eq!(memories[0].id, "mem-001");
            assert_eq!(memories[0].title, "Error Handling Pattern");
            assert_eq!(memories[1].id, "mem-002");
            assert_eq!(memories[1].category, "rust");
        }

        #[tokio::test]
        async fn test_share_auth_header() {
            use wiremock::matchers::{header, method, path};
            use wiremock::{Mock, MockServer, ResponseTemplate};

            let mock_server = MockServer::start().await;

            // Only respond if the Authorization header is correct
            Mock::given(method("POST"))
                .and(path("/v1/memories"))
                .and(header("Authorization", "Bearer test-api-key-12345"))
                .respond_with(
                    ResponseTemplate::new(200)
                        .set_body_json(serde_json::json!({"id": "mem-auth"})),
                )
                .expect(1)
                .mount(&mock_server)
                .await;

            let client = BrainClient::with_config(
                mock_server.uri(),
                Some("test-api-key-12345".to_string()),
            );

            let result = client
                .share("testing", "Auth test", "Verify bearer token is sent", vec![])
                .await;

            assert!(result.is_ok(), "share with auth should succeed: {:?}", result.err());
            assert_eq!(result.unwrap(), "mem-auth");
        }

        #[tokio::test]
        async fn test_share_handles_server_error() {
            use wiremock::matchers::{method, path};
            use wiremock::{Mock, MockServer, ResponseTemplate};

            let mock_server = MockServer::start().await;

            Mock::given(method("POST"))
                .and(path("/v1/memories"))
                .respond_with(
                    ResponseTemplate::new(500)
                        .set_body_string("Internal Server Error"),
                )
                .expect(1)
                .mount(&mock_server)
                .await;

            let client = BrainClient::with_config(mock_server.uri(), None);
            let result = client
                .share("rust", "Test", "Content", vec![])
                .await;

            assert!(result.is_err());
            let err = result.unwrap_err();
            assert!(
                format!("{}", err).contains("500"),
                "Error should contain status code: {}",
                err
            );
        }
    }
}
