//! HTTP client for communicating with a remote nagual serve instance.

use chrono::{DateTime, Utc};
use tracing::debug;

use crate::error::{NagualError, Result};

use super::types::{CloudStatusResponse, PullResponse, PushRequest, PushResponse, SyncPatternData};

/// HTTP client for the nagual cloud sync API.
pub struct CloudClient {
    base_url: String,
    api_token: String,
    client: reqwest::Client,
}

impl CloudClient {
    /// Create a new cloud client.
    pub fn new(base_url: &str, api_token: &str) -> Self {
        let base_url = base_url.trim_end_matches('/').to_string();
        Self {
            base_url,
            api_token: api_token.to_string(),
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(60))
                .build()
                .unwrap_or_default(),
        }
    }

    /// Push patterns to the remote server.
    pub async fn push_patterns(&self, patterns: &[SyncPatternData]) -> Result<PushResponse> {
        let url = format!("{}/api/sync/push", self.base_url);
        let body = PushRequest {
            patterns: patterns.to_vec(),
        };

        debug!(url = %url, count = patterns.len(), "Pushing patterns to cloud");

        let response = self
            .client
            .post(&url)
            .bearer_auth(&self.api_token)
            .json(&body)
            .send()
            .await
            .map_err(|e| NagualError::internal(format!("Cloud push request failed: {}", e)))?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(NagualError::internal(format!(
                "Cloud push failed ({}): {}",
                status, body
            )));
        }

        response
            .json::<PushResponse>()
            .await
            .map_err(|e| NagualError::internal(format!("Failed to parse push response: {}", e)))
    }

    /// Pull patterns from the remote server modified since a given timestamp.
    pub async fn pull_patterns(
        &self,
        since: Option<DateTime<Utc>>,
        limit: usize,
        offset: usize,
    ) -> Result<PullResponse> {
        let mut url = format!(
            "{}/api/sync/pull?limit={}&offset={}",
            self.base_url, limit, offset
        );
        if let Some(since) = since {
            // Use Z suffix (not +00:00) to avoid URL-encoding issues with + in query params.
            // Use Nanos precision to match SQLite stored timestamps exactly.
            let since_str = since.to_rfc3339_opts(chrono::SecondsFormat::Nanos, true);
            url.push_str(&format!("&since={}", since_str));
        }

        debug!(url = %url, "Pulling patterns from cloud");

        let response = self
            .client
            .get(&url)
            .bearer_auth(&self.api_token)
            .send()
            .await
            .map_err(|e| NagualError::internal(format!("Cloud pull request failed: {}", e)))?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(NagualError::internal(format!(
                "Cloud pull failed ({}): {}",
                status, body
            )));
        }

        response
            .json::<PullResponse>()
            .await
            .map_err(|e| NagualError::internal(format!("Failed to parse pull response: {}", e)))
    }

    /// Check remote server status.
    pub async fn status(&self) -> Result<CloudStatusResponse> {
        let url = format!("{}/api/status", self.base_url);

        let response = self
            .client
            .get(&url)
            .bearer_auth(&self.api_token)
            .send()
            .await
            .map_err(|e| NagualError::internal(format!("Cloud status request failed: {}", e)))?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(NagualError::internal(format!(
                "Cloud status failed ({}): {}",
                status, body
            )));
        }

        // The /api/status endpoint returns a generic JSON; parse what we can
        let json: serde_json::Value = response
            .json()
            .await
            .map_err(|e| NagualError::internal(format!("Failed to parse status response: {}", e)))?;

        Ok(CloudStatusResponse {
            status: json
                .get("status")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown")
                .to_string(),
            pattern_count: json
                .get("patterns")
                .and_then(|v| v.get("count"))
                .and_then(|v| v.as_u64())
                .map(|v| v as usize),
            version: json
                .get("version")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string()),
        })
    }

    /// Get the base URL.
    pub fn base_url(&self) -> &str {
        &self.base_url
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_client_creation() {
        let client = CloudClient::new("https://nagual.example.com/", "test-token");
        assert_eq!(client.base_url(), "https://nagual.example.com");
    }

    #[test]
    fn test_client_trailing_slash_stripped() {
        let client = CloudClient::new("https://example.com///", "token");
        assert_eq!(client.base_url(), "https://example.com");
    }
}
