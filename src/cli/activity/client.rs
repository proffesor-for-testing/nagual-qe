//! HTTP client for Screenpipe's REST API.
//!
//! Covers the full v2 API surface including search, raw SQL, embeddings,
//! tags, speakers, UI events, semantic search, and pipe management.

use chrono::{DateTime, Utc};

use crate::error::Result;

use super::helpers::urlencoding;
use super::types::*;

/// Search parameters for the Screenpipe `/search` endpoint.
#[derive(Default)]
pub(super) struct SearchParams<'a> {
    pub query: &'a str,
    pub content_type: &'a str,
    pub start: Option<&'a DateTime<Utc>>,
    pub end: Option<&'a DateTime<Utc>>,
    pub app_name: Option<&'a str>,
    pub window_name: Option<&'a str>,
    pub focused: Option<bool>,
    pub min_length: Option<usize>,
    pub limit: usize,
    pub offset: usize,
}

/// HTTP client for Screenpipe's REST API (v2).
pub(super) struct ScreenpipeClient {
    base_url: String,
    http: reqwest::Client,
}

impl ScreenpipeClient {
    pub fn new(base_url: &str) -> Self {
        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            http: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(30))
                .build()
                .unwrap_or_default(),
        }
    }

    // ----- Existing endpoints -----

    /// Check if Screenpipe is reachable.
    pub async fn health(&self) -> Option<HealthResponse> {
        let url = format!("{}/health", self.base_url);
        match self.http.get(&url).send().await {
            Ok(resp) if resp.status().is_success() => resp.json().await.ok(),
            _ => None,
        }
    }

    /// List audio devices.
    pub async fn audio_devices(&self) -> Vec<AudioDevice> {
        let url = format!("{}/audio/list", self.base_url);
        let resp = match self.http.get(&url).send().await {
            Ok(r) if r.status().is_success() => r,
            _ => return vec![],
        };
        resp.json().await.unwrap_or_default()
    }

    /// List vision monitors.
    pub async fn vision_monitors(&self) -> Vec<VisionMonitor> {
        let url = format!("{}/vision/list", self.base_url);
        let resp = match self.http.get(&url).send().await {
            Ok(r) if r.status().is_success() => r,
            _ => return vec![],
        };
        resp.json().await.unwrap_or_default()
    }

    /// Execute raw SQL against Screenpipe's internal SQLite database.
    pub async fn raw_sql(&self, query: &str) -> Result<Vec<serde_json::Value>> {
        let url = format!("{}/raw_sql", self.base_url);
        let body = serde_json::json!({ "query": query });
        let resp = self
            .http
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| crate::error::NagualError::Http(e.to_string()))?;
        if !resp.status().is_success() {
            return Err(crate::error::NagualError::Http(format!(
                "raw_sql returned {}",
                resp.status()
            )));
        }
        resp.json()
            .await
            .map_err(|e| crate::error::NagualError::Http(e.to_string()))
    }

    /// Generate embeddings via Screenpipe's `/v1/embeddings` (768-dim, local).
    pub async fn embed(&self, text: &str) -> Option<Vec<f32>> {
        let url = format!("{}/v1/embeddings", self.base_url);
        let body = serde_json::json!({ "input": text, "model": "default" });
        let resp = self.http.post(&url).json(&body).send().await.ok()?;
        if !resp.status().is_success() {
            return None;
        }
        let json: serde_json::Value = resp.json().await.ok()?;
        json.get("data")
            .and_then(|d| d.get(0))
            .and_then(|item| item.get("embedding"))
            .and_then(|e| serde_json::from_value(e.clone()).ok())
    }

    /// Search Screenpipe content with full v2 parameter support.
    pub async fn search(&self, params: &SearchParams<'_>) -> Result<ScreenpipeSearchResponse> {
        let mut url = format!(
            "{}/search?q={}&limit={}&offset={}",
            self.base_url,
            urlencoding(params.query),
            params.limit,
            params.offset,
        );
        if let Some(start) = params.start {
            url.push_str(&format!(
                "&start_time={}",
                start.format("%Y-%m-%dT%H:%M:%SZ")
            ));
        }
        if let Some(end) = params.end {
            url.push_str(&format!(
                "&end_time={}",
                end.format("%Y-%m-%dT%H:%M:%SZ")
            ));
        }
        if params.content_type != "all" {
            url.push_str(&format!("&content_type={}", params.content_type));
        }
        if let Some(app) = params.app_name {
            url.push_str(&format!("&app_name={}", urlencoding(app)));
        }
        if let Some(win) = params.window_name {
            url.push_str(&format!("&window_name={}", urlencoding(win)));
        }
        if let Some(true) = params.focused {
            url.push_str("&focused=true");
        }
        if let Some(min) = params.min_length {
            url.push_str(&format!("&min_length={}", min));
        }

        let resp = self
            .http
            .get(&url)
            .send()
            .await
            .map_err(|e| crate::error::NagualError::Http(e.to_string()))?;

        if !resp.status().is_success() {
            return Err(crate::error::NagualError::Http(format!(
                "Screenpipe returned {}",
                resp.status()
            )));
        }

        resp.json()
            .await
            .map_err(|e| crate::error::NagualError::Http(e.to_string()))
    }

    // ----- Tags -----

    /// Add tags to a screenpipe content item.
    pub async fn add_tags(
        &self,
        content_type: &str,
        id: i64,
        tags: Vec<String>,
    ) -> Result<()> {
        let url = format!("{}/tags/{}/{}", self.base_url, content_type, id);
        let body = serde_json::json!({ "tags": tags });
        let resp = self
            .http
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| crate::error::NagualError::Http(e.to_string()))?;
        if !resp.status().is_success() {
            return Err(crate::error::NagualError::Http(format!(
                "add_tags returned {}",
                resp.status()
            )));
        }
        Ok(())
    }

    /// Remove tags from a screenpipe content item.
    pub async fn remove_tags(
        &self,
        content_type: &str,
        id: i64,
        tags: Vec<String>,
    ) -> Result<()> {
        let url = format!("{}/tags/{}/{}", self.base_url, content_type, id);
        let body = serde_json::json!({ "tags": tags });
        let resp = self
            .http
            .delete(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| crate::error::NagualError::Http(e.to_string()))?;
        if !resp.status().is_success() {
            return Err(crate::error::NagualError::Http(format!(
                "remove_tags returned {}",
                resp.status()
            )));
        }
        Ok(())
    }

    // ----- Speakers -----

    /// Search speakers by name.
    pub async fn speakers_search(&self, name: Option<&str>) -> Result<Vec<Speaker>> {
        let mut url = format!("{}/speakers/search", self.base_url);
        if let Some(n) = name {
            url.push_str(&format!("?name={}", urlencoding(n)));
        }
        let resp = self
            .http
            .get(&url)
            .send()
            .await
            .map_err(|e| crate::error::NagualError::Http(e.to_string()))?;
        if !resp.status().is_success() {
            return Err(crate::error::NagualError::Http(format!(
                "speakers_search returned {}",
                resp.status()
            )));
        }
        resp.json()
            .await
            .map_err(|e| crate::error::NagualError::Http(e.to_string()))
    }

    /// List unnamed speakers.
    pub async fn speakers_unnamed(
        &self,
        limit: usize,
        offset: usize,
    ) -> Result<Vec<Speaker>> {
        let url = format!(
            "{}/speakers/unnamed?limit={}&offset={}",
            self.base_url, limit, offset
        );
        let resp = self
            .http
            .get(&url)
            .send()
            .await
            .map_err(|e| crate::error::NagualError::Http(e.to_string()))?;
        if !resp.status().is_success() {
            return Err(crate::error::NagualError::Http(format!(
                "speakers_unnamed returned {}",
                resp.status()
            )));
        }
        resp.json()
            .await
            .map_err(|e| crate::error::NagualError::Http(e.to_string()))
    }

    /// Update a speaker's name.
    pub async fn speakers_update(&self, id: i64, name: &str) -> Result<()> {
        let url = format!("{}/speakers/update", self.base_url);
        let body = serde_json::json!({ "id": id, "name": name });
        let resp = self
            .http
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| crate::error::NagualError::Http(e.to_string()))?;
        if !resp.status().is_success() {
            return Err(crate::error::NagualError::Http(format!(
                "speakers_update returned {}",
                resp.status()
            )));
        }
        Ok(())
    }

    /// Delete a speaker.
    pub async fn speakers_delete(&self, id: i64) -> Result<()> {
        let url = format!("{}/speakers/delete", self.base_url);
        let body = serde_json::json!({ "id": id });
        let resp = self
            .http
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| crate::error::NagualError::Http(e.to_string()))?;
        if !resp.status().is_success() {
            return Err(crate::error::NagualError::Http(format!(
                "speakers_delete returned {}",
                resp.status()
            )));
        }
        Ok(())
    }

    /// Merge two speakers (keep one, absorb the other).
    pub async fn speakers_merge(&self, keep_id: i64, merge_id: i64) -> Result<()> {
        let url = format!("{}/speakers/merge", self.base_url);
        let body =
            serde_json::json!({ "speaker_to_keep": keep_id, "speaker_to_merge": merge_id });
        let resp = self
            .http
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| crate::error::NagualError::Http(e.to_string()))?;
        if !resp.status().is_success() {
            return Err(crate::error::NagualError::Http(format!(
                "speakers_merge returned {}",
                resp.status()
            )));
        }
        Ok(())
    }

    /// Find speakers similar to a given speaker.
    pub async fn speakers_similar(
        &self,
        speaker_id: i64,
        limit: usize,
    ) -> Result<Vec<Speaker>> {
        let url = format!(
            "{}/speakers/similar?speaker_id={}&limit={}",
            self.base_url, speaker_id, limit
        );
        let resp = self
            .http
            .get(&url)
            .send()
            .await
            .map_err(|e| crate::error::NagualError::Http(e.to_string()))?;
        if !resp.status().is_success() {
            return Err(crate::error::NagualError::Http(format!(
                "speakers_similar returned {}",
                resp.status()
            )));
        }
        resp.json()
            .await
            .map_err(|e| crate::error::NagualError::Http(e.to_string()))
    }

    // ----- UI Events -----

    /// Query UI events (clicks, keystrokes, clipboard, app switches).
    pub async fn ui_events(&self, params: &UiEventsParams<'_>) -> Result<UiEventsResponse> {
        let mut url = format!(
            "{}/ui-events?limit={}&offset={}",
            self.base_url, params.limit, params.offset
        );
        if let Some(s) = params.start_time {
            url.push_str(&format!("&start_time={}", s));
        }
        if let Some(e) = params.end_time {
            url.push_str(&format!("&end_time={}", e));
        }
        if let Some(t) = params.event_type {
            url.push_str(&format!("&event_type={}", urlencoding(t)));
        }
        if let Some(app) = params.app_name {
            url.push_str(&format!("&app_name={}", urlencoding(app)));
        }
        if let Some(win) = params.window_name {
            url.push_str(&format!("&window_name={}", urlencoding(win)));
        }

        let resp = self
            .http
            .get(&url)
            .send()
            .await
            .map_err(|e| crate::error::NagualError::Http(e.to_string()))?;
        if !resp.status().is_success() {
            return Err(crate::error::NagualError::Http(format!(
                "ui_events returned {}",
                resp.status()
            )));
        }
        resp.json()
            .await
            .map_err(|e| crate::error::NagualError::Http(e.to_string()))
    }

    /// Get UI event statistics grouped by event type.
    pub async fn ui_events_stats(
        &self,
        start: &str,
        end: &str,
    ) -> Result<Vec<UiEventStat>> {
        let url = format!(
            "{}/ui-events/stats?start_time={}&end_time={}",
            self.base_url, start, end
        );
        let resp = self
            .http
            .get(&url)
            .send()
            .await
            .map_err(|e| crate::error::NagualError::Http(e.to_string()))?;
        if !resp.status().is_success() {
            return Err(crate::error::NagualError::Http(format!(
                "ui_events_stats returned {}",
                resp.status()
            )));
        }
        resp.json()
            .await
            .map_err(|e| crate::error::NagualError::Http(e.to_string()))
    }

    // ----- Search variants -----

    /// Semantic (embedding-based) search across Screenpipe content.
    pub async fn semantic_search(
        &self,
        text: &str,
        limit: usize,
        threshold: f32,
        content_type: Option<&str>,
    ) -> Result<SemanticSearchResponse> {
        let mut url = format!(
            "{}/semantic-search?q={}&limit={}&threshold={}",
            self.base_url,
            urlencoding(text),
            limit,
            threshold,
        );
        if let Some(ct) = content_type {
            url.push_str(&format!("&content_type={}", ct));
        }
        let resp = self
            .http
            .get(&url)
            .send()
            .await
            .map_err(|e| crate::error::NagualError::Http(e.to_string()))?;
        if !resp.status().is_success() {
            return Err(crate::error::NagualError::Http(format!(
                "semantic_search returned {}",
                resp.status()
            )));
        }
        resp.json()
            .await
            .map_err(|e| crate::error::NagualError::Http(e.to_string()))
    }

    /// Keyword search with text positions.
    pub async fn keyword_search(
        &self,
        keyword: &str,
        limit: usize,
        content_type: Option<&str>,
        start: Option<&DateTime<Utc>>,
        end: Option<&DateTime<Utc>>,
    ) -> Result<ScreenpipeSearchResponse> {
        let mut url = format!(
            "{}/search/keyword?k={}&limit={}",
            self.base_url,
            urlencoding(keyword),
            limit,
        );
        if let Some(ct) = content_type {
            url.push_str(&format!("&content_type={}", ct));
        }
        if let Some(s) = start {
            url.push_str(&format!(
                "&start_time={}",
                s.format("%Y-%m-%dT%H:%M:%SZ")
            ));
        }
        if let Some(e) = end {
            url.push_str(&format!(
                "&end_time={}",
                e.format("%Y-%m-%dT%H:%M:%SZ")
            ));
        }
        let resp = self
            .http
            .get(&url)
            .send()
            .await
            .map_err(|e| crate::error::NagualError::Http(e.to_string()))?;
        if !resp.status().is_success() {
            return Err(crate::error::NagualError::Http(format!(
                "keyword_search returned {}",
                resp.status()
            )));
        }
        resp.json()
            .await
            .map_err(|e| crate::error::NagualError::Http(e.to_string()))
    }

    // ----- Pipes -----

    /// List all installed Screenpipe pipes (plugins).
    pub async fn pipes_list(&self) -> Result<Vec<PipeInfo>> {
        let url = format!("{}/pipes/list", self.base_url);
        let resp = self
            .http
            .get(&url)
            .send()
            .await
            .map_err(|e| crate::error::NagualError::Http(e.to_string()))?;
        if !resp.status().is_success() {
            return Err(crate::error::NagualError::Http(format!(
                "pipes_list returned {}",
                resp.status()
            )));
        }
        resp.json()
            .await
            .map_err(|e| crate::error::NagualError::Http(e.to_string()))
    }

    /// Get info about a specific pipe.
    pub async fn pipes_info(&self, pipe_id: &str) -> Result<PipeInfo> {
        let url = format!(
            "{}/pipes/info?pipe_id={}",
            self.base_url,
            urlencoding(pipe_id)
        );
        let resp = self
            .http
            .get(&url)
            .send()
            .await
            .map_err(|e| crate::error::NagualError::Http(e.to_string()))?;
        if !resp.status().is_success() {
            return Err(crate::error::NagualError::Http(format!(
                "pipes_info returned {}",
                resp.status()
            )));
        }
        resp.json()
            .await
            .map_err(|e| crate::error::NagualError::Http(e.to_string()))
    }

    /// Enable a pipe.
    pub async fn pipes_enable(&self, pipe_id: &str) -> Result<()> {
        let url = format!("{}/pipes/enable", self.base_url);
        let body = serde_json::json!({ "pipe_id": pipe_id });
        let resp = self
            .http
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| crate::error::NagualError::Http(e.to_string()))?;
        if !resp.status().is_success() {
            return Err(crate::error::NagualError::Http(format!(
                "pipes_enable returned {}",
                resp.status()
            )));
        }
        Ok(())
    }

    /// Disable a pipe.
    pub async fn pipes_disable(&self, pipe_id: &str) -> Result<()> {
        let url = format!("{}/pipes/disable", self.base_url);
        let body = serde_json::json!({ "pipe_id": pipe_id });
        let resp = self
            .http
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| crate::error::NagualError::Http(e.to_string()))?;
        if !resp.status().is_success() {
            return Err(crate::error::NagualError::Http(format!(
                "pipes_disable returned {}",
                resp.status()
            )));
        }
        Ok(())
    }

    // ----- Vision -----

    /// Get vision capture status.
    #[allow(dead_code)]
    pub async fn vision_status(&self) -> Result<VisionStatus> {
        let url = format!("{}/vision/status", self.base_url);
        let resp = self
            .http
            .get(&url)
            .send()
            .await
            .map_err(|e| crate::error::NagualError::Http(e.to_string()))?;
        if !resp.status().is_success() {
            return Err(crate::error::NagualError::Http(format!(
                "vision_status returned {}",
                resp.status()
            )));
        }
        resp.json()
            .await
            .map_err(|e| crate::error::NagualError::Http(e.to_string()))
    }
}
