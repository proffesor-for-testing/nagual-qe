//! Shared types for cloud sync protocol.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::reasoning_bank::pattern::{Pattern, PatternCategory, PatternId};

/// Full pattern data for sync transfer.
///
/// Includes all fields needed to reconstruct a pattern on the remote side,
/// preserving the original ID and timestamps.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncPatternData {
    pub id: String,
    pub problem: String,
    pub solution: String,
    #[serde(default)]
    pub context: String,
    pub domain: String,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub reward: f32,
    #[serde(default = "default_confidence")]
    pub confidence: f32,
    #[serde(default = "default_true")]
    pub success: bool,
    #[serde(default)]
    pub reuse_count: u32,
    #[serde(default)]
    pub effectiveness: f32,
    pub created_at: String,
    pub updated_at: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub agent_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content_hash: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub critique: Option<String>,
}

fn default_confidence() -> f32 {
    0.5
}

fn default_true() -> bool {
    true
}

impl From<&Pattern> for SyncPatternData {
    fn from(p: &Pattern) -> Self {
        SyncPatternData {
            id: p.id().to_string(),
            problem: p.problem().to_string(),
            solution: p.solution().to_string(),
            context: p.context().to_string(),
            domain: p.category().to_string(),
            tags: p.tags().to_vec(),
            reward: p.reward(),
            confidence: p.confidence(),
            success: p.success(),
            reuse_count: p.reuse_count(),
            effectiveness: p.effectiveness(),
            created_at: p.timestamp().to_rfc3339(),
            updated_at: p.updated_at().to_rfc3339(),
            agent_id: p.agent_id().map(|s| s.to_string()),
            session_id: p.session_id().map(|s| s.to_string()),
            content_hash: p.content_hash().map(|s| s.to_string()),
            critique: {
                let c = p.critique();
                if c.is_empty() { None } else { Some(c.to_string()) }
            },
        }
    }
}

impl SyncPatternData {
    /// Convert to a Pattern using the builder.
    pub fn to_pattern(&self) -> Pattern {
        let timestamp = DateTime::parse_from_rfc3339(&self.created_at)
            .map(|dt| dt.with_timezone(&Utc))
            .unwrap_or_else(|_| Utc::now());

        let updated_at = DateTime::parse_from_rfc3339(&self.updated_at)
            .map(|dt| dt.with_timezone(&Utc))
            .unwrap_or_else(|_| Utc::now());

        let mut builder = Pattern::builder()
            .id(PatternId::from_string(&self.id))
            .timestamp(timestamp)
            .updated_at(updated_at)
            .problem(&self.problem)
            .solution(&self.solution)
            .context(&self.context)
            .category(PatternCategory::from(self.domain.as_str()))
            .tags(self.tags.clone())
            .reward(self.reward)
            .confidence(self.confidence)
            .success(self.success)
            .reuse_count(self.reuse_count)
            .effectiveness(self.effectiveness);

        if let Some(ref agent_id) = self.agent_id {
            builder = builder.agent_id(agent_id);
        }
        if let Some(ref session_id) = self.session_id {
            builder = builder.session_id(session_id);
        }
        if let Some(ref hash) = self.content_hash {
            builder = builder.content_hash(hash);
        }
        if let Some(ref critique) = self.critique {
            builder = builder.critique(critique);
        }

        builder.build()
    }
}

/// Request body for push endpoint.
#[derive(Debug, Serialize, Deserialize)]
pub struct PushRequest {
    pub patterns: Vec<SyncPatternData>,
}

/// Response from push endpoint.
#[derive(Debug, Serialize, Deserialize)]
pub struct PushResponse {
    pub received: usize,
    pub created: usize,
    pub updated: usize,
    pub skipped: usize,
}

/// Response from pull endpoint.
#[derive(Debug, Serialize, Deserialize)]
pub struct PullResponse {
    pub patterns: Vec<SyncPatternData>,
    pub total: usize,
    pub has_more: bool,
}

/// Cloud server status.
#[derive(Debug, Serialize, Deserialize)]
pub struct CloudStatusResponse {
    pub status: String,
    #[serde(default)]
    pub pattern_count: Option<usize>,
    #[serde(default)]
    pub version: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sync_pattern_data_roundtrip() {
        let pattern = Pattern::builder()
            .problem("test problem")
            .solution("test solution")
            .context("test context")
            .category(PatternCategory::from("rust"))
            .tags(vec!["async".to_string(), "tokio".to_string()])
            .reward(0.8)
            .confidence(0.9)
            .agent_id("test-agent")
            .build();

        let sync_data = SyncPatternData::from(&pattern);
        assert_eq!(sync_data.problem, "test problem");
        assert_eq!(sync_data.domain, "rust");
        assert_eq!(sync_data.tags, vec!["async", "tokio"]);

        let reconstructed = sync_data.to_pattern();
        assert_eq!(reconstructed.problem(), pattern.problem());
        assert_eq!(reconstructed.solution(), pattern.solution());
        assert_eq!(reconstructed.category().to_string(), pattern.category().to_string());
        assert_eq!(reconstructed.reward(), pattern.reward());
    }

    #[test]
    fn test_sync_pattern_data_serialization() {
        let data = SyncPatternData {
            id: "test-id".to_string(),
            problem: "problem".to_string(),
            solution: "solution".to_string(),
            context: String::new(),
            domain: "general".to_string(),
            tags: vec![],
            reward: 0.5,
            confidence: 0.5,
            success: true,
            reuse_count: 0,
            effectiveness: 0.5,
            created_at: "2026-01-01T00:00:00+00:00".to_string(),
            updated_at: "2026-01-01T00:00:00+00:00".to_string(),
            agent_id: None,
            session_id: None,
            content_hash: None,
            critique: None,
        };

        let json = serde_json::to_string(&data).unwrap();
        let deserialized: SyncPatternData = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.id, "test-id");
        assert_eq!(deserialized.problem, "problem");
    }

    #[test]
    fn test_push_request_serialization() {
        let req = PushRequest { patterns: vec![] };
        let json = serde_json::to_string(&req).unwrap();
        let deserialized: PushRequest = serde_json::from_str(&json).unwrap();
        assert!(deserialized.patterns.is_empty());
    }

    #[test]
    fn test_pull_response_serialization() {
        let resp = PullResponse {
            patterns: vec![],
            total: 0,
            has_more: false,
        };
        let json = serde_json::to_string(&resp).unwrap();
        let deserialized: PullResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.total, 0);
        assert!(!deserialized.has_more);
    }
}
