//! Pre-compaction flush: extract and persist patterns before context compaction.
//!
//! When a session's context is about to be compacted (truncated to fit the
//! context window), this module scans the pending context for:
//! 1. Unrecorded error resolutions
//! 2. Insights mentioned but not stored
//! 3. Code patterns applied but not captured
//! 4. Active task context that should persist

use axum::extract::State;
use axum::response::IntoResponse;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::Mutex as TokioMutex;

use crate::reasoning_bank::pattern::Pattern;
use crate::reasoning_bank::storage::PatternStorage;

use super::AppState;

/// Compaction flush configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompactionFlushConfig {
    /// Whether pre-compaction flush is enabled.
    pub enabled: bool,
    /// Minimum context items to trigger pattern extraction.
    pub min_items_threshold: usize,
    /// Maximum patterns to extract per flush.
    pub max_patterns_per_flush: usize,
}

impl Default for CompactionFlushConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            min_items_threshold: 5,
            max_patterns_per_flush: 10,
        }
    }
}

/// Context item that might contain extractable patterns.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextItem {
    pub content: String,
    pub item_type: ContextItemType,
    pub timestamp: String,
}

/// Type of context item, used to determine extraction priority.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ContextItemType {
    ErrorResolution,
    Insight,
    CodeChange,
    TaskResult,
    Unknown,
}

/// Result of a pre-compaction flush.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlushResult {
    pub patterns_extracted: usize,
    pub items_scanned: usize,
    pub errors: Vec<String>,
}

/// Request body for the flush endpoint.
#[derive(Debug, Deserialize)]
pub struct FlushRequest {
    pub context_items: Vec<ContextItem>,
}

/// Extract patterns from context items and store them before compaction.
///
/// High-value items (ErrorResolution, Insight) are stored as patterns.
/// TaskResult items are counted but handled separately.
/// CodeChange and Unknown items are skipped.
pub async fn flush_before_compaction(
    context_items: &[ContextItem],
    storage: &Arc<TokioMutex<PatternStorage>>,
    config: &CompactionFlushConfig,
) -> FlushResult {
    if !config.enabled || context_items.len() < config.min_items_threshold {
        return FlushResult {
            patterns_extracted: 0,
            items_scanned: context_items.len(),
            errors: Vec::new(),
        };
    }

    let mut extracted = 0;
    let mut errors = Vec::new();

    for item in context_items.iter() {
        if extracted >= config.max_patterns_per_flush {
            break;
        }

        match item.item_type {
            ContextItemType::ErrorResolution | ContextItemType::Insight => {
                // High-value items -- always extract as patterns
                let pattern = Pattern::builder()
                    .problem(&item.content)
                    .solution("Extracted during pre-compaction flush")
                    .category("compaction-flush".into())
                    .build();

                let guard = storage.lock().await;
                match guard.store_pattern(&pattern).await {
                    Ok(_) => extracted += 1,
                    Err(e) => errors.push(format!("Failed to store: {}", e)),
                }
            }
            ContextItemType::TaskResult => {
                // Task results noted but not stored as patterns
                // Don't increment extracted — only count actually-stored items
            }
            ContextItemType::CodeChange | ContextItemType::Unknown => {
                // Skip low-value items
            }
        }
    }

    FlushResult {
        patterns_extracted: extracted,
        items_scanned: context_items.len(),
        errors,
    }
}

/// POST /api/compaction/flush
///
/// Called by Claude Code hooks or external orchestrators before context
/// compaction. Extracts pending patterns from the provided context items
/// and stores them in the pattern store.
///
/// Body: `{ "context_items": [...] }`
pub async fn compaction_flush_handler(
    State(state): State<AppState>,
    axum::Json(req): axum::Json<FlushRequest>,
) -> impl IntoResponse {
    let config = CompactionFlushConfig::default();

    if let Some(ref storage) = state.storage {
        let result = flush_before_compaction(&req.context_items, storage, &config).await;
        axum::Json(result)
    } else {
        axum::Json(FlushResult {
            patterns_extracted: 0,
            items_scanned: 0,
            errors: vec!["Storage not available".into()],
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compaction_config_default() {
        let config = CompactionFlushConfig::default();
        assert!(config.enabled);
        assert_eq!(config.min_items_threshold, 5);
        assert_eq!(config.max_patterns_per_flush, 10);
    }

    #[test]
    fn test_flush_result_serialization() {
        let result = FlushResult {
            patterns_extracted: 3,
            items_scanned: 10,
            errors: vec!["some error".to_string()],
        };
        let json = serde_json::to_string(&result).unwrap();
        assert!(json.contains("\"patterns_extracted\":3"));
        assert!(json.contains("\"items_scanned\":10"));
        assert!(json.contains("some error"));

        let deserialized: FlushResult = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.patterns_extracted, 3);
        assert_eq!(deserialized.items_scanned, 10);
        assert_eq!(deserialized.errors.len(), 1);
    }

    #[test]
    fn test_context_item_type_serialization() {
        let types = vec![
            (ContextItemType::ErrorResolution, "\"error_resolution\""),
            (ContextItemType::Insight, "\"insight\""),
            (ContextItemType::CodeChange, "\"code_change\""),
            (ContextItemType::TaskResult, "\"task_result\""),
            (ContextItemType::Unknown, "\"unknown\""),
        ];

        for (item_type, expected_json) in types {
            let json = serde_json::to_string(&item_type).unwrap();
            assert_eq!(json, expected_json, "Failed for {:?}", item_type);

            let deserialized: ContextItemType = serde_json::from_str(&json).unwrap();
            assert_eq!(deserialized, item_type);
        }
    }

    #[test]
    fn test_flush_request_deserialization() {
        let json = r#"{
            "context_items": [
                {
                    "content": "Fixed timeout by adding retry logic",
                    "item_type": "error_resolution",
                    "timestamp": "2026-03-08T10:00:00Z"
                },
                {
                    "content": "Pattern matching is faster with match guards",
                    "item_type": "insight",
                    "timestamp": "2026-03-08T10:05:00Z"
                }
            ]
        }"#;

        let req: FlushRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.context_items.len(), 2);
        assert_eq!(req.context_items[0].item_type, ContextItemType::ErrorResolution);
        assert_eq!(req.context_items[1].item_type, ContextItemType::Insight);
        assert_eq!(req.context_items[0].content, "Fixed timeout by adding retry logic");
    }

    #[tokio::test]
    async fn test_flush_empty_context() {
        let config = CompactionFlushConfig::default();
        // Empty context should return 0 regardless -- below threshold
        let storage = create_test_storage().await;
        let result = flush_before_compaction(&[], &storage, &config).await;
        assert_eq!(result.patterns_extracted, 0);
        assert_eq!(result.items_scanned, 0);
        assert!(result.errors.is_empty());
    }

    #[tokio::test]
    async fn test_flush_below_threshold() {
        let config = CompactionFlushConfig {
            enabled: true,
            min_items_threshold: 5,
            max_patterns_per_flush: 10,
        };

        let items = vec![make_item("test insight", ContextItemType::Insight)];
        let storage = create_test_storage().await;
        let result = flush_before_compaction(&items, &storage, &config).await;
        assert_eq!(result.patterns_extracted, 0);
        assert_eq!(result.items_scanned, 1);
        assert!(result.errors.is_empty());
    }

    #[tokio::test]
    async fn test_flush_extracts_error_resolution() {
        let config = CompactionFlushConfig {
            enabled: true,
            min_items_threshold: 1, // low threshold for testing
            max_patterns_per_flush: 10,
        };

        let items = vec![
            make_item("Fixed timeout by adding retry logic", ContextItemType::ErrorResolution),
            make_item("Pattern matching is faster with match guards", ContextItemType::Insight),
            make_item("Changed src/main.rs", ContextItemType::CodeChange),
            make_item("Task completed successfully", ContextItemType::TaskResult),
            make_item("Unknown thing", ContextItemType::Unknown),
        ];

        let storage = create_test_storage().await;
        let result = flush_before_compaction(&items, &storage, &config).await;

        // Only ErrorResolution and Insight should be extracted
        assert_eq!(result.patterns_extracted, 2);
        assert_eq!(result.items_scanned, 5);
        assert!(result.errors.is_empty());
    }

    #[tokio::test]
    async fn test_flush_respects_max_limit() {
        let config = CompactionFlushConfig {
            enabled: true,
            min_items_threshold: 1,
            max_patterns_per_flush: 2, // limit to 2
        };

        let items = vec![
            make_item("insight 1", ContextItemType::Insight),
            make_item("insight 2", ContextItemType::Insight),
            make_item("insight 3", ContextItemType::Insight),
        ];

        let storage = create_test_storage().await;
        let result = flush_before_compaction(&items, &storage, &config).await;

        assert_eq!(result.patterns_extracted, 2); // capped at max
    }

    #[tokio::test]
    async fn test_flush_disabled() {
        let config = CompactionFlushConfig {
            enabled: false,
            min_items_threshold: 1,
            max_patterns_per_flush: 10,
        };

        let items = vec![make_item("test", ContextItemType::Insight)];
        let storage = create_test_storage().await;
        let result = flush_before_compaction(&items, &storage, &config).await;

        assert_eq!(result.patterns_extracted, 0);
    }

    // --- Helpers ---

    fn make_item(content: &str, item_type: ContextItemType) -> ContextItem {
        ContextItem {
            content: content.to_string(),
            item_type,
            timestamp: "2026-03-08T10:00:00Z".to_string(),
        }
    }

    async fn create_test_storage() -> Arc<TokioMutex<PatternStorage>> {
        use crate::db::DualWriteAdapter;
        use crate::reasoning_bank::storage::StorageConfig;

        let adapter = Arc::new(DualWriteAdapter::new_for_testing().unwrap());
        let storage = PatternStorage::new(adapter, StorageConfig::default())
            .await
            .unwrap();
        Arc::new(TokioMutex::new(storage))
    }
}
