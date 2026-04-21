//! Auto-promotion engine: promote patterns meeting recurrence thresholds.
//!
//! Rules (configurable via `AutoPromotionCriteria`):
//! - 3+ occurrences across 2+ distinct sessions within 30 days -> promote one tier
//! - Booster -> Crystal, Crystal -> Reflex
//! - Runs periodically (heartbeat) or on demand via CLI

use crate::reasoning_bank::storage::PatternStorage;
use crate::reasoning_bank::AutoPromotionCriteria;

use tracing::{debug, info};

/// Result of an auto-promotion scan.
#[derive(Debug, Clone, serde::Serialize)]
pub struct PromotionResult {
    /// Total number of patterns scanned (non-reflex tier).
    pub patterns_scanned: usize,
    /// Number of patterns that were promoted.
    pub patterns_promoted: usize,
    /// Details of each promotion.
    pub promotions: Vec<PromotionRecord>,
}

/// A single pattern promotion record.
#[derive(Debug, Clone, serde::Serialize)]
pub struct PromotionRecord {
    /// The pattern ID that was promoted.
    pub pattern_id: String,
    /// The tier before promotion.
    pub old_tier: String,
    /// The tier after promotion.
    pub new_tier: String,
    /// Total occurrences within the window.
    pub occurrences: u32,
    /// Distinct sessions/contexts within the window.
    pub distinct_contexts: u32,
}

/// Scan all non-Reflex patterns and promote those meeting the criteria.
///
/// Promotion path: Booster -> Crystal -> Reflex (one step at a time).
pub async fn run_auto_promotion(
    storage: &PatternStorage,
    criteria: &AutoPromotionCriteria,
) -> crate::error::Result<PromotionResult> {
    // 1. Query all Booster and Crystal tier patterns
    let patterns = storage
        .list_patterns_by_tier(
            &["booster", "crystal"],
            1000, // reasonable upper bound
        )
        .await?;

    info!(
        count = patterns.len(),
        "Auto-promotion: scanning non-reflex patterns"
    );

    let mut result = PromotionResult {
        patterns_scanned: patterns.len(),
        patterns_promoted: 0,
        promotions: Vec::new(),
    };

    // 2. For each pattern, count usage contexts within the window
    for pattern in &patterns {
        let id_str = pattern.id().to_string();

        let (uses, contexts) = storage
            .count_pattern_usage_contexts(&id_str, criteria.window_days)
            .await?;

        debug!(
            pattern_id = %id_str,
            uses = uses,
            contexts = contexts,
            "Checking auto-promotion eligibility"
        );

        // 3. Check if criteria are met
        if uses >= criteria.min_occurrences && contexts >= criteria.min_distinct_contexts {
            // Determine current tier from the pattern's stored tier column.
            // pattern_from_row reads the tier column but stores it in the
            // Pattern builder; we need to figure out what tier this pattern
            // has. We'll read it from the metadata or use the query filter.
            // Since we queried by tier IN ('booster', 'crystal'), we can
            // check via the raw row. But pattern.rs (CLI type) doesn't
            // expose tier directly. Let's query it.
            let tier_str = storage
                .adapter()
                .sqlite()
                .query_one(
                    "SELECT COALESCE(tier, 'booster') FROM reasoning_patterns WHERE id = ?",
                    &[&id_str],
                    |row| row.get::<_, String>(0),
                )
                .await?
                .unwrap_or_else(|| "booster".to_string());

            let old_tier = tier_str.as_str();
            let new_tier = match old_tier {
                "booster" => "crystal",
                "crystal" => "reflex",
                _ => continue, // Already reflex or unknown, skip
            };

            // 4. Update pattern tier
            storage.update_pattern_tier(&id_str, new_tier).await?;

            info!(
                pattern_id = %id_str,
                old_tier = old_tier,
                new_tier = new_tier,
                uses = uses,
                contexts = contexts,
                "Pattern auto-promoted"
            );

            result.promotions.push(PromotionRecord {
                pattern_id: id_str,
                old_tier: old_tier.to_string(),
                new_tier: new_tier.to_string(),
                occurrences: uses,
                distinct_contexts: contexts,
            });
            result.patterns_promoted += 1;
        }
    }

    info!(
        scanned = result.patterns_scanned,
        promoted = result.patterns_promoted,
        "Auto-promotion scan complete"
    );

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::DualWriteAdapter;
    use crate::reasoning_bank::pattern::{Pattern, PatternCategory};
    use crate::reasoning_bank::storage::StorageConfig;
    use std::sync::Arc;

    async fn setup_storage() -> PatternStorage {
        let adapter = Arc::new(DualWriteAdapter::new_for_testing().unwrap());
        PatternStorage::new(adapter, StorageConfig::default())
            .await
            .unwrap()
    }

    async fn store_pattern_with_tier(
        storage: &PatternStorage,
        problem: &str,
        tier: &str,
    ) -> String {
        let pattern = Pattern::builder()
            .problem(problem)
            .solution("test solution")
            .category(PatternCategory::default())
            .build();
        let id = storage.store_pattern(&pattern).await.unwrap();
        let id_str = id.to_string();
        storage.update_pattern_tier(&id_str, tier).await.unwrap();
        id_str
    }

    #[test]
    fn test_auto_promotion_criteria_default() {
        let c = AutoPromotionCriteria::default();
        assert_eq!(c.min_occurrences, 3);
        assert_eq!(c.min_distinct_contexts, 2);
        assert_eq!(c.window_days, 30);
    }

    #[test]
    fn test_promotion_result_serialization() {
        let result = PromotionResult {
            patterns_scanned: 10,
            patterns_promoted: 2,
            promotions: vec![PromotionRecord {
                pattern_id: "p-123".to_string(),
                old_tier: "booster".to_string(),
                new_tier: "crystal".to_string(),
                occurrences: 5,
                distinct_contexts: 3,
            }],
        };

        let json = serde_json::to_string(&result).unwrap();
        assert!(json.contains("patterns_scanned"));
        assert!(json.contains("patterns_promoted"));
        assert!(json.contains("p-123"));
    }

    #[test]
    fn test_promotion_record_serialization() {
        let record = PromotionRecord {
            pattern_id: "p-456".to_string(),
            old_tier: "crystal".to_string(),
            new_tier: "reflex".to_string(),
            occurrences: 7,
            distinct_contexts: 4,
        };

        let json = serde_json::to_string(&record).unwrap();
        assert!(json.contains("\"old_tier\":\"crystal\""));
        assert!(json.contains("\"new_tier\":\"reflex\""));
        assert!(json.contains("\"occurrences\":7"));
    }

    #[tokio::test]
    async fn test_auto_promotion_booster_to_crystal() {
        let storage = setup_storage().await;
        let criteria = AutoPromotionCriteria::default();

        let id = store_pattern_with_tier(&storage, "booster pattern", "booster").await;

        // Record 3 uses across 2 sessions
        storage
            .record_pattern_usage(&id, Some("s1"), None, "success")
            .await
            .unwrap();
        storage
            .record_pattern_usage(&id, Some("s2"), None, "success")
            .await
            .unwrap();
        storage
            .record_pattern_usage(&id, Some("s2"), None, "success")
            .await
            .unwrap();

        let result = run_auto_promotion(&storage, &criteria).await.unwrap();

        assert_eq!(result.patterns_promoted, 1);
        assert_eq!(result.promotions[0].old_tier, "booster");
        assert_eq!(result.promotions[0].new_tier, "crystal");
    }

    #[tokio::test]
    async fn test_auto_promotion_crystal_to_reflex() {
        let storage = setup_storage().await;
        let criteria = AutoPromotionCriteria::default();

        let id = store_pattern_with_tier(&storage, "crystal pattern", "crystal").await;

        // Record 4 uses across 3 sessions
        storage
            .record_pattern_usage(&id, Some("s1"), None, "success")
            .await
            .unwrap();
        storage
            .record_pattern_usage(&id, Some("s2"), None, "success")
            .await
            .unwrap();
        storage
            .record_pattern_usage(&id, Some("s3"), None, "success")
            .await
            .unwrap();
        storage
            .record_pattern_usage(&id, Some("s3"), None, "success")
            .await
            .unwrap();

        let result = run_auto_promotion(&storage, &criteria).await.unwrap();

        assert_eq!(result.patterns_promoted, 1);
        assert_eq!(result.promotions[0].old_tier, "crystal");
        assert_eq!(result.promotions[0].new_tier, "reflex");
    }

    #[tokio::test]
    async fn test_auto_promotion_below_threshold() {
        let storage = setup_storage().await;
        let criteria = AutoPromotionCriteria::default();

        let id = store_pattern_with_tier(&storage, "low usage pattern", "booster").await;

        // Only 2 uses (below min_occurrences of 3)
        storage
            .record_pattern_usage(&id, Some("s1"), None, "success")
            .await
            .unwrap();
        storage
            .record_pattern_usage(&id, Some("s2"), None, "success")
            .await
            .unwrap();

        let result = run_auto_promotion(&storage, &criteria).await.unwrap();

        assert_eq!(result.patterns_scanned, 1);
        assert_eq!(result.patterns_promoted, 0);
    }

    #[tokio::test]
    async fn test_auto_promotion_single_context() {
        let storage = setup_storage().await;
        let criteria = AutoPromotionCriteria::default();

        let id = store_pattern_with_tier(&storage, "single session pattern", "booster").await;

        // 5 uses but all in 1 session (below min_distinct_contexts of 2)
        for _ in 0..5 {
            storage
                .record_pattern_usage(&id, Some("same-session"), None, "success")
                .await
                .unwrap();
        }

        let result = run_auto_promotion(&storage, &criteria).await.unwrap();

        assert_eq!(result.patterns_scanned, 1);
        assert_eq!(result.patterns_promoted, 0);
    }

    #[tokio::test]
    async fn test_auto_promotion_reflex_not_scanned() {
        let storage = setup_storage().await;
        let criteria = AutoPromotionCriteria::default();

        // A reflex pattern should not appear in the scan
        let _id = store_pattern_with_tier(&storage, "reflex pattern", "reflex").await;

        let result = run_auto_promotion(&storage, &criteria).await.unwrap();

        // Reflex patterns are not included in the scan
        assert_eq!(result.patterns_scanned, 0);
        assert_eq!(result.patterns_promoted, 0);
    }

    #[tokio::test]
    async fn test_auto_promotion_no_patterns() {
        let storage = setup_storage().await;
        let criteria = AutoPromotionCriteria::default();

        let result = run_auto_promotion(&storage, &criteria).await.unwrap();

        assert_eq!(result.patterns_scanned, 0);
        assert_eq!(result.patterns_promoted, 0);
        assert!(result.promotions.is_empty());
    }
}
