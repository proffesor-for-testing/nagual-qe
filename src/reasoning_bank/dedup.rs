//! Deduplication module for pattern storage.
//!
//! Provides functionality to find and merge duplicate patterns using:
//! - Exact duplicates: BLAKE3 content hash matching
//! - Near-duplicates: Cosine similarity of embeddings above threshold
//!
//! # Usage
//!
//! ```bash
//! # Scan for duplicates (read-only)
//! nagual learn dedup --scan
//!
//! # Auto-merge exact duplicates
//! nagual learn dedup --auto
//!
//! # Generate detailed report
//! nagual learn dedup --report
//!
//! # Find near-duplicates with custom threshold
//! nagual learn dedup --scan --threshold 0.92
//! ```

use std::collections::HashMap;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};

use super::pattern::{Pattern, PatternId};
use super::storage::PatternStorage;
use crate::error::Result;
use crate::ml::{cosine_similarity, to_array1};

/// Result of a deduplication scan or merge operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DedupResult {
    /// Groups of exact duplicates (same content_hash).
    pub exact_duplicates: Vec<DuplicateGroup>,

    /// Groups of near-duplicates (above similarity threshold).
    pub near_duplicates: Vec<DuplicateGroup>,

    /// Total number of patterns scanned.
    pub total_patterns: usize,

    /// Number of duplicate patterns found.
    pub duplicate_count: usize,

    /// Number of patterns merged (only for auto mode).
    pub merged_count: usize,

    /// Estimated space savings in bytes (problem + solution lengths).
    pub space_savings_bytes: usize,

    /// Duration of the operation in milliseconds.
    pub duration_ms: u64,

    /// Errors encountered during the operation.
    pub errors: Vec<String>,

    /// Whether this was a dry-run (scan-only).
    pub dry_run: bool,
}

impl Default for DedupResult {
    fn default() -> Self {
        Self {
            exact_duplicates: Vec::new(),
            near_duplicates: Vec::new(),
            total_patterns: 0,
            duplicate_count: 0,
            merged_count: 0,
            space_savings_bytes: 0,
            duration_ms: 0,
            errors: Vec::new(),
            dry_run: true,
        }
    }
}

/// A group of duplicate patterns.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DuplicateGroup {
    /// The ID of the canonical (kept) pattern (highest reward).
    pub canonical_id: String,

    /// IDs of duplicate patterns to merge/delete.
    pub duplicate_ids: Vec<String>,

    /// Similarity score (1.0 for exact, <1.0 for near-duplicates).
    pub similarity: f32,

    /// Content hash (for exact duplicates).
    pub content_hash: Option<String>,

    /// Combined reward of all patterns in the group.
    pub combined_reward: f32,

    /// Total reuse count across all patterns.
    pub total_reuse_count: u32,
}

/// Configuration for deduplication operations.
#[derive(Debug, Clone)]
pub struct DedupConfig {
    /// Similarity threshold for near-duplicates (0.0-1.0).
    pub similarity_threshold: f32,

    /// Maximum number of patterns to process at once.
    pub batch_size: usize,

    /// Whether to scan only (no modifications).
    pub scan_only: bool,

    /// Whether to generate a detailed report.
    pub generate_report: bool,

    /// Minimum patterns in a group to consider it for merging.
    pub min_group_size: usize,
}

impl Default for DedupConfig {
    fn default() -> Self {
        Self {
            similarity_threshold: 0.95,
            batch_size: 1000,
            scan_only: true,
            generate_report: false,
            min_group_size: 2,
        }
    }
}

impl DedupConfig {
    /// Create a new config with the given similarity threshold.
    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.similarity_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Enable or disable scan-only mode.
    pub fn with_scan_only(mut self, scan_only: bool) -> Self {
        self.scan_only = scan_only;
        self
    }

    /// Enable or disable report generation.
    pub fn with_report(mut self, report: bool) -> Self {
        self.generate_report = report;
        self
    }
}

/// Find exact duplicates by content_hash.
///
/// This queries the database directly to find patterns with the same
/// BLAKE3 content hash.
pub async fn find_exact_duplicates(storage: &Arc<PatternStorage>) -> Result<Vec<DuplicateGroup>> {
    let sql = r#"
        SELECT content_hash, GROUP_CONCAT(id) as ids, COUNT(*) as cnt
        FROM reasoning_patterns
        WHERE content_hash IS NOT NULL AND content_hash != ''
        GROUP BY content_hash
        HAVING cnt > 1
        ORDER BY cnt DESC
    "#;

    let rows: Vec<(String, String, i64)> = storage
        .adapter()
        .sqlite()
        .query(sql, &[], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, i64>(2)?,
            ))
        })
        .await?;

    let mut groups = Vec::new();

    for (content_hash, ids_str, _count) in rows {
        let ids: Vec<&str> = ids_str.split(',').collect();
        if ids.len() < 2 {
            continue;
        }

        // Fetch patterns to determine canonical (highest reward)
        let patterns = fetch_patterns_by_ids(storage, &ids).await?;
        if patterns.is_empty() {
            continue;
        }

        // Sort by reward descending, then by reuse_count descending
        let mut sorted_patterns = patterns;
        sorted_patterns.sort_by(|a, b| {
            b.reward()
                .partial_cmp(&a.reward())
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(b.reuse_count().cmp(&a.reuse_count()))
        });

        let canonical = &sorted_patterns[0];
        let duplicates: Vec<String> = sorted_patterns[1..]
            .iter()
            .map(|p| p.id().to_string())
            .collect();

        let combined_reward: f32 = sorted_patterns.iter().map(|p| p.reward()).sum();
        let total_reuse: u32 = sorted_patterns.iter().map(|p| p.reuse_count()).sum();

        groups.push(DuplicateGroup {
            canonical_id: canonical.id().to_string(),
            duplicate_ids: duplicates,
            similarity: 1.0, // Exact match
            content_hash: Some(content_hash),
            combined_reward,
            total_reuse_count: total_reuse,
        });
    }

    Ok(groups)
}

/// Find near-duplicates using embedding cosine similarity.
///
/// This compares patterns with embeddings to find those above the
/// similarity threshold.
pub async fn find_near_duplicates(
    storage: &Arc<PatternStorage>,
    threshold: f32,
) -> Result<Vec<DuplicateGroup>> {
    // Get all patterns with embeddings
    let patterns = storage.get_all_with_embeddings().await?;

    if patterns.len() < 2 {
        return Ok(Vec::new());
    }

    info!(
        "Scanning {} patterns for near-duplicates (threshold: {:.2})",
        patterns.len(),
        threshold
    );

    // Build list of patterns with their embeddings as arrays
    let mut pattern_embeddings: Vec<(&Pattern, ndarray::Array1<f32>)> = Vec::new();
    for pattern in &patterns {
        if let Some(emb) = pattern.embedding() {
            pattern_embeddings.push((pattern, to_array1(emb)));
        }
    }

    // Track which patterns are already in a group
    let mut in_group: HashMap<String, usize> = HashMap::new();
    let mut groups: Vec<DuplicateGroup> = Vec::new();

    // Compare each pair (O(n^2) but necessary for accurate similarity)
    for i in 0..pattern_embeddings.len() {
        let (p_i, emb_i) = &pattern_embeddings[i];
        let id_i = p_i.id().to_string();

        // Skip if already in a group
        if in_group.contains_key(&id_i) {
            continue;
        }

        // Find all patterns similar to this one
        let mut similar: Vec<(&Pattern, f32)> = Vec::new();

        for j in (i + 1)..pattern_embeddings.len() {
            let (p_j, emb_j) = &pattern_embeddings[j];
            let id_j = p_j.id().to_string();

            // Skip if already in a different group
            if in_group.contains_key(&id_j) {
                continue;
            }

            let sim = cosine_similarity(&emb_i.view(), &emb_j.view());
            if sim >= threshold {
                similar.push((p_j, sim));
            }
        }

        if !similar.is_empty() {
            // Create a group with this pattern as initial candidate
            let mut group_patterns: Vec<(&Pattern, f32)> =
                vec![(p_i, 1.0)]; // Self is similarity 1.0
            group_patterns.extend(similar);

            // Sort by reward descending to find canonical
            group_patterns.sort_by(|a, b| {
                b.0.reward()
                    .partial_cmp(&a.0.reward())
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(b.0.reuse_count().cmp(&a.0.reuse_count()))
            });

            let canonical = group_patterns[0].0;
            let duplicates: Vec<String> = group_patterns[1..]
                .iter()
                .map(|(p, _)| p.id().to_string())
                .collect();

            // Calculate average similarity (excluding self)
            let avg_similarity: f32 = if duplicates.is_empty() {
                1.0
            } else {
                group_patterns[1..]
                    .iter()
                    .map(|(_, sim)| sim)
                    .sum::<f32>()
                    / duplicates.len() as f32
            };

            let combined_reward: f32 = group_patterns.iter().map(|(p, _)| p.reward()).sum();
            let total_reuse: u32 = group_patterns.iter().map(|(p, _)| p.reuse_count()).sum();

            // Mark all as in this group
            let group_idx = groups.len();
            for (p, _) in &group_patterns {
                in_group.insert(p.id().to_string(), group_idx);
            }

            groups.push(DuplicateGroup {
                canonical_id: canonical.id().to_string(),
                duplicate_ids: duplicates,
                similarity: avg_similarity,
                content_hash: None,
                combined_reward,
                total_reuse_count: total_reuse,
            });
        }
    }

    Ok(groups)
}

/// Merge a duplicate group by keeping the canonical and deleting duplicates.
///
/// The canonical pattern is updated with the combined reuse_count.
pub async fn merge_duplicate_group(
    storage: &Arc<PatternStorage>,
    group: &DuplicateGroup,
) -> Result<usize> {
    let mut merged_count = 0;

    // First, get the canonical pattern and update its reuse_count
    let canonical_id = PatternId::from_string(&group.canonical_id);

    if let Some(mut canonical) = storage.get_pattern(&canonical_id).await? {
        // Aggregate reuse count from all duplicates
        for dup_id in &group.duplicate_ids {
            let dup_pattern_id = PatternId::from_string(dup_id);
            if let Some(dup_pattern) = storage.get_pattern(&dup_pattern_id).await? {
                // Add duplicate's reuse count to canonical
                for _ in 0..dup_pattern.reuse_count() {
                    canonical.increment_reuse_count();
                }
            }
        }

        // Update the canonical pattern
        storage.update_pattern(&canonical).await?;

        // Delete the duplicates
        for dup_id in &group.duplicate_ids {
            let dup_pattern_id = PatternId::from_string(dup_id);
            match storage.delete_pattern(&dup_pattern_id).await {
                Ok(()) => {
                    merged_count += 1;
                    debug!(
                        canonical = %group.canonical_id,
                        duplicate = %dup_id,
                        "Merged duplicate pattern"
                    );
                }
                Err(e) => {
                    warn!(
                        duplicate = %dup_id,
                        error = %e,
                        "Failed to delete duplicate pattern"
                    );
                }
            }
        }
    }

    Ok(merged_count)
}

/// Scan for duplicates without making changes.
pub async fn scan_duplicates(
    storage: &Arc<PatternStorage>,
    config: &DedupConfig,
) -> Result<DedupResult> {
    let start = std::time::Instant::now();

    let total_patterns = storage.count().await?;
    let mut result = DedupResult {
        total_patterns,
        dry_run: true,
        ..Default::default()
    };

    // Find exact duplicates
    result.exact_duplicates = find_exact_duplicates(storage).await?;

    // Find near-duplicates
    result.near_duplicates =
        find_near_duplicates(storage, config.similarity_threshold).await?;

    // Calculate statistics
    result.duplicate_count = result
        .exact_duplicates
        .iter()
        .map(|g| g.duplicate_ids.len())
        .sum::<usize>()
        + result
            .near_duplicates
            .iter()
            .map(|g| g.duplicate_ids.len())
            .sum::<usize>();

    // Estimate space savings (very rough: just count characters)
    result.space_savings_bytes = estimate_space_savings(storage, &result).await?;

    result.duration_ms = start.elapsed().as_millis() as u64;

    Ok(result)
}

/// Auto-merge exact duplicates.
pub async fn auto_merge(
    storage: &Arc<PatternStorage>,
    config: &DedupConfig,
) -> Result<DedupResult> {
    let start = std::time::Instant::now();

    let total_patterns = storage.count().await?;
    let mut result = DedupResult {
        total_patterns,
        dry_run: false,
        ..Default::default()
    };

    // Find exact duplicates
    result.exact_duplicates = find_exact_duplicates(storage).await?;

    // Also find near-duplicates for reporting (but don't auto-merge them)
    result.near_duplicates =
        find_near_duplicates(storage, config.similarity_threshold).await?;

    // Merge exact duplicates
    for group in &result.exact_duplicates {
        if group.duplicate_ids.len() >= config.min_group_size - 1 {
            match merge_duplicate_group(storage, group).await {
                Ok(count) => {
                    result.merged_count += count;
                }
                Err(e) => {
                    result.errors.push(format!(
                        "Failed to merge group {}: {}",
                        group.canonical_id, e
                    ));
                }
            }
        }
    }

    result.duplicate_count = result
        .exact_duplicates
        .iter()
        .map(|g| g.duplicate_ids.len())
        .sum::<usize>()
        + result
            .near_duplicates
            .iter()
            .map(|g| g.duplicate_ids.len())
            .sum::<usize>();

    result.space_savings_bytes = estimate_space_savings(storage, &result).await?;
    result.duration_ms = start.elapsed().as_millis() as u64;

    Ok(result)
}

/// Estimate space savings from removing duplicates.
async fn estimate_space_savings(
    storage: &Arc<PatternStorage>,
    result: &DedupResult,
) -> Result<usize> {
    let mut total_bytes = 0;

    // Get all duplicate IDs
    let mut duplicate_ids: Vec<&str> = Vec::new();
    for group in &result.exact_duplicates {
        duplicate_ids.extend(group.duplicate_ids.iter().map(|s| s.as_str()));
    }
    for group in &result.near_duplicates {
        duplicate_ids.extend(group.duplicate_ids.iter().map(|s| s.as_str()));
    }

    // Estimate size of each duplicate
    for id in duplicate_ids {
        let pattern_id = PatternId::from_string(id);
        if let Some(pattern) = storage.get_pattern(&pattern_id).await? {
            // Rough estimate: problem + solution + context
            total_bytes += pattern.problem().len();
            total_bytes += pattern.solution().len();
            total_bytes += pattern.context().len();
        }
    }

    Ok(total_bytes)
}

/// Fetch patterns by a list of IDs.
async fn fetch_patterns_by_ids(
    storage: &Arc<PatternStorage>,
    ids: &[&str],
) -> Result<Vec<Pattern>> {
    let mut patterns = Vec::new();
    for id in ids {
        let pattern_id = PatternId::from_string(*id);
        if let Some(pattern) = storage.get_pattern(&pattern_id).await? {
            patterns.push(pattern);
        }
    }
    Ok(patterns)
}

/// Print a human-readable deduplication report.
pub fn print_report(result: &DedupResult) {
    println!("\nDeduplication Report");
    println!("{:-<60}", "");
    println!("  Total patterns scanned: {}", result.total_patterns);
    println!("  Exact duplicates found: {}", result.exact_duplicates.len());
    println!("  Near-duplicates found: {}", result.near_duplicates.len());
    println!(
        "  Total duplicate patterns: {}",
        result.duplicate_count
    );
    println!(
        "  Estimated space savings: {} bytes ({:.1} KB)",
        result.space_savings_bytes,
        result.space_savings_bytes as f64 / 1024.0
    );
    println!("  Duration: {}ms", result.duration_ms);

    if result.dry_run {
        println!("\n  DRY RUN: No changes were made.");
    } else {
        println!("\n  Patterns merged: {}", result.merged_count);
    }

    if !result.exact_duplicates.is_empty() {
        println!("\nExact Duplicate Groups ({}):", result.exact_duplicates.len());
        for (i, group) in result.exact_duplicates.iter().take(10).enumerate() {
            println!(
                "  {}. Canonical: {} | Duplicates: {} | Total reuse: {}",
                i + 1,
                &group.canonical_id[..8.min(group.canonical_id.len())],
                group.duplicate_ids.len(),
                group.total_reuse_count
            );
        }
        if result.exact_duplicates.len() > 10 {
            println!(
                "  ... and {} more groups",
                result.exact_duplicates.len() - 10
            );
        }
    }

    if !result.near_duplicates.is_empty() {
        println!("\nNear-Duplicate Groups ({}):", result.near_duplicates.len());
        for (i, group) in result.near_duplicates.iter().take(10).enumerate() {
            println!(
                "  {}. Canonical: {} | Duplicates: {} | Similarity: {:.2} | Total reuse: {}",
                i + 1,
                &group.canonical_id[..8.min(group.canonical_id.len())],
                group.duplicate_ids.len(),
                group.similarity,
                group.total_reuse_count
            );
        }
        if result.near_duplicates.len() > 10 {
            println!(
                "  ... and {} more groups",
                result.near_duplicates.len() - 10
            );
        }
    }

    if !result.errors.is_empty() {
        println!("\nErrors ({}):", result.errors.len());
        for err in &result.errors {
            println!("  - {}", err);
        }
    }
}

/// Result of backfilling content hashes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackfillResult {
    /// Total patterns checked.
    pub total_patterns: usize,
    /// Patterns that already had hashes.
    pub already_hashed: usize,
    /// Patterns that were updated with new hashes.
    pub updated: usize,
    /// Errors encountered.
    pub errors: Vec<String>,
    /// Duration in milliseconds.
    pub duration_ms: u64,
}

/// Backfill content_hash for all patterns that don't have one.
///
/// Computes BLAKE3 hash of problem + solution for each pattern
/// and updates the database.
pub async fn backfill_content_hashes(
    storage: &Arc<PatternStorage>,
    dry_run: bool,
) -> Result<BackfillResult> {
    use std::time::Instant;

    let start = Instant::now();

    // Get all patterns without content_hash
    let sql = "SELECT id, problem, solution FROM reasoning_patterns WHERE content_hash IS NULL OR content_hash = ''";
    let rows: Vec<(String, String, String)> = storage
        .adapter()
        .sqlite()
        .query(sql, &[], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
            ))
        })
        .await?;

    // Count patterns with existing hashes
    let count_sql = "SELECT COUNT(*) FROM reasoning_patterns WHERE content_hash IS NOT NULL AND content_hash != ''";
    let count_result: Vec<i64> = storage
        .adapter()
        .sqlite()
        .query(count_sql, &[], |row| Ok(row.get::<_, i64>(0)?))
        .await?;
    let already_hashed = count_result.first().copied().unwrap_or(0);

    let total_patterns = rows.len() + already_hashed as usize;
    let mut updated = 0;
    let mut errors = Vec::new();

    if !dry_run {
        for (id, problem, solution) in &rows {
            let content = format!("{}\n{}", problem, solution);
            let hash = blake3::hash(content.as_bytes()).to_hex().to_string();

            let update_sql = "UPDATE reasoning_patterns SET content_hash = ? WHERE id = ?";
            match storage
                .adapter()
                .sqlite()
                .execute(update_sql, &[&hash as &dyn rusqlite::ToSql, id])
                .await
            {
                Ok(_) => updated += 1,
                Err(e) => errors.push(format!("Failed to update {}: {}", id, e)),
            }
        }
    } else {
        updated = rows.len(); // Would update this many
    }

    info!(
        "Backfill complete: {} patterns need hashes, {} already hashed",
        rows.len(),
        already_hashed
    );

    Ok(BackfillResult {
        total_patterns,
        already_hashed: already_hashed as usize,
        updated,
        errors,
        duration_ms: start.elapsed().as_millis() as u64,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::DualWriteAdapter;
    use crate::reasoning_bank::storage::StorageConfig;

    async fn create_test_storage() -> Arc<PatternStorage> {
        let adapter = Arc::new(DualWriteAdapter::new_for_testing().unwrap());
        Arc::new(
            PatternStorage::new(adapter, StorageConfig::default())
                .await
                .unwrap(),
        )
    }

    #[tokio::test]
    async fn test_dedup_config_default() {
        let config = DedupConfig::default();
        assert_eq!(config.similarity_threshold, 0.95);
        assert!(config.scan_only);
        assert!(!config.generate_report);
    }

    #[tokio::test]
    async fn test_dedup_config_builder() {
        let config = DedupConfig::default()
            .with_threshold(0.92)
            .with_scan_only(false)
            .with_report(true);

        assert!((config.similarity_threshold - 0.92).abs() < 0.001);
        assert!(!config.scan_only);
        assert!(config.generate_report);
    }

    #[tokio::test]
    async fn test_scan_empty_db() {
        let storage = create_test_storage().await;
        let config = DedupConfig::default();

        let result = scan_duplicates(&storage, &config).await.unwrap();

        assert_eq!(result.total_patterns, 0);
        assert!(result.exact_duplicates.is_empty());
        assert!(result.near_duplicates.is_empty());
        assert!(result.dry_run);
    }

    #[tokio::test]
    async fn test_find_exact_duplicates() {
        let storage = create_test_storage().await;

        // Create patterns with same content (will have same hash)
        let mut p1 = Pattern::new("Same problem", "Same solution");
        p1.set_reward(0.9);
        p1.compute_content_hash();

        let mut p2 = Pattern::new("Same problem", "Same solution");
        p2.set_reward(0.5);
        p2.compute_content_hash();

        let mut p3 = Pattern::new("Different problem", "Different solution");
        p3.compute_content_hash();

        storage.store_pattern(&p1).await.unwrap();
        storage.store_pattern(&p2).await.unwrap();
        storage.store_pattern(&p3).await.unwrap();

        let duplicates = find_exact_duplicates(&storage).await.unwrap();

        assert_eq!(duplicates.len(), 1);
        assert_eq!(duplicates[0].duplicate_ids.len(), 1);
        assert_eq!(duplicates[0].similarity, 1.0);
        // The canonical should be p1 (higher reward)
        assert_eq!(duplicates[0].canonical_id, p1.id().to_string());
    }

    #[tokio::test]
    async fn test_merge_duplicate_group() {
        let storage = create_test_storage().await;

        // Create patterns with same content
        let mut p1 = Pattern::new("Same problem", "Same solution");
        p1.set_reward(0.9);
        p1.compute_content_hash();

        let mut p2 = Pattern::new("Same problem", "Same solution");
        p2.set_reward(0.5);
        p2.increment_reuse_count();
        p2.increment_reuse_count();
        p2.compute_content_hash();

        // Store the IDs before storing (they won't change, but for clarity)
        let p1_id = p1.id().to_string();
        let p2_id = p2.id().to_string();

        storage.store_pattern(&p1).await.unwrap();
        storage.store_pattern(&p2).await.unwrap();

        // Verify patterns are stored
        let stored_p1 = storage.get_pattern(&PatternId::from_string(&p1_id)).await.unwrap();
        assert!(stored_p1.is_some(), "p1 should be stored");
        let stored_p2 = storage.get_pattern(&PatternId::from_string(&p2_id)).await.unwrap();
        assert!(stored_p2.is_some(), "p2 should be stored");

        let group = DuplicateGroup {
            canonical_id: p1_id.clone(),
            duplicate_ids: vec![p2_id.clone()],
            similarity: 1.0,
            content_hash: p1.content_hash().map(|s| s.to_string()),
            combined_reward: p1.reward() + p2.reward(),
            total_reuse_count: p1.reuse_count() + p2.reuse_count(),
        };

        let merged = merge_duplicate_group(&storage, &group).await.unwrap();
        assert_eq!(merged, 1);

        // Verify p2 is deleted
        let p2_check = storage.get_pattern(&PatternId::from_string(&p2_id)).await.unwrap();
        assert!(p2_check.is_none());

        // Verify p1 has updated reuse count
        let p1_check = storage.get_pattern(&PatternId::from_string(&p1_id)).await.unwrap().unwrap();
        assert_eq!(p1_check.reuse_count(), 2); // p2 had 2 reuse counts
    }

    #[tokio::test]
    async fn test_auto_merge() {
        let storage = create_test_storage().await;

        // Create patterns with same content
        let mut p1 = Pattern::new("Same problem", "Same solution");
        p1.set_reward(0.9);
        p1.compute_content_hash();

        let mut p2 = Pattern::new("Same problem", "Same solution");
        p2.set_reward(0.5);
        p2.compute_content_hash();

        storage.store_pattern(&p1).await.unwrap();
        storage.store_pattern(&p2).await.unwrap();

        let config = DedupConfig::default().with_scan_only(false);
        let result = auto_merge(&storage, &config).await.unwrap();

        assert_eq!(result.merged_count, 1);
        assert!(!result.dry_run);
        assert_eq!(storage.count().await.unwrap(), 1);
    }
}
