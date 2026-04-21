//! Pattern consolidation and maintenance algorithms.
//!
//! This module provides algorithms for:
//! - Consolidating similar patterns to reduce redundancy
//! - Detecting low-reward patterns for review or removal
//! - Finding stale patterns that may need refreshing
//! - Managing pattern lifecycle (review, archive)

use std::collections::HashSet;

use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use tracing::{debug, info, instrument};

use crate::error::{NagualError, Result};
use crate::reasoning_bank::pattern::{Pattern, PatternId};
use crate::reasoning_bank::storage::PatternStorage;

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for pattern consolidation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsolidationConfig {
    /// Minimum similarity threshold for considering patterns as duplicates (0.0 - 1.0).
    /// Patterns with similarity >= this threshold will be merged.
    #[serde(default = "default_similarity_threshold")]
    pub similarity_threshold: f32,

    /// Minimum number of patterns needed to consider consolidation.
    #[serde(default = "default_min_patterns")]
    pub min_patterns_for_consolidation: usize,

    /// Maximum patterns to process in one consolidation run.
    #[serde(default = "default_max_patterns")]
    pub max_patterns_to_process: usize,

    /// Whether to actually delete/archive consolidated patterns.
    /// If false, only marks them and returns what would be done.
    #[serde(default)]
    pub dry_run: bool,

    /// Keep original patterns as archived instead of deleting.
    #[serde(default = "default_keep_archived")]
    pub keep_archived: bool,

    /// Strategy for merging rewards when consolidating.
    #[serde(default)]
    pub reward_merge_strategy: RewardMergeStrategy,

    /// Maximum tags per merged pattern. Matches StorageConfig::max_tags.
    #[serde(default = "default_max_tags")]
    pub max_tags: usize,
}

fn default_similarity_threshold() -> f32 {
    0.85
}

fn default_min_patterns() -> usize {
    2
}

fn default_max_patterns() -> usize {
    1000
}

fn default_keep_archived() -> bool {
    true
}

fn default_max_tags() -> usize {
    20
}

impl Default for ConsolidationConfig {
    fn default() -> Self {
        Self {
            similarity_threshold: default_similarity_threshold(),
            min_patterns_for_consolidation: default_min_patterns(),
            max_patterns_to_process: default_max_patterns(),
            dry_run: false,
            keep_archived: default_keep_archived(),
            reward_merge_strategy: RewardMergeStrategy::default(),
            max_tags: default_max_tags(),
        }
    }
}

/// Strategy for merging rewards when consolidating patterns.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RewardMergeStrategy {
    /// Take the average of all rewards
    #[default]
    Average,
    /// Take the maximum reward
    Maximum,
    /// Take the minimum reward (conservative)
    Minimum,
    /// Weighted average based on reuse count
    WeightedByUsage,
}

// ============================================================================
// Consolidation Result Types
// ============================================================================

/// Result of a consolidation operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsolidationResult {
    /// Groups of patterns that were/would be consolidated
    pub groups: Vec<ConsolidatedGroup>,

    /// Total patterns processed
    pub patterns_processed: usize,

    /// Patterns that were consolidated
    pub patterns_consolidated: usize,

    /// Number of unique groups formed
    pub groups_formed: usize,

    /// Patterns that couldn't be processed
    pub errors: Vec<String>,

    /// Whether this was a dry run
    pub dry_run: bool,

    /// Time taken for consolidation
    pub duration_ms: u64,
}

impl ConsolidationResult {
    /// Create an empty result.
    pub fn empty(dry_run: bool) -> Self {
        Self {
            groups: Vec::new(),
            patterns_processed: 0,
            patterns_consolidated: 0,
            groups_formed: 0,
            errors: Vec::new(),
            dry_run,
            duration_ms: 0,
        }
    }
}

/// A group of patterns that were consolidated.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsolidatedGroup {
    /// The merged/primary pattern ID
    pub primary_id: PatternId,

    /// IDs of patterns that were merged into the primary
    pub merged_ids: Vec<PatternId>,

    /// Average similarity within the group
    pub average_similarity: f32,

    /// Combined tags from all patterns
    pub combined_tags: Vec<String>,

    /// Combined related patterns
    pub combined_related: Vec<PatternId>,

    /// Merged reward value
    pub merged_reward: f32,

    /// Merged effectiveness value
    pub merged_effectiveness: f32,

    /// Total reuse count across all patterns
    pub total_reuse_count: u32,
}

// ============================================================================
// Low-Reward Pattern Detection
// ============================================================================

/// Report for patterns with low reward scores.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LowRewardPatternReport {
    /// The pattern ID
    pub pattern_id: PatternId,

    /// Current reward score
    pub reward: f32,

    /// Current effectiveness
    pub effectiveness: f32,

    /// Number of times the pattern was used
    pub usage_count: u32,

    /// Success rate (if applicable)
    pub success_rate: f32,

    /// When the pattern was last updated
    pub last_updated: DateTime<Utc>,

    /// Problem description (for context)
    pub problem_summary: String,

    /// Recommendation for handling this pattern
    pub recommendation: LowRewardRecommendation,

    /// Potential reasons for low reward
    pub possible_reasons: Vec<String>,
}

/// Recommendation for handling a low-reward pattern.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LowRewardRecommendation {
    /// Pattern should be reviewed by a human
    Review,
    /// Pattern should be retried with modifications
    Retry,
    /// Pattern should be archived (kept but not used)
    Archive,
    /// Pattern should be deleted
    Delete,
    /// Pattern needs more data before deciding
    NeedsMoreData,
}

impl LowRewardRecommendation {
    /// Get a description of the recommendation.
    pub fn description(&self) -> &'static str {
        match self {
            Self::Review => "Manual review recommended to understand why pattern is underperforming",
            Self::Retry => "Pattern may work in different contexts; consider retry with modifications",
            Self::Archive => "Pattern should be archived - kept for reference but not actively used",
            Self::Delete => "Pattern provides no value and should be deleted",
            Self::NeedsMoreData => "Insufficient usage data; continue monitoring before deciding",
        }
    }
}

/// Find patterns with low reward scores.
///
/// # Arguments
///
/// * `storage` - Pattern storage reference
/// * `min_reward_threshold` - Patterns below this reward are flagged (default: 0.4)
/// * `min_usage_count` - Minimum usages before flagging (default: 3)
///
/// # Returns
///
/// List of reports for low-reward patterns.
#[instrument(skip(storage))]
pub async fn find_low_reward_patterns(
    storage: &PatternStorage,
    min_reward_threshold: f32,
    min_usage_count: u32,
) -> Result<Vec<LowRewardPatternReport>> {
    let sql = r#"
        SELECT * FROM reasoning_patterns
        WHERE reward < ?
        AND reuse_count >= ?
        ORDER BY reward ASC
    "#;

    let threshold = min_reward_threshold as f64;
    let usage = min_usage_count as i64;

    let patterns = storage
        .adapter()
        .sqlite()
        .query(sql, &[&threshold, &usage], |row| {
            // Reconstruct pattern from row
            let id: String = row.get("id")?;
            let reward: f64 = row.get("reward")?;
            let effectiveness: f64 = row.get("effectiveness")?;
            let reuse_count: i32 = row.get("reuse_count")?;
            let success: bool = row.get::<_, i32>("success")? != 0;
            let updated_at_str: String = row.get("updated_at")?;
            let problem: String = row.get("problem")?;

            let updated_at = DateTime::parse_from_rfc3339(&updated_at_str)
                .map(|dt| dt.with_timezone(&Utc))
                .unwrap_or_else(|_| Utc::now());

            Ok((
                id,
                reward as f32,
                effectiveness as f32,
                reuse_count as u32,
                success,
                updated_at,
                problem,
            ))
        })
        .await?;

    let mut reports = Vec::new();

    for (id, reward, effectiveness, reuse_count, success, updated_at, problem) in patterns {
        let success_rate = if success { 1.0 } else { 0.0 };

        // Determine recommendation based on pattern characteristics
        let recommendation = determine_low_reward_recommendation(
            reward,
            effectiveness,
            reuse_count,
            success_rate,
        );

        // Analyze possible reasons
        let possible_reasons = analyze_low_reward_reasons(
            reward,
            effectiveness,
            reuse_count,
            success_rate,
        );

        reports.push(LowRewardPatternReport {
            pattern_id: PatternId::from_string(id),
            reward,
            effectiveness,
            usage_count: reuse_count,
            success_rate,
            last_updated: updated_at,
            problem_summary: truncate_string(&problem, 100),
            recommendation,
            possible_reasons,
        });
    }

    info!(
        count = reports.len(),
        threshold = min_reward_threshold,
        "Found low-reward patterns"
    );

    Ok(reports)
}

/// Determine the appropriate recommendation for a low-reward pattern.
fn determine_low_reward_recommendation(
    reward: f32,
    effectiveness: f32,
    usage_count: u32,
    success_rate: f32,
) -> LowRewardRecommendation {
    // Not enough data yet
    if usage_count < 5 {
        return LowRewardRecommendation::NeedsMoreData;
    }

    // Very low reward with many uses - likely just bad
    if reward < 0.2 && usage_count >= 10 {
        return LowRewardRecommendation::Delete;
    }

    // Low reward but some effectiveness - might be contextual
    if reward < 0.4 && effectiveness > 0.5 {
        return LowRewardRecommendation::Review;
    }

    // Low reward with low success rate - archive it
    if reward < 0.4 && success_rate < 0.3 {
        return LowRewardRecommendation::Archive;
    }

    // Moderate issues - worth reviewing
    LowRewardRecommendation::Review
}

/// Analyze possible reasons for low reward.
fn analyze_low_reward_reasons(
    reward: f32,
    effectiveness: f32,
    usage_count: u32,
    success_rate: f32,
) -> Vec<String> {
    let mut reasons = Vec::new();

    if success_rate < 0.3 {
        reasons.push("Low success rate indicates pattern may not work as intended".to_string());
    }

    if effectiveness < 0.3 {
        reasons.push("Low effectiveness suggests pattern may be outdated or context-specific".to_string());
    }

    if usage_count > 10 && reward < 0.3 {
        reasons.push("Consistently low reward across many uses indicates fundamental issues".to_string());
    }

    if effectiveness > 0.6 && reward < 0.4 {
        reasons.push("High effectiveness but low reward may indicate user preference issues".to_string());
    }

    if reasons.is_empty() {
        reasons.push("No specific issues identified; may need manual investigation".to_string());
    }

    reasons
}

// ============================================================================
// Stale Pattern Detection
// ============================================================================

/// Report for stale patterns.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StalePatternReport {
    /// The pattern ID
    pub pattern_id: PatternId,

    /// Days since last use/update
    pub days_since_update: i64,

    /// Total usage count
    pub usage_count: u32,

    /// Current reward
    pub reward: f32,

    /// When the pattern was created
    pub created_at: DateTime<Utc>,

    /// When the pattern was last updated
    pub last_updated: DateTime<Utc>,

    /// Problem summary (for context)
    pub problem_summary: String,

    /// Current review status
    pub status: PatternReviewStatus,
}

/// Status for pattern review tracking.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PatternReviewStatus {
    /// Pattern is active and in use
    Active,
    /// Pattern is marked for review
    PendingReview,
    /// Pattern is archived (kept but not actively used)
    Archived,
    /// Pattern is scheduled for deletion
    MarkedForDeletion,
}

impl Default for PatternReviewStatus {
    fn default() -> Self {
        Self::Active
    }
}

/// Find patterns that are stale (old and rarely used).
///
/// # Arguments
///
/// * `storage` - Pattern storage reference
/// * `max_age_days` - Patterns older than this are considered stale (default: 30)
/// * `min_usage_count` - Patterns with fewer uses are considered stale (default: 2)
///
/// # Returns
///
/// List of reports for stale patterns.
#[instrument(skip(storage))]
pub async fn find_stale_patterns(
    storage: &PatternStorage,
    max_age_days: i64,
    min_usage_count: u32,
) -> Result<Vec<StalePatternReport>> {
    let cutoff_date = Utc::now() - Duration::days(max_age_days);
    let cutoff_str = cutoff_date.to_rfc3339();

    let sql = r#"
        SELECT * FROM reasoning_patterns
        WHERE updated_at < ?
        AND reuse_count < ?
        ORDER BY updated_at ASC
    "#;

    let usage = min_usage_count as i64;

    let patterns = storage
        .adapter()
        .sqlite()
        .query(sql, &[&cutoff_str, &usage], |row| {
            let id: String = row.get("id")?;
            let reuse_count: i32 = row.get("reuse_count")?;
            let reward: f64 = row.get("reward")?;
            let timestamp_str: String = row.get("timestamp")?;
            let updated_at_str: String = row.get("updated_at")?;
            let problem: String = row.get("problem")?;

            let created_at = DateTime::parse_from_rfc3339(&timestamp_str)
                .map(|dt| dt.with_timezone(&Utc))
                .unwrap_or_else(|_| Utc::now());

            let updated_at = DateTime::parse_from_rfc3339(&updated_at_str)
                .map(|dt| dt.with_timezone(&Utc))
                .unwrap_or_else(|_| Utc::now());

            Ok((id, reuse_count as u32, reward as f32, created_at, updated_at, problem))
        })
        .await?;

    let now = Utc::now();
    let mut reports = Vec::new();

    for (id, usage_count, reward, created_at, updated_at, problem) in patterns {
        let days_since_update = (now - updated_at).num_days();

        reports.push(StalePatternReport {
            pattern_id: PatternId::from_string(id),
            days_since_update,
            usage_count,
            reward,
            created_at,
            last_updated: updated_at,
            problem_summary: truncate_string(&problem, 100),
            status: PatternReviewStatus::Active, // Will be updated by mark_for_review
        });
    }

    info!(
        count = reports.len(),
        max_age_days = max_age_days,
        min_usage = min_usage_count,
        "Found stale patterns"
    );

    Ok(reports)
}

/// Mark a pattern for review.
///
/// Updates the pattern metadata to indicate it needs review.
#[instrument(skip(storage))]
pub async fn mark_for_review(
    storage: &PatternStorage,
    pattern_id: &PatternId,
    reason: &str,
) -> Result<()> {
    let pattern = storage
        .get_pattern(pattern_id)
        .await?
        .ok_or_else(|| NagualError::Internal {
            message: format!("Pattern not found: {}", pattern_id),
        })?;

    let mut updated = pattern.clone();

    // Add review note to critique
    let review_note = format!(
        "[MARKED FOR REVIEW: {}] {}",
        Utc::now().format("%Y-%m-%d"),
        reason
    );

    let current_critique = updated.critique();
    let new_critique = if current_critique.is_empty() {
        review_note
    } else {
        format!("{}\n---\n{}", current_critique, review_note)
    };

    updated.set_critique(new_critique);
    updated.touch();

    storage.update_pattern(&updated).await?;

    info!(
        pattern_id = %pattern_id,
        reason = reason,
        "Pattern marked for review"
    );

    Ok(())
}

/// Archive a pattern (keep it but mark as not actively used).
///
/// Archived patterns are kept for reference but won't be returned
/// in normal searches.
#[instrument(skip(storage))]
pub async fn archive_pattern(
    storage: &PatternStorage,
    pattern_id: &PatternId,
    reason: &str,
) -> Result<()> {
    let pattern = storage
        .get_pattern(pattern_id)
        .await?
        .ok_or_else(|| NagualError::Internal {
            message: format!("Pattern not found: {}", pattern_id),
        })?;

    let mut updated = pattern.clone();

    // Add archive tag, making room if at the tag limit (max 20).
    // The archive marker is stored in the critique field anyway, so
    // dropping a low-value tag to fit __archived__ is safe.
    if !updated.tags().contains(&"__archived__".to_string()) {
        const MAX_TAGS: usize = 20;
        if updated.tags().len() >= MAX_TAGS {
            // Evict the last tag to make room
            let mut tags = updated.tags().to_vec();
            tags.pop();
            tags.push("__archived__".to_string());
            updated.set_tags(tags);
        } else {
            updated.add_tag("__archived__".to_string());
        }
    }

    // Add archive note to critique
    let archive_note = format!(
        "[ARCHIVED: {}] {}",
        Utc::now().format("%Y-%m-%d"),
        reason
    );

    let current_critique = updated.critique();
    let new_critique = if current_critique.is_empty() {
        archive_note
    } else {
        format!("{}\n---\n{}", current_critique, archive_note)
    };

    updated.set_critique(new_critique);

    // Set low effectiveness to prevent retrieval
    updated.set_effectiveness(0.0);
    updated.touch();

    storage.update_pattern(&updated).await?;

    info!(
        pattern_id = %pattern_id,
        reason = reason,
        "Pattern archived"
    );

    Ok(())
}

// ============================================================================
// Pattern Consolidation
// ============================================================================

/// Consolidate similar patterns to reduce redundancy.
///
/// Patterns with similarity above the threshold are merged into a single
/// pattern that combines their metadata, tags, and related patterns.
///
/// # Algorithm
///
/// 1. Load all patterns with embeddings
/// 2. Compute pairwise similarity using cosine similarity
/// 3. Group patterns that exceed similarity threshold
/// 4. For each group, create a merged pattern:
///    - Keep the pattern with highest reward as primary
///    - Combine tags and related patterns
///    - Average/max rewards and effectiveness
///    - Sum reuse counts
/// 5. Archive or delete merged patterns
///
/// # Arguments
///
/// * `storage` - Pattern storage reference
/// * `config` - Consolidation configuration
///
/// # Returns
///
/// Consolidation result with details about merged patterns.
#[instrument(skip(storage))]
pub async fn consolidate_patterns(
    storage: &PatternStorage,
    config: &ConsolidationConfig,
) -> Result<ConsolidationResult> {
    let start = std::time::Instant::now();

    // Get patterns with embeddings
    let patterns = storage.get_all_with_embeddings().await?;

    if patterns.len() < config.min_patterns_for_consolidation {
        debug!(
            count = patterns.len(),
            min = config.min_patterns_for_consolidation,
            "Not enough patterns for consolidation"
        );
        return Ok(ConsolidationResult::empty(config.dry_run));
    }

    // Limit patterns to process
    let patterns: Vec<_> = patterns
        .into_iter()
        .take(config.max_patterns_to_process)
        .collect();

    info!(
        pattern_count = patterns.len(),
        threshold = config.similarity_threshold,
        "Starting pattern consolidation"
    );

    // Build similarity groups
    let groups = find_similar_groups(&patterns, config.similarity_threshold);

    let mut result = ConsolidationResult {
        groups: Vec::new(),
        patterns_processed: patterns.len(),
        patterns_consolidated: 0,
        groups_formed: groups.len(),
        errors: Vec::new(),
        dry_run: config.dry_run,
        duration_ms: 0,
    };

    // Process each group
    for group_indices in groups {
        if group_indices.len() < 2 {
            continue;
        }

        let group_patterns: Vec<&Pattern> = group_indices
            .iter()
            .filter_map(|&i| patterns.get(i))
            .collect();

        match merge_pattern_group(&group_patterns, config) {
            Ok(consolidated_group) => {
                result.patterns_consolidated += consolidated_group.merged_ids.len();

                // Apply changes if not dry run
                if !config.dry_run {
                    if let Err(e) = apply_consolidation(storage, &consolidated_group, config).await {
                        result.errors.push(format!(
                            "Failed to apply consolidation for group with primary {}: {}",
                            consolidated_group.primary_id, e
                        ));
                    }
                }

                result.groups.push(consolidated_group);
            }
            Err(e) => {
                result.errors.push(format!("Failed to merge group: {}", e));
            }
        }
    }

    result.duration_ms = start.elapsed().as_millis() as u64;

    info!(
        groups_formed = result.groups_formed,
        patterns_consolidated = result.patterns_consolidated,
        duration_ms = result.duration_ms,
        dry_run = config.dry_run,
        "Pattern consolidation complete"
    );

    Ok(result)
}

/// Find groups of similar patterns using their embeddings.
fn find_similar_groups(patterns: &[Pattern], threshold: f32) -> Vec<Vec<usize>> {
    let n = patterns.len();
    let mut visited = vec![false; n];
    let mut groups = Vec::new();

    for i in 0..n {
        if visited[i] {
            continue;
        }

        let Some(emb_i) = patterns[i].embedding() else {
            continue;
        };

        let mut group = vec![i];
        visited[i] = true;

        for j in (i + 1)..n {
            if visited[j] {
                continue;
            }

            let Some(emb_j) = patterns[j].embedding() else {
                continue;
            };

            let similarity = cosine_similarity(emb_i, emb_j);

            if similarity >= threshold {
                group.push(j);
                visited[j] = true;
            }
        }

        if group.len() >= 2 {
            groups.push(group);
        }
    }

    groups
}

/// Compute cosine similarity between two vectors.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let mut dot = 0.0;
    let mut norm_a = 0.0;
    let mut norm_b = 0.0;

    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }

    let denom = (norm_a.sqrt()) * (norm_b.sqrt());
    if denom == 0.0 {
        0.0
    } else {
        dot / denom
    }
}

/// Merge a group of similar patterns.
fn merge_pattern_group(
    patterns: &[&Pattern],
    config: &ConsolidationConfig,
) -> Result<ConsolidatedGroup> {
    if patterns.is_empty() {
        return Err(NagualError::Internal {
            message: "Empty pattern group".to_string(),
        });
    }

    // Find primary pattern (highest reward)
    let primary = patterns
        .iter()
        .max_by(|a, b| a.reward().partial_cmp(&b.reward()).unwrap())
        .unwrap();

    let primary_id = primary.id().clone();

    // Collect merged IDs (all except primary)
    let merged_ids: Vec<PatternId> = patterns
        .iter()
        .filter(|p| p.id() != &primary_id)
        .map(|p| p.id().clone())
        .collect();

    // Combine tags: filter noise, deduplicate, and cap to max_tags.
    // Noise tags accumulate across consolidation cycles and provide no semantic value.
    let noise_tags: HashSet<&str> = ["__archived__", "imported"].iter().copied().collect();
    let mut combined_tags: HashSet<String> = HashSet::new();
    for p in patterns {
        for tag in p.tags() {
            if !noise_tags.contains(tag.as_str()) {
                combined_tags.insert(tag.clone());
            }
        }
    }

    // If over the limit, keep primary's tags first, then rank by frequency across patterns
    if combined_tags.len() > config.max_tags {
        let primary_tags: HashSet<_> = primary.tags().iter().cloned().collect();
        let mut scored: Vec<(String, usize)> = combined_tags
            .iter()
            .map(|t| {
                let freq = patterns.iter().filter(|p| p.tags().contains(t)).count();
                // Primary's tags get a large boost so they're always kept
                let primary_boost = if primary_tags.contains(t) { 1000 } else { 0 };
                (t.clone(), freq + primary_boost)
            })
            .collect();
        scored.sort_by(|a, b| b.1.cmp(&a.1));
        combined_tags = scored
            .into_iter()
            .take(config.max_tags)
            .map(|(t, _)| t)
            .collect();
    }

    // Combine related patterns (unique)
    let mut combined_related: HashSet<PatternId> = HashSet::new();
    for p in patterns {
        for related in p.related_patterns() {
            // Don't include patterns from this group
            if !patterns.iter().any(|pat| pat.id() == related) {
                combined_related.insert(related.clone());
            }
        }
    }

    // Merge reward based on strategy
    let merged_reward = match config.reward_merge_strategy {
        RewardMergeStrategy::Average => {
            patterns.iter().map(|p| p.reward()).sum::<f32>() / patterns.len() as f32
        }
        RewardMergeStrategy::Maximum => {
            patterns.iter().map(|p| p.reward()).fold(0.0_f32, f32::max)
        }
        RewardMergeStrategy::Minimum => {
            patterns.iter().map(|p| p.reward()).fold(1.0_f32, f32::min)
        }
        RewardMergeStrategy::WeightedByUsage => {
            let total_usage: f32 = patterns.iter().map(|p| p.reuse_count() as f32).sum();
            if total_usage == 0.0 {
                patterns.iter().map(|p| p.reward()).sum::<f32>() / patterns.len() as f32
            } else {
                patterns
                    .iter()
                    .map(|p| p.reward() * p.reuse_count() as f32)
                    .sum::<f32>()
                    / total_usage
            }
        }
    };

    // Average effectiveness
    let merged_effectiveness =
        patterns.iter().map(|p| p.effectiveness()).sum::<f32>() / patterns.len() as f32;

    // Sum reuse counts (keep the higher one for more accurate total)
    let total_reuse_count: u32 = patterns.iter().map(|p| p.reuse_count()).max().unwrap_or(0)
        + patterns
            .iter()
            .map(|p| p.reuse_count())
            .sum::<u32>()
            .saturating_sub(patterns.iter().map(|p| p.reuse_count()).max().unwrap_or(0));

    // Calculate average similarity (simplified - just using first pattern as reference)
    let avg_similarity = if patterns.len() > 1 {
        if let Some(ref_emb) = patterns[0].embedding() {
            let sims: Vec<f32> = patterns[1..]
                .iter()
                .filter_map(|p| p.embedding())
                .map(|emb| cosine_similarity(ref_emb, emb))
                .collect();

            if sims.is_empty() {
                1.0
            } else {
                sims.iter().sum::<f32>() / sims.len() as f32
            }
        } else {
            1.0
        }
    } else {
        1.0
    };

    Ok(ConsolidatedGroup {
        primary_id,
        merged_ids,
        average_similarity: avg_similarity,
        combined_tags: combined_tags.into_iter().collect(),
        combined_related: combined_related.into_iter().collect(),
        merged_reward,
        merged_effectiveness,
        total_reuse_count,
    })
}

/// Apply consolidation changes to storage.
async fn apply_consolidation(
    storage: &PatternStorage,
    group: &ConsolidatedGroup,
    config: &ConsolidationConfig,
) -> Result<()> {
    // Update primary pattern with merged data
    let primary = storage
        .get_pattern(&group.primary_id)
        .await?
        .ok_or_else(|| NagualError::Internal {
            message: format!("Primary pattern not found: {}", group.primary_id),
        })?;

    let mut updated_primary = primary.clone();

    // Update reward and effectiveness
    updated_primary.set_reward(group.merged_reward);
    updated_primary.set_effectiveness(group.merged_effectiveness);

    // Replace tags with the curated set from merge (noise-filtered and capped)
    updated_primary.set_tags(group.combined_tags.clone());

    // Add combined related patterns
    for related in &group.combined_related {
        updated_primary.add_related_pattern(related.clone());
    }

    // Add consolidation note
    let consolidation_note = format!(
        "[CONSOLIDATED: {}] Merged {} patterns with avg similarity {:.2}",
        Utc::now().format("%Y-%m-%d"),
        group.merged_ids.len(),
        group.average_similarity
    );

    let current_critique = updated_primary.critique();
    let new_critique = if current_critique.is_empty() {
        consolidation_note
    } else {
        format!("{}\n---\n{}", current_critique, consolidation_note)
    };

    updated_primary.set_critique(new_critique);
    updated_primary.touch();

    // Save updated primary
    storage.update_pattern(&updated_primary).await?;

    // Archive or delete merged patterns
    for merged_id in &group.merged_ids {
        if config.keep_archived {
            archive_pattern(
                storage,
                merged_id,
                &format!("Consolidated into {}", group.primary_id),
            )
            .await?;
        } else {
            storage.delete_pattern(merged_id).await?;
        }
    }

    Ok(())
}

/// Truncate a string to a maximum length.
fn truncate_string(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len - 3])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_consolidation_config_default() {
        let config = ConsolidationConfig::default();
        assert!((config.similarity_threshold - 0.85).abs() < 0.001);
        assert_eq!(config.min_patterns_for_consolidation, 2);
        assert!(!config.dry_run);
        assert!(config.keep_archived);
    }

    #[test]
    fn test_cosine_similarity() {
        // Identical vectors
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 0.001);

        // Orthogonal vectors
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        assert!(cosine_similarity(&a, &b).abs() < 0.001);

        // Opposite vectors
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![-1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) + 1.0).abs() < 0.001);

        // Similar but not identical
        let a = vec![1.0, 0.5, 0.0];
        let b = vec![1.0, 0.4, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim > 0.99);
    }

    #[test]
    fn test_cosine_similarity_edge_cases() {
        // Empty vectors
        assert_eq!(cosine_similarity(&[], &[]), 0.0);

        // Different lengths
        assert_eq!(cosine_similarity(&[1.0], &[1.0, 2.0]), 0.0);

        // Zero vectors
        assert_eq!(cosine_similarity(&[0.0, 0.0], &[0.0, 0.0]), 0.0);
    }

    #[test]
    fn test_low_reward_recommendation() {
        // Not enough data
        let rec = determine_low_reward_recommendation(0.3, 0.5, 3, 0.5);
        assert_eq!(rec, LowRewardRecommendation::NeedsMoreData);

        // Very low reward with many uses
        let rec = determine_low_reward_recommendation(0.1, 0.3, 15, 0.2);
        assert_eq!(rec, LowRewardRecommendation::Delete);

        // Low reward but good effectiveness
        let rec = determine_low_reward_recommendation(0.35, 0.7, 10, 0.6);
        assert_eq!(rec, LowRewardRecommendation::Review);

        // Low reward and low success rate
        let rec = determine_low_reward_recommendation(0.35, 0.3, 10, 0.2);
        assert_eq!(rec, LowRewardRecommendation::Archive);
    }

    #[test]
    fn test_analyze_low_reward_reasons() {
        let reasons = analyze_low_reward_reasons(0.2, 0.2, 15, 0.1);
        assert!(!reasons.is_empty());
        assert!(reasons.iter().any(|r| r.contains("success rate")));
        assert!(reasons.iter().any(|r| r.contains("effectiveness")));
        assert!(reasons.iter().any(|r| r.contains("Consistently low")));
    }

    #[test]
    fn test_truncate_string() {
        assert_eq!(truncate_string("short", 10), "short");
        assert_eq!(truncate_string("this is a long string", 10), "this is...");
        assert_eq!(truncate_string("exact len!", 10), "exact len!");
    }

    #[test]
    fn test_reward_merge_strategies() {
        let rewards = vec![0.3, 0.5, 0.7, 0.9];

        // Average
        let avg: f32 = rewards.iter().sum::<f32>() / rewards.len() as f32;
        assert!((avg - 0.6).abs() < 0.001);

        // Max
        let max = rewards.iter().cloned().fold(0.0_f32, f32::max);
        assert!((max - 0.9).abs() < 0.001);

        // Min
        let min = rewards.iter().cloned().fold(1.0_f32, f32::min);
        assert!((min - 0.3).abs() < 0.001);
    }

    #[test]
    fn test_consolidation_result_empty() {
        let result = ConsolidationResult::empty(true);
        assert!(result.dry_run);
        assert_eq!(result.groups.len(), 0);
        assert_eq!(result.patterns_processed, 0);
    }

    #[test]
    fn test_pattern_review_status_default() {
        let status = PatternReviewStatus::default();
        assert_eq!(status, PatternReviewStatus::Active);
    }

    #[test]
    fn test_low_reward_recommendation_description() {
        assert!(LowRewardRecommendation::Review.description().contains("review"));
        assert!(LowRewardRecommendation::Archive.description().contains("archived"));
        assert!(LowRewardRecommendation::Delete.description().contains("deleted"));
    }

    #[test]
    fn test_merge_filters_noise_tags() {
        let p1 = Pattern::builder()
            .problem("p1")
            .solution("s1")
            .reward(0.9)
            .tag("rust")
            .tag("async")
            .tag("__archived__")
            .tag("imported")
            .build();
        let p2 = Pattern::builder()
            .problem("p2")
            .solution("s2")
            .reward(0.5)
            .tag("tokio")
            .tag("__archived__")
            .tag("__archived__")
            .tag("imported")
            .build();

        let config = ConsolidationConfig::default();
        let patterns: Vec<&Pattern> = vec![&p1, &p2];
        let group = merge_pattern_group(&patterns, &config).unwrap();

        // Noise tags filtered out, only semantic tags remain
        assert!(!group.combined_tags.contains(&"__archived__".to_string()));
        assert!(!group.combined_tags.contains(&"imported".to_string()));
        assert!(group.combined_tags.contains(&"rust".to_string()));
        assert!(group.combined_tags.contains(&"async".to_string()));
        assert!(group.combined_tags.contains(&"tokio".to_string()));
    }

    #[test]
    fn test_merge_caps_tags_to_max() {
        // Create a pattern with many unique tags that would exceed max_tags
        let mut builder1 = Pattern::builder().problem("p1").solution("s1").reward(0.9);
        for i in 0..12 {
            builder1 = builder1.tag(format!("primary-tag-{}", i));
        }
        let p1 = builder1.build();

        let mut builder2 = Pattern::builder().problem("p2").solution("s2").reward(0.5);
        for i in 0..12 {
            builder2 = builder2.tag(format!("secondary-tag-{}", i));
        }
        let p2 = builder2.build();

        let mut config = ConsolidationConfig::default();
        config.max_tags = 15;
        let patterns: Vec<&Pattern> = vec![&p1, &p2];
        let group = merge_pattern_group(&patterns, &config).unwrap();

        // Must not exceed max_tags
        assert!(group.combined_tags.len() <= 15);

        // Primary's tags should be preserved (they get priority boost)
        let primary_kept = group
            .combined_tags
            .iter()
            .filter(|t| t.starts_with("primary-tag-"))
            .count();
        assert_eq!(primary_kept, 12, "all primary tags should be kept");
    }

    #[test]
    fn test_merge_under_limit_keeps_all() {
        let p1 = Pattern::builder()
            .problem("p1")
            .solution("s1")
            .reward(0.8)
            .tag("a")
            .tag("b")
            .build();
        let p2 = Pattern::builder()
            .problem("p2")
            .solution("s2")
            .reward(0.5)
            .tag("c")
            .tag("d")
            .build();

        let config = ConsolidationConfig::default(); // max_tags = 20
        let patterns: Vec<&Pattern> = vec![&p1, &p2];
        let group = merge_pattern_group(&patterns, &config).unwrap();

        // 4 unique tags, well under 20 — all kept
        assert_eq!(group.combined_tags.len(), 4);
    }

    #[test]
    fn test_config_max_tags_default() {
        let config = ConsolidationConfig::default();
        assert_eq!(config.max_tags, 20);
    }
}
