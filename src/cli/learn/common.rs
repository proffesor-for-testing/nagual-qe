//! Shared utilities for learning commands.

use std::path::PathBuf;
use std::sync::Arc;

use crate::cli::common::init_storage;
use crate::error::Result;
use crate::learning::{
    archive_pattern, consolidate_patterns, mark_for_review, ConsolidationTrigger,
    PatternConsolidationConfig, Recommendation, RecommendationType, TimeWindow,
};
use crate::reasoning_bank::pattern::{Pattern, PatternCategory};
use crate::reasoning_bank::storage::PatternStorage;

/// Load patterns from database or return demo patterns if demo flag is set.
pub async fn load_patterns_from_db(
    db_path: &PathBuf,
    demo: bool,
    limit: usize,
) -> Result<Vec<Pattern>> {
    if demo {
        return Ok(create_demo_patterns());
    }

    let storage = init_storage(db_path, None).await?;
    let patterns = storage.get_recent(limit).await?;

    if patterns.is_empty() {
        tracing::info!("No patterns found in database, using demo data");
        Ok(create_demo_patterns())
    } else {
        Ok(patterns)
    }
}

/// Apply a single recommendation to the database.
///
/// Returns the number of patterns affected.
pub async fn apply_recommendation_impl(
    storage: &Arc<PatternStorage>,
    recommendation: &Recommendation,
    verbose: bool,
) -> Result<usize> {
    let mut affected_count = 0;

    match recommendation.recommendation_type {
        RecommendationType::Archive => {
            // Archive low-performing patterns
            for pattern_id in &recommendation.target_patterns {
                if let Err(e) =
                    archive_pattern(storage, pattern_id, &recommendation.rationale).await
                {
                    if verbose {
                        tracing::warn!("Failed to archive pattern {}: {}", pattern_id, e);
                    }
                } else {
                    affected_count += 1;
                }
            }
        }
        RecommendationType::Review => {
            // Mark patterns for review
            for pattern_id in &recommendation.target_patterns {
                if let Err(e) =
                    mark_for_review(storage, pattern_id, &recommendation.rationale).await
                {
                    if verbose {
                        tracing::warn!("Failed to mark pattern {} for review: {}", pattern_id, e);
                    }
                } else {
                    affected_count += 1;
                }
            }
        }
        RecommendationType::Consolidate => {
            // Run consolidation on the target patterns
            if recommendation.target_patterns.len() >= 2 {
                let config = PatternConsolidationConfig::default();
                match consolidate_patterns(storage, &config).await {
                    Ok(result) => {
                        affected_count = result.patterns_consolidated;
                        if verbose {
                            tracing::info!(
                                "Consolidated {} patterns into {} groups",
                                result.patterns_consolidated,
                                result.groups_formed
                            );
                        }
                    }
                    Err(e) => {
                        if verbose {
                            tracing::warn!("Consolidation failed: {}", e);
                        }
                    }
                }
            }
        }
        RecommendationType::Promote | RecommendationType::Improve | RecommendationType::Split => {
            // These require manual intervention or more complex logic
            // For now, just mark them for review
            for pattern_id in &recommendation.target_patterns {
                let reason = format!(
                    "{} recommended: {}",
                    recommendation.recommendation_type, recommendation.rationale
                );
                if let Err(e) = mark_for_review(storage, pattern_id, &reason).await {
                    if verbose {
                        tracing::warn!("Failed to mark pattern {} for review: {}", pattern_id, e);
                    }
                } else {
                    affected_count += 1;
                }
            }
        }
    }

    Ok(affected_count)
}

/// Parse time window string.
pub fn parse_time_windows(input: &str) -> Vec<TimeWindow> {
    input
        .split(',')
        .filter_map(|s| {
            let s = s.trim().to_lowercase();
            match s.as_str() {
                "24h" => Some(TimeWindow::Hours24),
                "7d" => Some(TimeWindow::Days7),
                "30d" => Some(TimeWindow::Days30),
                "90d" => Some(TimeWindow::Days90),
                "365d" => Some(TimeWindow::Days365),
                "all" => Some(TimeWindow::AllTime),
                _ => {
                    // Try to parse custom format like "48h"
                    if s.ends_with('h') {
                        s[..s.len() - 1].parse().ok().map(TimeWindow::Custom)
                    } else {
                        None
                    }
                }
            }
        })
        .collect()
}

/// Parse trigger type string.
#[allow(dead_code)]
pub fn parse_trigger(input: &str) -> ConsolidationTrigger {
    match input.to_lowercase().as_str() {
        "manual" => ConsolidationTrigger::Manual,
        "time" => ConsolidationTrigger::time_based(),
        "count" => ConsolidationTrigger::count_based(),
        _ => ConsolidationTrigger::Manual,
    }
}

/// Create demo patterns for testing.
pub fn create_demo_patterns() -> Vec<Pattern> {
    vec![
        Pattern::builder()
            .problem("How to handle async errors in Rust")
            .solution("Use Result type with async/await and proper error propagation")
            .category(PatternCategory::Resilience)
            .reward(0.92)
            .reuse_count(15)
            .effectiveness(0.88)
            .confidence(0.90)
            .tag("rust")
            .tag("async")
            .tag("error-handling")
            .build(),
        Pattern::builder()
            .problem("Database connection pooling best practices")
            .solution("Use sqlx pool with proper configuration and health checks")
            .category(PatternCategory::Performance)
            .reward(0.85)
            .reuse_count(12)
            .effectiveness(0.82)
            .confidence(0.88)
            .tag("database")
            .tag("performance")
            .build(),
        Pattern::builder()
            .problem("API rate limiting implementation")
            .solution("Implement token bucket algorithm with Redis backend")
            .category(PatternCategory::ApiDesign)
            .reward(0.78)
            .reuse_count(8)
            .effectiveness(0.75)
            .confidence(0.80)
            .tag("api")
            .tag("rate-limiting")
            .build(),
        Pattern::builder()
            .problem("Memory leak detection in long-running services")
            .solution("Use valgrind and memory profiling with periodic snapshots")
            .category(PatternCategory::Performance)
            .reward(0.65)
            .reuse_count(5)
            .effectiveness(0.60)
            .confidence(0.70)
            .tag("memory")
            .tag("debugging")
            .build(),
        Pattern::builder()
            .problem("Outdated caching strategy")
            .solution("Simple in-memory cache without TTL")
            .category(PatternCategory::Performance)
            .reward(0.35)
            .reuse_count(8)
            .effectiveness(0.30)
            .confidence(0.40)
            .tag("caching")
            .tag("outdated")
            .build(),
        Pattern::builder()
            .problem("Complex monolithic authentication")
            .solution(
                "A very long and complex authentication solution that handles multiple auth providers \
                including OAuth2, SAML, OpenID Connect, and custom token-based authentication. \
                It also includes session management, token refresh, rate limiting per user, \
                IP-based blocking, device fingerprinting, and comprehensive audit logging. \
                The implementation spans multiple modules and requires careful coordination \
                between the auth server, session store, and the main application. \
                This pattern could potentially be split into smaller, more focused patterns.",
            )
            .category(PatternCategory::Security)
            .reward(0.70)
            .reuse_count(3)
            .effectiveness(0.65)
            .confidence(0.72)
            .tag("auth")
            .tag("security")
            .tag("complex")
            .build(),
        Pattern::builder()
            .problem("Testing async code patterns")
            .solution("Use tokio-test with proper timeout handling")
            .category(PatternCategory::Testing)
            .reward(0.88)
            .reuse_count(20)
            .effectiveness(0.85)
            .confidence(0.90)
            .tag("testing")
            .tag("async")
            .build(),
        Pattern::builder()
            .problem("Rust lifetime annotations for complex data structures")
            .solution("Use explicit lifetime bounds with PhantomData for advanced cases")
            .category(PatternCategory::CodeQuality)
            .reward(0.75)
            .reuse_count(6)
            .effectiveness(0.70)
            .confidence(0.78)
            .tag("rust")
            .tag("lifetimes")
            .build(),
    ]
}
