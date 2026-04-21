//! Pyramid Summary Auto-Generator for Nagual patterns.
//!
//! Provides extraction-based generation of title (10 words) and summary (50 words)
//! for patterns that are missing them. This implements a hierarchical summary approach:
//!
//! - **Title**: First 10 words of the problem, cleaned of trailing punctuation
//! - **Summary**: First paragraph or 50 words of the solution
//!
//! The pyramid structure allows:
//! - Quick scanning via titles in listing views
//! - Medium-depth understanding via summaries
//! - Full context via problem + solution when needed
//!
//! # Usage
//!
//! ```bash
//! # Show pyramid statistics
//! nagual patterns pyramid --stats
//!
//! # Generate pyramids for patterns missing them (dry run)
//! nagual patterns pyramid --generate --dry-run
//!
//! # Generate pyramids for real
//! nagual patterns pyramid --generate
//!
//! # Limit processing to first N patterns
//! nagual patterns pyramid --generate --limit 100
//! ```

use std::sync::Arc;

use tracing::{debug, info};

use crate::db::{DualWriteAdapter, DualWriteConfig, SqliteDb};
use crate::error::Result;

/// Statistics about pyramid summaries in the pattern database.
#[derive(Debug, Clone, Default)]
pub struct PyramidStats {
    /// Total number of patterns in the database.
    pub total_patterns: usize,

    /// Number of patterns with both title and summary.
    pub with_pyramid: usize,

    /// Number of patterns missing title or summary (or both).
    pub without_pyramid: usize,

    /// Number of patterns with only title.
    pub with_title_only: usize,

    /// Number of patterns with only summary.
    pub with_summary_only: usize,

    /// Number of pyramids generated in this run (for generate command).
    pub generated: usize,
}

impl PyramidStats {
    /// Calculate the percentage of patterns with pyramid summaries.
    pub fn coverage_percent(&self) -> f32 {
        if self.total_patterns == 0 {
            0.0
        } else {
            100.0 * self.with_pyramid as f32 / self.total_patterns as f32
        }
    }
}

/// Generate a title from problem text.
///
/// Extracts the first 10 words from the problem description, cleaning
/// trailing punctuation while preserving sentence readability.
///
/// # Arguments
///
/// * `problem` - The problem text to extract a title from
///
/// # Returns
///
/// A string containing the first 10 words of the problem.
///
/// # Examples
///
/// ```
/// use nagual::reasoning_bank::pyramid::generate_title;
///
/// let problem = "How to implement a rate limiter for API endpoints using token bucket algorithm";
/// let title = generate_title(problem);
/// assert_eq!(title, "How to implement a rate limiter for API endpoints using");
/// ```
pub fn generate_title(problem: &str) -> String {
    let words: Vec<&str> = problem.split_whitespace().take(10).collect();
    let title = words.join(" ");

    // Trim trailing punctuation that might look awkward (but keep essential ones like ?)
    title
        .trim_end_matches(|c: char| matches!(c, ',' | ';' | ':' | '-' | '.'))
        .to_string()
}

/// Generate a summary from solution text.
///
/// Extracts the first paragraph or first ~50 words from the solution,
/// whichever is shorter. Adds ellipsis if truncated.
///
/// If the solution is empty, returns "(No solution provided)".
///
/// # Arguments
///
/// * `solution` - The solution text to extract a summary from
///
/// # Returns
///
/// A string containing the summary (up to 50 words), with ellipsis if truncated.
///
/// # Examples
///
/// ```
/// use nagual::reasoning_bank::pyramid::generate_summary;
///
/// let solution = "Use Redis with TTL-based expiration. This provides fast lookups.";
/// let summary = generate_summary(solution);
/// assert_eq!(summary, "Use Redis with TTL-based expiration. This provides fast lookups.");
/// ```
pub fn generate_summary(solution: &str) -> String {
    let trimmed = solution.trim();

    // Handle empty solutions
    if trimmed.is_empty() {
        return "(No solution provided)".to_string();
    }

    // Try to get first paragraph (split on double newlines)
    let first_para = trimmed.split("\n\n").next().unwrap_or(trimmed).trim();

    // Also consider single newlines for simpler content
    let first_logical = first_para.split('\n').next().unwrap_or(first_para).trim();

    // Use the shorter of first paragraph or first line if first line is substantial
    let base = if first_logical.len() >= 80 && first_logical.len() < first_para.len() {
        first_logical
    } else {
        first_para
    };

    // Limit to ~50 words
    let words: Vec<&str> = base.split_whitespace().collect();

    if words.is_empty() {
        "(No solution provided)".to_string()
    } else if words.len() <= 50 {
        words.join(" ")
    } else {
        let truncated = words[..50].join(" ");
        format!("{}...", truncated)
    }
}

/// Query patterns that are missing pyramid summaries.
///
/// Returns pattern IDs, problem, and solution for patterns that need
/// title and/or summary generation.
pub async fn get_patterns_without_pyramid(
    adapter: &Arc<DualWriteAdapter>,
    limit: Option<usize>,
) -> Result<Vec<PatternForPyramid>> {
    let limit_clause = match limit {
        Some(n) => format!("LIMIT {}", n),
        None => String::new(),
    };

    let sql = format!(
        r#"
        SELECT id, problem, solution
        FROM reasoning_patterns
        WHERE title IS NULL OR summary IS NULL OR title = '' OR summary = ''
        ORDER BY timestamp DESC
        {}
        "#,
        limit_clause
    );

    let patterns = adapter
        .sqlite()
        .query(&sql, &[], |row| {
            Ok(PatternForPyramid {
                id: row.get("id")?,
                problem: row.get("problem")?,
                solution: row.get("solution")?,
            })
        })
        .await?;

    Ok(patterns)
}

/// Minimal pattern data needed for pyramid generation.
#[derive(Debug, Clone)]
pub struct PatternForPyramid {
    /// Pattern ID.
    pub id: String,
    /// Problem text (source for title).
    pub problem: String,
    /// Solution text (source for summary).
    pub solution: String,
}

/// Update a pattern's pyramid fields (title and summary).
pub async fn update_pattern_pyramid(
    adapter: &Arc<DualWriteAdapter>,
    id: &str,
    title: &str,
    summary: &str,
) -> Result<()> {
    let sql = r#"
        UPDATE reasoning_patterns
        SET title = ?, summary = ?, updated_at = datetime('now')
        WHERE id = ?
    "#;

    adapter
        .sqlite()
        .execute(sql, &[&title.to_string(), &summary.to_string(), &id.to_string()])
        .await?;

    debug!(pattern_id = %id, "Updated pyramid summary");
    Ok(())
}

/// Count total patterns in the database.
pub async fn count_all_patterns(adapter: &Arc<DualWriteAdapter>) -> Result<usize> {
    let sql = "SELECT COUNT(*) FROM reasoning_patterns";
    let count = adapter
        .sqlite()
        .query_one(sql, &[], |row| row.get::<_, i64>(0))
        .await?
        .unwrap_or(0);
    Ok(count as usize)
}

/// Count patterns with complete pyramid summaries (both title and summary).
pub async fn count_patterns_with_pyramid(adapter: &Arc<DualWriteAdapter>) -> Result<usize> {
    let sql = r#"
        SELECT COUNT(*) FROM reasoning_patterns
        WHERE title IS NOT NULL AND summary IS NOT NULL
          AND title != '' AND summary != ''
    "#;
    let count = adapter
        .sqlite()
        .query_one(sql, &[], |row| row.get::<_, i64>(0))
        .await?
        .unwrap_or(0);
    Ok(count as usize)
}

/// Count patterns with only title set.
pub async fn count_patterns_with_title_only(adapter: &Arc<DualWriteAdapter>) -> Result<usize> {
    let sql = r#"
        SELECT COUNT(*) FROM reasoning_patterns
        WHERE title IS NOT NULL AND title != ''
          AND (summary IS NULL OR summary = '')
    "#;
    let count = adapter
        .sqlite()
        .query_one(sql, &[], |row| row.get::<_, i64>(0))
        .await?
        .unwrap_or(0);
    Ok(count as usize)
}

/// Count patterns with only summary set.
pub async fn count_patterns_with_summary_only(adapter: &Arc<DualWriteAdapter>) -> Result<usize> {
    let sql = r#"
        SELECT COUNT(*) FROM reasoning_patterns
        WHERE (title IS NULL OR title = '')
          AND summary IS NOT NULL AND summary != ''
    "#;
    let count = adapter
        .sqlite()
        .query_one(sql, &[], |row| row.get::<_, i64>(0))
        .await?
        .unwrap_or(0);
    Ok(count as usize)
}

/// Get pyramid statistics for the database.
pub async fn get_pyramid_stats(adapter: &Arc<DualWriteAdapter>) -> Result<PyramidStats> {
    let total = count_all_patterns(adapter).await?;
    let with_pyramid = count_patterns_with_pyramid(adapter).await?;
    let with_title_only = count_patterns_with_title_only(adapter).await?;
    let with_summary_only = count_patterns_with_summary_only(adapter).await?;

    let without_pyramid = total.saturating_sub(with_pyramid);

    Ok(PyramidStats {
        total_patterns: total,
        with_pyramid,
        without_pyramid,
        with_title_only,
        with_summary_only,
        generated: 0,
    })
}

/// Generate pyramid summaries for patterns missing them.
///
/// Iterates through patterns without title/summary and generates them
/// using extraction-based methods.
///
/// # Arguments
///
/// * `adapter` - The database adapter
/// * `dry_run` - If true, only calculate what would be done without making changes
/// * `limit` - Optional limit on number of patterns to process
///
/// # Returns
///
/// Statistics about the generation process.
pub async fn generate_missing_pyramids(
    adapter: &Arc<DualWriteAdapter>,
    dry_run: bool,
    limit: Option<usize>,
) -> Result<PyramidStats> {
    let patterns = get_patterns_without_pyramid(adapter, limit).await?;
    let patterns_to_process = patterns.len();

    info!(
        "Found {} patterns without pyramid summaries{}",
        patterns_to_process,
        if dry_run { " (dry run)" } else { "" }
    );

    let mut generated = 0;

    for pattern in &patterns {
        let title = generate_title(&pattern.problem);
        let summary = generate_summary(&pattern.solution);

        if !dry_run {
            update_pattern_pyramid(adapter, &pattern.id, &title, &summary).await?;
            generated += 1;

            // Log progress every 100 patterns
            if generated % 100 == 0 {
                info!("Generated {} / {} pyramids", generated, patterns_to_process);
            }
        }
    }

    if !dry_run && generated > 0 {
        info!("Generated {} pyramid summaries", generated);
    }

    // Get updated stats
    let total = count_all_patterns(adapter).await?;
    let with_pyramid = if dry_run {
        count_patterns_with_pyramid(adapter).await?
    } else {
        count_patterns_with_pyramid(adapter).await?
    };
    let with_title_only = count_patterns_with_title_only(adapter).await?;
    let with_summary_only = count_patterns_with_summary_only(adapter).await?;

    Ok(PyramidStats {
        total_patterns: total,
        with_pyramid,
        without_pyramid: total.saturating_sub(with_pyramid),
        with_title_only,
        with_summary_only,
        generated: if dry_run { patterns_to_process } else { generated },
    })
}

/// Initialize the adapter from a database path.
///
/// This also ensures the schema migrations are applied (adding title/summary columns).
pub async fn init_adapter(
    db_path: &std::path::Path,
) -> Result<Arc<DualWriteAdapter>> {
    // Ensure parent directory exists
    if let Some(parent) = db_path.parent() {
        if !parent.exists() {
            std::fs::create_dir_all(parent)?;
        }
    }

    // Open SQLite database
    let sqlite = Arc::new(SqliteDb::open(db_path)?);

    // Create DualWriteAdapter (SQLite-only mode for local CLI)
    let config = DualWriteConfig {
        dlq_path: db_path
            .with_extension("dlq.db")
            .to_string_lossy()
            .to_string(),
        ..Default::default()
    };
    let adapter = Arc::new(DualWriteAdapter::new(sqlite, None, config)?);

    // Ensure pyramid columns exist (safe to run multiple times)
    ensure_pyramid_columns(&adapter).await?;

    Ok(adapter)
}

/// Ensure the title and summary columns exist in the database.
///
/// This migration is idempotent - it's safe to run multiple times.
async fn ensure_pyramid_columns(adapter: &Arc<DualWriteAdapter>) -> Result<()> {
    // Try to add columns; ignore errors if they already exist
    let _ = adapter
        .sqlite()
        .execute("ALTER TABLE reasoning_patterns ADD COLUMN title TEXT", &[])
        .await;
    let _ = adapter
        .sqlite()
        .execute("ALTER TABLE reasoning_patterns ADD COLUMN summary TEXT", &[])
        .await;

    debug!("Ensured pyramid columns exist");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_title_short() {
        let problem = "How to cache";
        let title = generate_title(problem);
        assert_eq!(title, "How to cache");
    }

    #[test]
    fn test_generate_title_exactly_10_words() {
        let problem = "one two three four five six seven eight nine ten";
        let title = generate_title(problem);
        assert_eq!(title, "one two three four five six seven eight nine ten");
    }

    #[test]
    fn test_generate_title_more_than_10_words() {
        let problem = "one two three four five six seven eight nine ten eleven twelve";
        let title = generate_title(problem);
        assert_eq!(title, "one two three four five six seven eight nine ten");
    }

    #[test]
    fn test_generate_title_trailing_punctuation() {
        let problem = "How to implement caching using Redis, with proper expiration,";
        let title = generate_title(problem);
        assert_eq!(title, "How to implement caching using Redis, with proper expiration");
    }

    #[test]
    fn test_generate_title_preserves_question_mark() {
        let problem = "How do you implement caching using Redis? What is the best approach?";
        let title = generate_title(problem);
        // Should take first 10 words
        assert_eq!(title, "How do you implement caching using Redis? What is the");
    }

    #[test]
    fn test_generate_summary_short() {
        let solution = "Use Redis with TTL expiration.";
        let summary = generate_summary(solution);
        assert_eq!(summary, "Use Redis with TTL expiration.");
    }

    #[test]
    fn test_generate_summary_empty() {
        let solution = "";
        let summary = generate_summary(solution);
        assert_eq!(summary, "(No solution provided)");
    }

    #[test]
    fn test_generate_summary_whitespace_only() {
        let solution = "   \n\t  ";
        let summary = generate_summary(solution);
        assert_eq!(summary, "(No solution provided)");
    }

    #[test]
    fn test_generate_summary_with_paragraph() {
        let solution = "Use Redis with TTL expiration.\n\nThis provides fast lookups and automatic cleanup.";
        let summary = generate_summary(solution);
        assert_eq!(summary, "Use Redis with TTL expiration.");
    }

    #[test]
    fn test_generate_summary_long() {
        let words: Vec<&str> = (0..60).map(|_| "word").collect();
        let solution = words.join(" ");
        let summary = generate_summary(&solution);

        // Should be truncated to 50 words with ellipsis
        assert!(summary.ends_with("..."));
        let word_count = summary.trim_end_matches("...").split_whitespace().count();
        assert_eq!(word_count, 50);
    }

    #[test]
    fn test_generate_summary_exactly_50_words() {
        let words: Vec<&str> = (0..50).map(|_| "word").collect();
        let solution = words.join(" ");
        let summary = generate_summary(&solution);

        // Should NOT have ellipsis
        assert!(!summary.ends_with("..."));
        assert_eq!(summary.split_whitespace().count(), 50);
    }

    #[test]
    fn test_pyramid_stats_coverage() {
        let stats = PyramidStats {
            total_patterns: 100,
            with_pyramid: 75,
            without_pyramid: 25,
            with_title_only: 10,
            with_summary_only: 5,
            generated: 0,
        };

        assert!((stats.coverage_percent() - 75.0).abs() < 0.001);
    }

    #[test]
    fn test_pyramid_stats_coverage_empty() {
        let stats = PyramidStats::default();
        assert!((stats.coverage_percent() - 0.0).abs() < 0.001);
    }
}
