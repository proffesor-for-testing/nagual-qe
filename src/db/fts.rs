//! Full-Text Search (FTS5) support for SQLite.
//!
//! Provides FTS5 virtual tables for fast text search on patterns,
//! with ranking, tokenization configuration, and sync triggers.

use rusqlite::{Connection, ToSql};
use tracing::{debug, info};

use crate::error::{DatabaseError, NagualError, Result};

/// FTS5 configuration for a virtual table.
#[derive(Debug, Clone)]
pub struct Fts5Config {
    /// Name of the FTS5 virtual table.
    pub fts_table: String,

    /// Name of the source content table.
    pub content_table: String,

    /// Name of the rowid column in the content table.
    pub content_rowid: String,

    /// Columns to index for full-text search.
    pub indexed_columns: Vec<String>,

    /// Tokenizer to use (e.g., "porter ascii", "unicode61").
    pub tokenizer: Fts5Tokenizer,

    /// Prefix index sizes for faster prefix queries.
    pub prefix_sizes: Vec<u8>,
}

impl Fts5Config {
    /// Create a new FTS5 configuration.
    pub fn new(content_table: &str, indexed_columns: Vec<String>) -> Self {
        Self {
            fts_table: format!("{}_fts", content_table),
            content_table: content_table.to_string(),
            content_rowid: "rowid".to_string(),
            indexed_columns,
            tokenizer: Fts5Tokenizer::default(),
            prefix_sizes: vec![2, 3],
        }
    }

    /// Set the FTS table name.
    pub fn with_fts_table(mut self, name: &str) -> Self {
        self.fts_table = name.to_string();
        self
    }

    /// Set the content rowid column name.
    pub fn with_content_rowid(mut self, column: &str) -> Self {
        self.content_rowid = column.to_string();
        self
    }

    /// Set the tokenizer.
    pub fn with_tokenizer(mut self, tokenizer: Fts5Tokenizer) -> Self {
        self.tokenizer = tokenizer;
        self
    }

    /// Set prefix index sizes.
    pub fn with_prefixes(mut self, sizes: Vec<u8>) -> Self {
        self.prefix_sizes = sizes;
        self
    }
}

/// FTS5 tokenizer configuration.
#[derive(Debug, Clone)]
pub enum Fts5Tokenizer {
    /// Unicode tokenizer with optional case folding.
    Unicode61 { remove_diacritics: bool },

    /// Porter stemmer for English text (includes unicode61).
    Porter,

    /// ASCII tokenizer (faster but limited to ASCII).
    Ascii,

    /// Trigram tokenizer for substring matching.
    Trigram,

    /// Custom tokenizer string.
    Custom(String),
}

impl Default for Fts5Tokenizer {
    fn default() -> Self {
        Self::Porter
    }
}

impl Fts5Tokenizer {
    /// Get the tokenizer string for FTS5 configuration.
    fn as_str(&self) -> String {
        match self {
            Fts5Tokenizer::Unicode61 { remove_diacritics } => {
                if *remove_diacritics {
                    "unicode61 remove_diacritics 1".to_string()
                } else {
                    "unicode61".to_string()
                }
            }
            Fts5Tokenizer::Porter => "porter ascii".to_string(),
            Fts5Tokenizer::Ascii => "ascii".to_string(),
            Fts5Tokenizer::Trigram => "trigram".to_string(),
            Fts5Tokenizer::Custom(s) => s.clone(),
        }
    }
}

/// FTS5 search result with ranking information.
#[derive(Debug, Clone)]
pub struct FtsSearchResult<T> {
    /// The matched row data.
    pub data: T,

    /// BM25 relevance score (lower is better).
    pub rank: f64,

    /// Snippet of matching text with highlights.
    pub snippet: Option<String>,
}

/// Full-text search manager for patterns.
pub struct PatternFts {
    config: Fts5Config,
}

impl PatternFts {
    /// Create a new PatternFts with the default configuration.
    pub fn new() -> Self {
        Self {
            config: Fts5Config::new(
                "patterns",
                vec![
                    "problem".to_string(),
                    "solution".to_string(),
                    "domain".to_string(),
                ],
            ),
        }
    }

    /// Create a PatternFts with a custom configuration.
    pub fn with_config(config: Fts5Config) -> Self {
        Self { config }
    }

    /// Get the configuration.
    pub fn config(&self) -> &Fts5Config {
        &self.config
    }

    /// Create the FTS5 virtual table and sync triggers.
    pub fn create_fts_table(&self, conn: &Connection) -> Result<()> {
        let columns = self.config.indexed_columns.join(", ");
        let tokenizer = self.config.tokenizer.as_str();

        let prefix_str = if self.config.prefix_sizes.is_empty() {
            String::new()
        } else {
            format!(
                ", prefix='{}'",
                self.config
                    .prefix_sizes
                    .iter()
                    .map(|s| s.to_string())
                    .collect::<Vec<_>>()
                    .join(" ")
            )
        };

        // Create FTS5 virtual table
        let create_fts = format!(
            "CREATE VIRTUAL TABLE IF NOT EXISTS {} USING fts5(
                {},
                content='{}',
                content_rowid='{}',
                tokenize='{}'{}
            );",
            self.config.fts_table,
            columns,
            self.config.content_table,
            self.config.content_rowid,
            tokenizer,
            prefix_str
        );

        conn.execute_batch(&create_fts)
            .map_err(DatabaseError::from)?;

        // Create triggers to keep FTS in sync
        self.create_sync_triggers(conn)?;

        info!(
            fts_table = %self.config.fts_table,
            content_table = %self.config.content_table,
            columns = %columns,
            "Created FTS5 virtual table"
        );

        Ok(())
    }

    /// Create triggers to automatically sync the FTS index with the content table.
    fn create_sync_triggers(&self, conn: &Connection) -> Result<()> {
        let columns = self.config.indexed_columns.join(", ");
        let new_columns = self
            .config
            .indexed_columns
            .iter()
            .map(|c| format!("NEW.{}", c))
            .collect::<Vec<_>>()
            .join(", ");
        let old_columns = self
            .config
            .indexed_columns
            .iter()
            .map(|c| format!("OLD.{}", c))
            .collect::<Vec<_>>()
            .join(", ");

        // INSERT trigger
        let insert_trigger = format!(
            "CREATE TRIGGER IF NOT EXISTS {}_ai AFTER INSERT ON {} BEGIN
                INSERT INTO {}({}, rowid) VALUES ({}, NEW.{});
            END;",
            self.config.content_table,
            self.config.content_table,
            self.config.fts_table,
            columns,
            new_columns,
            self.config.content_rowid
        );

        // DELETE trigger
        let delete_trigger = format!(
            "CREATE TRIGGER IF NOT EXISTS {}_ad AFTER DELETE ON {} BEGIN
                INSERT INTO {}({}, rowid, {}) VALUES ('delete', OLD.{}, {});
            END;",
            self.config.content_table,
            self.config.content_table,
            self.config.fts_table,
            self.config.fts_table,
            columns,
            self.config.content_rowid,
            old_columns
        );

        // UPDATE trigger
        let update_trigger = format!(
            "CREATE TRIGGER IF NOT EXISTS {}_au AFTER UPDATE ON {} BEGIN
                INSERT INTO {}({}, rowid, {}) VALUES ('delete', OLD.{}, {});
                INSERT INTO {}({}, rowid) VALUES ({}, NEW.{});
            END;",
            self.config.content_table,
            self.config.content_table,
            self.config.fts_table,
            self.config.fts_table,
            columns,
            self.config.content_rowid,
            old_columns,
            self.config.fts_table,
            columns,
            new_columns,
            self.config.content_rowid
        );

        conn.execute_batch(&insert_trigger)
            .map_err(DatabaseError::from)?;
        conn.execute_batch(&delete_trigger)
            .map_err(DatabaseError::from)?;
        conn.execute_batch(&update_trigger)
            .map_err(DatabaseError::from)?;

        debug!(
            content_table = %self.config.content_table,
            "Created FTS sync triggers"
        );

        Ok(())
    }

    /// Rebuild the FTS index from the content table.
    pub fn rebuild_index(&self, conn: &Connection) -> Result<()> {
        let rebuild_sql = format!(
            "INSERT INTO {}({}) VALUES ('rebuild');",
            self.config.fts_table, self.config.fts_table
        );

        conn.execute_batch(&rebuild_sql)
            .map_err(DatabaseError::from)?;

        info!(
            fts_table = %self.config.fts_table,
            "Rebuilt FTS index"
        );

        Ok(())
    }

    /// Optimize the FTS index for better query performance.
    pub fn optimize_index(&self, conn: &Connection) -> Result<()> {
        let optimize_sql = format!(
            "INSERT INTO {}({}) VALUES ('optimize');",
            self.config.fts_table, self.config.fts_table
        );

        conn.execute_batch(&optimize_sql)
            .map_err(DatabaseError::from)?;

        debug!(
            fts_table = %self.config.fts_table,
            "Optimized FTS index"
        );

        Ok(())
    }

    /// Drop the FTS table and triggers.
    pub fn drop_fts_table(&self, conn: &Connection) -> Result<()> {
        let drop_sql = format!(
            "DROP TRIGGER IF EXISTS {}_ai;
             DROP TRIGGER IF EXISTS {}_ad;
             DROP TRIGGER IF EXISTS {}_au;
             DROP TABLE IF EXISTS {};",
            self.config.content_table,
            self.config.content_table,
            self.config.content_table,
            self.config.fts_table
        );

        conn.execute_batch(&drop_sql)
            .map_err(DatabaseError::from)?;

        info!(
            fts_table = %self.config.fts_table,
            "Dropped FTS table and triggers"
        );

        Ok(())
    }
}

impl Default for PatternFts {
    fn default() -> Self {
        Self::new()
    }
}

/// Options for FTS5 search queries.
#[derive(Debug, Clone)]
pub struct FtsSearchOptions {
    /// Maximum number of results to return.
    pub limit: usize,

    /// Offset for pagination.
    pub offset: usize,

    /// Generate snippets with highlighted matches.
    pub with_snippets: bool,

    /// Number of tokens before/after match in snippet.
    pub snippet_tokens: u32,

    /// Snippet highlight markers (start, end).
    pub snippet_markers: (String, String),

    /// Columns to search (empty = all indexed columns).
    pub search_columns: Vec<String>,

    /// Minimum BM25 rank threshold (lower is better match).
    pub min_rank: Option<f64>,
}

impl Default for FtsSearchOptions {
    fn default() -> Self {
        Self {
            limit: 100,
            offset: 0,
            with_snippets: false,
            snippet_tokens: 10,
            snippet_markers: ("<mark>".to_string(), "</mark>".to_string()),
            search_columns: Vec::new(),
            min_rank: None,
        }
    }
}

impl FtsSearchOptions {
    /// Set the limit.
    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = limit;
        self
    }

    /// Set the offset.
    pub fn with_offset(mut self, offset: usize) -> Self {
        self.offset = offset;
        self
    }

    /// Enable snippets.
    pub fn with_snippets(mut self) -> Self {
        self.with_snippets = true;
        self
    }

    /// Set specific columns to search.
    pub fn with_columns(mut self, columns: Vec<String>) -> Self {
        self.search_columns = columns;
        self
    }
}

/// Allowed column names for FTS5 search (prevents injection via search_columns).
const ALLOWED_FTS_COLUMNS: &[&str] = &[
    "problem", "solution", "domain", "context", "title", "content",
    "description", "tags", "category", "name", "text", "body", "summary",
];

/// Validate that all search columns are in the allowlist.
fn validate_search_columns(columns: &[String]) -> Result<()> {
    for col in columns {
        let col_lower = col.to_lowercase();
        if !ALLOWED_FTS_COLUMNS.contains(&col_lower.as_str()) {
            return Err(NagualError::config(format!(
                "Invalid FTS search column '{}'. Allowed columns: {:?}",
                col, ALLOWED_FTS_COLUMNS
            )));
        }
    }
    Ok(())
}

/// Perform an FTS5 search on a table.
///
/// Returns matching rowids with their BM25 ranks and optional snippets.
///
/// # Security
/// The `search_columns` in options are validated against an allowlist to prevent
/// SQL injection through column name manipulation.
pub fn fts_search(
    conn: &Connection,
    fts_table: &str,
    query: &str,
    options: &FtsSearchOptions,
) -> Result<Vec<(i64, f64, Option<String>)>> {
    // Validate search_columns against allowlist (security: prevents injection)
    if !options.search_columns.is_empty() {
        validate_search_columns(&options.search_columns)?;
    }

    // Build the query with optional column prefix
    let search_query = if options.search_columns.is_empty() {
        query.to_string()
    } else {
        // Search in specific columns: {col1 col2}: query
        format!(
            "{{{}}}: {}",
            options.search_columns.join(" "),
            query
        )
    };

    let sql = if options.with_snippets {
        format!(
            "SELECT rowid, rank, snippet({}, 0, ?, ?, '...', ?)
             FROM {} WHERE {} MATCH ?
             ORDER BY rank
             LIMIT ? OFFSET ?",
            fts_table, fts_table, fts_table
        )
    } else {
        format!(
            "SELECT rowid, rank, NULL
             FROM {} WHERE {} MATCH ?
             ORDER BY rank
             LIMIT ? OFFSET ?",
            fts_table, fts_table
        )
    };

    let mut stmt = conn.prepare(&sql).map_err(DatabaseError::from)?;

    let results: Vec<(i64, f64, Option<String>)> = if options.with_snippets {
        let rows = stmt
            .query_map(
                rusqlite::params![
                    &options.snippet_markers.0,
                    &options.snippet_markers.1,
                    options.snippet_tokens as i32,
                    &search_query,
                    options.limit as i64,
                    options.offset as i64,
                ],
                |row| {
                    Ok((
                        row.get::<_, i64>(0)?,
                        row.get::<_, f64>(1)?,
                        row.get::<_, Option<String>>(2)?,
                    ))
                },
            )
            .map_err(DatabaseError::from)?;

        rows.map(|r| r.map_err(DatabaseError::from))
            .collect::<std::result::Result<Vec<_>, _>>()?
    } else {
        let rows = stmt
            .query_map(
                rusqlite::params![
                    &search_query,
                    options.limit as i64,
                    options.offset as i64,
                ],
                |row| {
                    Ok((
                        row.get::<_, i64>(0)?,
                        row.get::<_, f64>(1)?,
                        row.get::<_, Option<String>>(2)?,
                    ))
                },
            )
            .map_err(DatabaseError::from)?;

        rows.map(|r| r.map_err(DatabaseError::from))
            .collect::<std::result::Result<Vec<_>, _>>()?
    };

    // Apply min rank filter if set
    let results = if let Some(min_rank) = options.min_rank {
        results.into_iter().filter(|(_, rank, _)| *rank <= min_rank).collect()
    } else {
        results
    };

    debug!(
        fts_table = %fts_table,
        query = %query,
        results = results.len(),
        "FTS search completed"
    );

    Ok(results)
}

/// Search patterns and return full pattern data with ranking.
pub fn fts_search_patterns<T, F>(
    conn: &Connection,
    content_table: &str,
    fts_table: &str,
    query: &str,
    options: &FtsSearchOptions,
    map_fn: F,
) -> Result<Vec<FtsSearchResult<T>>>
where
    F: Fn(&rusqlite::Row<'_>) -> rusqlite::Result<T>,
{
    // First, get the matching rowids and ranks
    let fts_results = fts_search(conn, fts_table, query, options)?;

    if fts_results.is_empty() {
        return Ok(Vec::new());
    }

    // Build placeholders for IN clause
    let placeholders: Vec<String> = (0..fts_results.len()).map(|_| "?".to_string()).collect();

    let sql = format!(
        "SELECT * FROM {} WHERE rowid IN ({})",
        content_table,
        placeholders.join(", ")
    );

    let rowids: Vec<i64> = fts_results.iter().map(|(id, _, _)| *id).collect();

    // Create a map from rowid to (rank, snippet)
    let rank_map: std::collections::HashMap<i64, (f64, Option<String>)> = fts_results
        .iter()
        .map(|(id, rank, snippet)| (*id, (*rank, snippet.clone())))
        .collect();

    let mut stmt = conn.prepare(&sql).map_err(DatabaseError::from)?;

    let params: Vec<&dyn ToSql> = rowids.iter().map(|id| id as &dyn ToSql).collect();

    let rows = stmt
        .query_map(params.as_slice(), |row| {
            // rowid should always be present in the query result, but default to 0 if missing
            // to avoid panicking on edge cases with virtual tables
            let rowid: i64 = row.get("rowid").unwrap_or(0);
            let data = map_fn(row)?;
            Ok((rowid, data))
        })
        .map_err(DatabaseError::from)?;

    let mut results: Vec<FtsSearchResult<T>> = Vec::new();
    for row in rows {
        let (rowid, data) = row.map_err(DatabaseError::from)?;
        if let Some((rank, snippet)) = rank_map.get(&rowid) {
            results.push(FtsSearchResult {
                data,
                rank: *rank,
                snippet: snippet.clone(),
            });
        }
    }

    // Sort by rank (lower is better in BM25)
    // Use Ordering::Equal as fallback for NaN comparisons (which shouldn't happen with valid BM25 scores)
    results.sort_by(|a, b| a.rank.partial_cmp(&b.rank).unwrap_or(std::cmp::Ordering::Equal));

    Ok(results)
}

/// Create the patterns table with all required columns for FTS indexing.
pub fn create_patterns_table(conn: &Connection) -> Result<()> {
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS patterns (
            rowid INTEGER PRIMARY KEY,
            id TEXT UNIQUE NOT NULL,
            problem TEXT NOT NULL,
            solution TEXT NOT NULL,
            domain TEXT NOT NULL,
            context TEXT,
            confidence REAL DEFAULT 0.0,
            usage_count INTEGER DEFAULT 0,
            success_rate REAL DEFAULT 0.0,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
            embedding BLOB
        );

        CREATE INDEX IF NOT EXISTS idx_patterns_domain ON patterns(domain);
        CREATE INDEX IF NOT EXISTS idx_patterns_confidence ON patterns(confidence);",
    )
    .map_err(DatabaseError::from)?;

    info!("Created patterns table");

    Ok(())
}

/// Initialize FTS5 for the patterns table.
pub fn init_patterns_fts(conn: &Connection) -> Result<()> {
    // Ensure patterns table exists
    create_patterns_table(conn)?;

    // Create FTS5 table
    let fts = PatternFts::new();
    fts.create_fts_table(conn)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rusqlite::params;

    fn setup_test_db() -> Connection {
        let conn = Connection::open_in_memory().unwrap();
        create_patterns_table(&conn).unwrap();
        let fts = PatternFts::new();
        fts.create_fts_table(&conn).unwrap();
        conn
    }

    fn insert_test_pattern(
        conn: &Connection,
        id: &str,
        problem: &str,
        solution: &str,
        domain: &str,
    ) {
        conn.execute(
            "INSERT INTO patterns (id, problem, solution, domain) VALUES (?, ?, ?, ?)",
            params![id, problem, solution, domain],
        )
        .unwrap();
    }

    #[test]
    fn test_create_fts_table() {
        let conn = Connection::open_in_memory().unwrap();
        create_patterns_table(&conn).unwrap();

        let fts = PatternFts::new();
        fts.create_fts_table(&conn).unwrap();

        // Verify FTS table exists
        let result: String = conn
            .query_row(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='patterns_fts'",
                [],
                |row| row.get(0),
            )
            .unwrap();

        assert_eq!(result, "patterns_fts");
    }

    #[test]
    fn test_fts_search_basic() {
        let conn = setup_test_db();

        // Insert test data
        insert_test_pattern(
            &conn,
            "p1",
            "Database connection timeout",
            "Increase pool size and timeout settings",
            "database",
        );
        insert_test_pattern(
            &conn,
            "p2",
            "Memory leak in application",
            "Profile and fix memory allocations",
            "performance",
        );
        insert_test_pattern(
            &conn,
            "p3",
            "Slow database queries",
            "Add indexes and optimize queries",
            "database",
        );

        // Search for "database"
        let results = fts_search(
            &conn,
            "patterns_fts",
            "database",
            &FtsSearchOptions::default(),
        )
        .unwrap();

        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_fts_search_with_snippets() {
        let conn = setup_test_db();

        insert_test_pattern(
            &conn,
            "p1",
            "How to handle database connection timeouts efficiently",
            "Use connection pooling with proper timeout settings",
            "infrastructure",
        );

        let options = FtsSearchOptions::default()
            .with_snippets()
            .with_limit(10);

        let results = fts_search(&conn, "patterns_fts", "timeout", &options).unwrap();

        assert_eq!(results.len(), 1);
        assert!(results[0].2.is_some());
    }

    #[test]
    fn test_fts_search_column_specific() {
        let conn = setup_test_db();

        insert_test_pattern(&conn, "p1", "API error handling", "Return proper error codes", "api");
        insert_test_pattern(&conn, "p2", "Fix login error", "Check credentials", "authentication");

        // Search only in "problem" column
        let options = FtsSearchOptions::default().with_columns(vec!["problem".to_string()]);

        let results = fts_search(&conn, "patterns_fts", "error", &options).unwrap();

        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_fts_trigger_insert() {
        let conn = setup_test_db();

        // Insert should automatically update FTS
        insert_test_pattern(&conn, "p1", "Test problem", "Test solution", "test");

        let results = fts_search(
            &conn,
            "patterns_fts",
            "problem",
            &FtsSearchOptions::default(),
        )
        .unwrap();

        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_fts_trigger_update() {
        let conn = setup_test_db();

        // Use unique text that only appears in the problem field
        insert_test_pattern(&conn, "p1", "Zebra unicorn problem", "Generic solution text", "test");

        // Verify original text is indexed
        let results = fts_search(
            &conn,
            "patterns_fts",
            "zebra unicorn",
            &FtsSearchOptions::default(),
        )
        .unwrap();
        assert_eq!(results.len(), 1, "Original text should be indexed");

        // Update the problem
        conn.execute(
            "UPDATE patterns SET problem = 'Elephant giraffe problem' WHERE id = 'p1'",
            [],
        )
        .unwrap();

        // Original text should not match after update
        let results = fts_search(
            &conn,
            "patterns_fts",
            "zebra unicorn",
            &FtsSearchOptions::default(),
        )
        .unwrap();
        assert_eq!(results.len(), 0, "Original text should be removed from index");

        // Updated text should match
        let results = fts_search(
            &conn,
            "patterns_fts",
            "elephant giraffe",
            &FtsSearchOptions::default(),
        )
        .unwrap();
        assert_eq!(results.len(), 1, "Updated text should be indexed");
    }

    #[test]
    fn test_fts_trigger_delete() {
        let conn = setup_test_db();

        insert_test_pattern(&conn, "p1", "Delete me problem", "Delete me solution", "test");

        // Verify it's indexed
        let results = fts_search(&conn, "patterns_fts", "Delete", &FtsSearchOptions::default())
            .unwrap();
        assert_eq!(results.len(), 1);

        // Delete the row
        conn.execute("DELETE FROM patterns WHERE id = 'p1'", [])
            .unwrap();

        // Should no longer be found
        let results = fts_search(&conn, "patterns_fts", "Delete", &FtsSearchOptions::default())
            .unwrap();
        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_rebuild_index() {
        let conn = setup_test_db();

        insert_test_pattern(&conn, "p1", "Test pattern", "Test solution", "test");

        let fts = PatternFts::new();
        fts.rebuild_index(&conn).unwrap();

        // Verify search still works
        let results = fts_search(&conn, "patterns_fts", "pattern", &FtsSearchOptions::default())
            .unwrap();
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_optimize_index() {
        let conn = setup_test_db();

        insert_test_pattern(&conn, "p1", "Test pattern", "Test solution", "test");

        let fts = PatternFts::new();
        fts.optimize_index(&conn).unwrap();

        // Verify search still works
        let results = fts_search(&conn, "patterns_fts", "pattern", &FtsSearchOptions::default())
            .unwrap();
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_drop_fts_table() {
        let conn = setup_test_db();

        let fts = PatternFts::new();
        fts.drop_fts_table(&conn).unwrap();

        // Verify table is gone
        let result: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='patterns_fts'",
                [],
                |row| row.get(0),
            )
            .unwrap();

        assert_eq!(result, 0);
    }

    #[test]
    fn test_fts_search_patterns() {
        let conn = setup_test_db();

        insert_test_pattern(
            &conn,
            "p1",
            "API rate limiting",
            "Implement token bucket algorithm",
            "api",
        );
        insert_test_pattern(
            &conn,
            "p2",
            "Database deadlock",
            "Use proper locking order",
            "database",
        );

        let results = fts_search_patterns(
            &conn,
            "patterns",
            "patterns_fts",
            "API",
            &FtsSearchOptions::default(),
            |row| {
                Ok((
                    row.get::<_, String>("id")?,
                    row.get::<_, String>("problem")?,
                    row.get::<_, String>("solution")?,
                ))
            },
        )
        .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].data.0, "p1");
        assert!(results[0].rank < 0.0); // BM25 returns negative values for matches
    }

    #[test]
    fn test_fts_porter_stemming() {
        let conn = setup_test_db();

        insert_test_pattern(
            &conn,
            "p1",
            "Running tests continuously",
            "Use CI/CD pipeline",
            "testing",
        );

        // Porter stemmer should match "run" to "running"
        let results = fts_search(&conn, "patterns_fts", "run", &FtsSearchOptions::default())
            .unwrap();

        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_fts_phrase_search() {
        let conn = setup_test_db();

        insert_test_pattern(
            &conn,
            "p1",
            "Handle connection timeout gracefully",
            "Implement retry logic",
            "networking",
        );
        insert_test_pattern(
            &conn,
            "p2",
            "Timeout on connection pool",
            "Increase pool size",
            "database",
        );

        // Phrase search with quotes
        let results = fts_search(
            &conn,
            "patterns_fts",
            "\"connection timeout\"",
            &FtsSearchOptions::default(),
        )
        .unwrap();

        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_fts_boolean_operators() {
        let conn = setup_test_db();

        insert_test_pattern(&conn, "p1", "Database performance", "Add indexes", "database");
        insert_test_pattern(&conn, "p2", "API performance", "Use caching", "api");
        insert_test_pattern(&conn, "p3", "Database security", "Encrypt data", "database");

        // AND operator (implicit in FTS5)
        let results = fts_search(
            &conn,
            "patterns_fts",
            "database performance",
            &FtsSearchOptions::default(),
        )
        .unwrap();

        assert_eq!(results.len(), 1);

        // OR operator
        let results = fts_search(
            &conn,
            "patterns_fts",
            "performance OR security",
            &FtsSearchOptions::default(),
        )
        .unwrap();

        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_fts_search_column_validation() {
        let conn = setup_test_db();

        insert_test_pattern(&conn, "p1", "Test problem", "Test solution", "test");

        // Valid columns should work
        let options = FtsSearchOptions::default().with_columns(vec!["problem".to_string()]);
        let results = fts_search(&conn, "patterns_fts", "test", &options);
        assert!(results.is_ok());

        // Invalid column should be rejected (security: prevents injection)
        let invalid_options =
            FtsSearchOptions::default().with_columns(vec!["malicious; DROP TABLE".to_string()]);
        let invalid_results = fts_search(&conn, "patterns_fts", "test", &invalid_options);
        assert!(invalid_results.is_err());

        // Case-insensitive validation should work
        let upper_options = FtsSearchOptions::default().with_columns(vec!["PROBLEM".to_string()]);
        let upper_results = fts_search(&conn, "patterns_fts", "test", &upper_options);
        assert!(upper_results.is_ok());
    }
}
