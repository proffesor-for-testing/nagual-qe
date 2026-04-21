//! Gene Transfusion - Extract reusable patterns from existing codebases.
//!
//! Inspired by StrongDM's approach to pattern extraction, this module scans
//! source code files and detects common patterns (error handling, async patterns,
//! API design, etc.) that can be stored in the ReasoningBank for future reuse.
//!
//! # Example
//!
//! ```ignore
//! use nagual::reasoning_bank::transfusion::{Transfuser, TransfusionConfig};
//! use std::path::Path;
//!
//! let config = TransfusionConfig::default();
//! let transfuser = Transfuser::new(config);
//! let result = transfuser.transfuse(Path::new("./src"))?;
//!
//! println!("Extracted {} patterns from {} files", result.patterns_extracted, result.files_scanned);
//! ```

use crate::reasoning_bank::pattern::{Pattern, PatternCategory, PatternMetadata};
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use walkdir::WalkDir;

/// A pattern extracted from source code.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedPattern {
    /// The problem this pattern solves
    pub problem: String,
    /// The solution/implementation
    pub solution: String,
    /// Domain classification
    pub domain: String,
    /// Tags for searchability
    pub tags: Vec<String>,
    /// Source file path
    pub source_file: String,
    /// Line number where pattern was found
    pub line_number: usize,
    /// Confidence score (0.0-1.0)
    pub confidence: f32,
    /// The detector that found this pattern
    pub detector_name: String,
}

impl ExtractedPattern {
    /// Convert to a Pattern for storage.
    pub fn to_pattern(&self) -> Pattern {
        Pattern::builder()
            .problem(&self.problem)
            .solution(&self.solution)
            .category(PatternCategory::from(self.domain.as_str()))
            .tags(self.tags.clone())
            .confidence(self.confidence)
            .effectiveness(0.5) // Default until validated
            .metadata(
                PatternMetadata::new()
                    .with_source("transfusion")
                    .with_extra("source_file", serde_json::json!(self.source_file))
                    .with_extra("line_number", serde_json::json!(self.line_number))
                    .with_extra("detector", serde_json::json!(self.detector_name)),
            )
            .build()
    }
}

/// Result of a transfusion operation.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TransfusionResult {
    /// Number of files scanned
    pub files_scanned: usize,
    /// Number of patterns extracted
    pub patterns_extracted: usize,
    /// Number of patterns stored (if not dry_run)
    pub patterns_stored: usize,
    /// Number of patterns skipped (low confidence or duplicates)
    pub patterns_skipped: usize,
    /// Patterns by category
    pub by_category: HashMap<String, usize>,
    /// Patterns by detector
    pub by_detector: HashMap<String, usize>,
    /// The extracted patterns themselves
    pub patterns: Vec<ExtractedPattern>,
    /// Errors encountered (file path -> error message)
    pub errors: HashMap<String, String>,
}

/// Configuration for transfusion operations.
#[derive(Debug, Clone)]
pub struct TransfusionConfig {
    /// Minimum confidence threshold (0.0-1.0)
    pub min_confidence: f32,
    /// File extensions to include
    pub include_extensions: Vec<String>,
    /// Directories to exclude
    pub exclude_dirs: Vec<String>,
    /// Patterns to exclude (regex)
    pub exclude_patterns: Vec<String>,
    /// Whether to perform a dry run (don't store patterns)
    pub dry_run: bool,
    /// Maximum file size to process (in bytes)
    pub max_file_size: usize,
    /// Maximum files to process (0 = unlimited)
    pub max_files: usize,
}

impl Default for TransfusionConfig {
    fn default() -> Self {
        Self {
            min_confidence: 0.7,
            include_extensions: vec![
                "rs".into(),
                "py".into(),
                "ts".into(),
                "tsx".into(),
                "js".into(),
                "jsx".into(),
                "go".into(),
                "java".into(),
                "kt".into(),
                "swift".into(),
                "rb".into(),
                "ex".into(),
                "exs".into(),
            ],
            exclude_dirs: vec![
                "target".into(),
                "node_modules".into(),
                ".git".into(),
                "vendor".into(),
                "dist".into(),
                "build".into(),
                "__pycache__".into(),
                ".venv".into(),
                "venv".into(),
                ".cargo".into(),
            ],
            exclude_patterns: vec![
                r"\.min\.js$".into(),
                r"\.bundle\.js$".into(),
                r"\.d\.ts$".into(),
                r"_test\.go$".into(), // Go test files are often noise
            ],
            dry_run: false,
            max_file_size: 1024 * 1024, // 1MB
            max_files: 0,               // unlimited
        }
    }
}

/// Trait for detecting patterns in source code.
pub trait PatternDetector: Send + Sync {
    /// Name of this detector.
    fn name(&self) -> &str;

    /// Detect patterns in the given file content.
    fn detect(&self, content: &str, file_path: &str) -> Vec<ExtractedPattern>;

    /// File extensions this detector applies to.
    fn supported_extensions(&self) -> &[&str] {
        &[]
    }
}

/// Detects error handling patterns in Rust code.
pub struct RustErrorHandlingDetector {
    custom_error_regex: Regex,
    thiserror_regex: Regex,
    map_err_regex: Regex,
    context_regex: Regex,
    result_fn_regex: Regex,
}

impl Default for RustErrorHandlingDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl RustErrorHandlingDetector {
    /// Create a new Rust error handling detector.
    pub fn new() -> Self {
        Self {
            custom_error_regex: Regex::new(
                r"(?m)impl\s+(?:std::)?(?:error::)?Error\s+for\s+(\w+)"
            ).unwrap(),
            thiserror_regex: Regex::new(
                r"(?m)#\[derive\([^\)]*Error[^\)]*\)\]"
            ).unwrap(),
            map_err_regex: Regex::new(
                r"(?m)\.map_err\(\|[^|]*\|\s*([^)]+)\)"
            ).unwrap(),
            context_regex: Regex::new(
                r"(?m)\.context\(([^)]+)\)"
            ).unwrap(),
            result_fn_regex: Regex::new(
                r"(?m)pub\s+(?:async\s+)?fn\s+(\w+)[^)]*\)\s*->\s*(?:anyhow::)?Result<"
            ).unwrap(),
        }
    }

    fn extract_impl_block(&self, content: &str, start_line: usize) -> String {
        let lines: Vec<&str> = content.lines().collect();
        let mut brace_count = 0;
        let mut in_impl = false;
        let mut result = Vec::new();

        for (i, line) in lines.iter().enumerate().skip(start_line) {
            if line.contains("impl") {
                in_impl = true;
            }

            if in_impl {
                result.push(*line);
                brace_count += line.chars().filter(|c| *c == '{').count();
                brace_count = brace_count.saturating_sub(line.chars().filter(|c| *c == '}').count());

                if brace_count == 0 && result.len() > 1 {
                    break;
                }
            }

            // Limit to 50 lines max
            if i - start_line > 50 {
                break;
            }
        }

        result.join("\n")
    }
}

impl PatternDetector for RustErrorHandlingDetector {
    fn name(&self) -> &str {
        "rust_error_handling"
    }

    fn supported_extensions(&self) -> &[&str] {
        &["rs"]
    }

    fn detect(&self, content: &str, file_path: &str) -> Vec<ExtractedPattern> {
        let mut patterns = Vec::new();
        let lines: Vec<&str> = content.lines().collect();

        // Detect custom Error impl
        for caps in self.custom_error_regex.captures_iter(content) {
            let error_type = caps.get(1).map(|m| m.as_str()).unwrap_or("CustomError");
            let line_num = content[..caps.get(0).unwrap().start()]
                .lines()
                .count();
            let impl_block = self.extract_impl_block(content, line_num.saturating_sub(1));

            if impl_block.contains("fn source") || impl_block.contains("fn description") {
                patterns.push(ExtractedPattern {
                    problem: format!(
                        "Implementing custom error type '{}' with proper Error trait",
                        error_type
                    ),
                    solution: impl_block,
                    domain: "rust.error_handling".into(),
                    tags: vec!["error".into(), "rust".into(), "impl".into(), "trait".into()],
                    source_file: file_path.into(),
                    line_number: line_num,
                    confidence: 0.85,
                    detector_name: self.name().into(),
                });
            }
        }

        // Detect thiserror usage
        for m in self.thiserror_regex.find_iter(content) {
            let line_num = content[..m.start()].lines().count();

            // Get the enum definition that follows
            let enum_def: String = lines
                .iter()
                .skip(line_num)
                .take(30)
                .cloned()
                .collect::<Vec<_>>()
                .join("\n");

            if enum_def.contains("enum") && enum_def.contains("Error") {
                patterns.push(ExtractedPattern {
                    problem: "Defining error types with thiserror derive macro".into(),
                    solution: enum_def,
                    domain: "rust.error_handling".into(),
                    tags: vec![
                        "error".into(),
                        "rust".into(),
                        "thiserror".into(),
                        "derive".into(),
                    ],
                    source_file: file_path.into(),
                    line_number: line_num,
                    confidence: 0.9,
                    detector_name: self.name().into(),
                });
            }
        }

        // Detect map_err patterns
        if content.contains(".map_err(") && content.contains("Result<") {
            let map_err_count = self.map_err_regex.find_iter(content).count();
            if map_err_count >= 2 {
                // Extract a representative example
                if let Some(caps) = self.map_err_regex.captures(content) {
                    let line_num = content[..caps.get(0).unwrap().start()]
                        .lines()
                        .count();

                    // Get surrounding context (5 lines before, 5 after)
                    let start = line_num.saturating_sub(5);
                    let end = (line_num + 5).min(lines.len());
                    let context: String = lines[start..end].join("\n");

                    patterns.push(ExtractedPattern {
                        problem: "Error mapping and transformation in Rust".into(),
                        solution: context,
                        domain: "rust.error_handling".into(),
                        tags: vec![
                            "error".into(),
                            "rust".into(),
                            "map_err".into(),
                            "result".into(),
                        ],
                        source_file: file_path.into(),
                        line_number: line_num,
                        confidence: 0.75,
                        detector_name: self.name().into(),
                    });
                }
            }
        }

        patterns
    }
}

/// Detects async/concurrency patterns in Rust code.
pub struct RustAsyncDetector {
    tokio_spawn_regex: Regex,
    async_fn_regex: Regex,
    select_regex: Regex,
    channel_regex: Regex,
    mutex_regex: Regex,
    rwlock_regex: Regex,
}

impl Default for RustAsyncDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl RustAsyncDetector {
    /// Create a new Rust async detector.
    pub fn new() -> Self {
        Self {
            tokio_spawn_regex: Regex::new(r"(?m)tokio::spawn\s*\(").unwrap(),
            async_fn_regex: Regex::new(r"(?m)pub\s+async\s+fn\s+(\w+)").unwrap(),
            select_regex: Regex::new(r"(?m)tokio::select!\s*\{").unwrap(),
            channel_regex: Regex::new(r"(?m)(mpsc|oneshot|broadcast|watch)::channel").unwrap(),
            mutex_regex: Regex::new(r"(?m)(tokio::sync::)?Mutex::new").unwrap(),
            rwlock_regex: Regex::new(r"(?m)(tokio::sync::)?RwLock::new").unwrap(),
        }
    }

    fn extract_function(&self, content: &str, start_line: usize) -> String {
        let lines: Vec<&str> = content.lines().collect();
        let mut brace_count = 0;
        let mut result = Vec::new();
        let mut started = false;

        for line in lines.iter().skip(start_line) {
            result.push(*line);

            if line.contains('{') {
                started = true;
            }

            if started {
                brace_count += line.chars().filter(|c| *c == '{').count();
                brace_count = brace_count.saturating_sub(line.chars().filter(|c| *c == '}').count());

                if brace_count == 0 {
                    break;
                }
            }

            // Limit to 80 lines max
            if result.len() > 80 {
                result.push("    // ... (truncated)");
                break;
            }
        }

        result.join("\n")
    }
}

impl PatternDetector for RustAsyncDetector {
    fn name(&self) -> &str {
        "rust_async"
    }

    fn supported_extensions(&self) -> &[&str] {
        &["rs"]
    }

    fn detect(&self, content: &str, file_path: &str) -> Vec<ExtractedPattern> {
        let mut patterns = Vec::new();
        let lines: Vec<&str> = content.lines().collect();

        // Detect tokio::spawn patterns
        for m in self.tokio_spawn_regex.find_iter(content) {
            let line_num = content[..m.start()].lines().count();
            let start = line_num.saturating_sub(2);
            let end = (line_num + 15).min(lines.len());
            let context: String = lines[start..end].join("\n");

            patterns.push(ExtractedPattern {
                problem: "Spawning async tasks with Tokio".into(),
                solution: context,
                domain: "rust.async".into(),
                tags: vec![
                    "async".into(),
                    "tokio".into(),
                    "spawn".into(),
                    "concurrency".into(),
                ],
                source_file: file_path.into(),
                line_number: line_num,
                confidence: 0.8,
                detector_name: self.name().into(),
            });
        }

        // Detect tokio::select! patterns
        for m in self.select_regex.find_iter(content) {
            let line_num = content[..m.start()].lines().count();
            let start = line_num.saturating_sub(2);
            let end = (line_num + 25).min(lines.len());
            let context: String = lines[start..end].join("\n");

            patterns.push(ExtractedPattern {
                problem: "Using tokio::select! for concurrent operations".into(),
                solution: context,
                domain: "rust.async".into(),
                tags: vec![
                    "async".into(),
                    "tokio".into(),
                    "select".into(),
                    "concurrency".into(),
                ],
                source_file: file_path.into(),
                line_number: line_num,
                confidence: 0.85,
                detector_name: self.name().into(),
            });
        }

        // Detect channel patterns
        for caps in self.channel_regex.captures_iter(content) {
            let channel_type = caps.get(1).map(|m| m.as_str()).unwrap_or("channel");
            let line_num = content[..caps.get(0).unwrap().start()]
                .lines()
                .count();
            let start = line_num.saturating_sub(2);
            let end = (line_num + 20).min(lines.len());
            let context: String = lines[start..end].join("\n");

            patterns.push(ExtractedPattern {
                problem: format!(
                    "Using Tokio {} channel for async message passing",
                    channel_type
                ),
                solution: context,
                domain: "rust.async".into(),
                tags: vec![
                    "async".into(),
                    "tokio".into(),
                    "channel".into(),
                    channel_type.into(),
                ],
                source_file: file_path.into(),
                line_number: line_num,
                confidence: 0.8,
                detector_name: self.name().into(),
            });
        }

        // Detect mutex/rwlock patterns
        if self.mutex_regex.is_match(content) || self.rwlock_regex.is_match(content) {
            // Find a good example
            let mutex_match = self.mutex_regex.find(content);
            let rwlock_match = self.rwlock_regex.find(content);

            if let Some(m) = mutex_match.or(rwlock_match) {
                let line_num = content[..m.start()].lines().count();
                let start = line_num.saturating_sub(3);
                let end = (line_num + 15).min(lines.len());
                let context: String = lines[start..end].join("\n");

                let lock_type = if self.mutex_regex.is_match(&context) {
                    "Mutex"
                } else {
                    "RwLock"
                };

                patterns.push(ExtractedPattern {
                    problem: format!("Using {} for thread-safe shared state", lock_type),
                    solution: context,
                    domain: "rust.async".into(),
                    tags: vec![
                        "async".into(),
                        "sync".into(),
                        lock_type.to_lowercase(),
                        "concurrency".into(),
                    ],
                    source_file: file_path.into(),
                    line_number: line_num,
                    confidence: 0.75,
                    detector_name: self.name().into(),
                });
            }
        }

        patterns
    }
}

/// Detects API design patterns (routes, handlers, middleware).
pub struct ApiPatternDetector {
    axum_route_regex: Regex,
    actix_route_regex: Regex,
    middleware_regex: Regex,
    handler_regex: Regex,
}

impl Default for ApiPatternDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl ApiPatternDetector {
    /// Create a new API pattern detector.
    pub fn new() -> Self {
        Self {
            axum_route_regex: Regex::new(
                r#"(?m)\.route\s*\(\s*["']([^"']+)["']"#
            ).unwrap(),
            actix_route_regex: Regex::new(
                r#"(?m)#\[(?:get|post|put|delete|patch)\s*\(\s*["']([^"']+)["']"#
            ).unwrap(),
            middleware_regex: Regex::new(
                r"(?m)\.layer\s*\(\s*(\w+)"
            ).unwrap(),
            handler_regex: Regex::new(
                r"(?m)async\s+fn\s+(\w+)\s*\([^)]*(?:State|Json|Path|Query|Extension)"
            ).unwrap(),
        }
    }

    fn extract_router_block(&self, content: &str, start_line: usize) -> String {
        let lines: Vec<&str> = content.lines().collect();
        let mut result = Vec::new();
        let mut paren_count = 0;
        let mut started = false;

        for line in lines.iter().skip(start_line.saturating_sub(3)) {
            result.push(*line);

            if line.contains("Router::") || line.contains(".route") {
                started = true;
            }

            if started {
                paren_count += line.chars().filter(|c| *c == '(').count();
                paren_count = paren_count.saturating_sub(line.chars().filter(|c| *c == ')').count());

                // Look for semicolon at end to stop
                if line.trim().ends_with(';') && paren_count == 0 {
                    break;
                }
            }

            if result.len() > 30 {
                break;
            }
        }

        result.join("\n")
    }
}

impl PatternDetector for ApiPatternDetector {
    fn name(&self) -> &str {
        "api_design"
    }

    fn supported_extensions(&self) -> &[&str] {
        &["rs"]
    }

    fn detect(&self, content: &str, file_path: &str) -> Vec<ExtractedPattern> {
        let mut patterns = Vec::new();
        let lines: Vec<&str> = content.lines().collect();

        // Detect Axum route definitions
        if content.contains("axum::") || content.contains("use axum") {
            for m in self.axum_route_regex.find_iter(content) {
                let line_num = content[..m.start()].lines().count();
                let router_block = self.extract_router_block(content, line_num);

                patterns.push(ExtractedPattern {
                    problem: "Defining API routes with Axum router".into(),
                    solution: router_block,
                    domain: "rust.api".into(),
                    tags: vec![
                        "api".into(),
                        "axum".into(),
                        "router".into(),
                        "web".into(),
                    ],
                    source_file: file_path.into(),
                    line_number: line_num,
                    confidence: 0.85,
                    detector_name: self.name().into(),
                });

                // Only capture one router block per file
                break;
            }
        }

        // Detect Actix-web route attributes
        for caps in self.actix_route_regex.captures_iter(content) {
            let route_path = caps.get(1).map(|m| m.as_str()).unwrap_or("/");
            let line_num = content[..caps.get(0).unwrap().start()]
                .lines()
                .count();
            let start = line_num.saturating_sub(1);
            let end = (line_num + 20).min(lines.len());
            let context: String = lines[start..end].join("\n");

            patterns.push(ExtractedPattern {
                problem: format!("Actix-web route handler for '{}'", route_path),
                solution: context,
                domain: "rust.api".into(),
                tags: vec![
                    "api".into(),
                    "actix".into(),
                    "handler".into(),
                    "web".into(),
                ],
                source_file: file_path.into(),
                line_number: line_num,
                confidence: 0.85,
                detector_name: self.name().into(),
            });

            // Limit to 3 per file
            if patterns.len() >= 3 {
                break;
            }
        }

        // Detect middleware layer usage
        for m in self.middleware_regex.find_iter(content) {
            let line_num = content[..m.start()].lines().count();
            let start = line_num.saturating_sub(2);
            let end = (line_num + 10).min(lines.len());
            let context: String = lines[start..end].join("\n");

            patterns.push(ExtractedPattern {
                problem: "Adding middleware layers to API router".into(),
                solution: context,
                domain: "rust.api".into(),
                tags: vec![
                    "api".into(),
                    "middleware".into(),
                    "layer".into(),
                    "web".into(),
                ],
                source_file: file_path.into(),
                line_number: line_num,
                confidence: 0.75,
                detector_name: self.name().into(),
            });

            // Only one middleware pattern per file
            break;
        }

        patterns
    }
}

/// Detects testing patterns.
pub struct TestPatternDetector {
    test_fn_regex: Regex,
    async_test_regex: Regex,
    mock_regex: Regex,
    proptest_regex: Regex,
    fixture_regex: Regex,
}

impl Default for TestPatternDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl TestPatternDetector {
    /// Create a new test pattern detector.
    pub fn new() -> Self {
        Self {
            test_fn_regex: Regex::new(r"(?m)#\[test\]").unwrap(),
            async_test_regex: Regex::new(r"(?m)#\[tokio::test\]").unwrap(),
            mock_regex: Regex::new(r"(?m)(?:mock!|Mock\w+::new|mockall)").unwrap(),
            proptest_regex: Regex::new(r"(?m)proptest!\s*\{").unwrap(),
            fixture_regex: Regex::new(r"(?m)fn\s+setup|fn\s+teardown|#\[fixture\]").unwrap(),
        }
    }

    fn extract_test_function(&self, content: &str, start_line: usize) -> String {
        let lines: Vec<&str> = content.lines().collect();
        let mut brace_count = 0;
        let mut result = Vec::new();
        let mut started = false;

        for line in lines.iter().skip(start_line) {
            result.push(*line);

            if line.contains("fn ") {
                started = true;
            }

            if started && line.contains('{') {
                brace_count += line.chars().filter(|c| *c == '{').count();
            }

            if started && line.contains('}') {
                brace_count = brace_count.saturating_sub(line.chars().filter(|c| *c == '}').count());
                if brace_count == 0 {
                    break;
                }
            }

            if result.len() > 50 {
                result.push("    // ... (truncated)");
                break;
            }
        }

        result.join("\n")
    }
}

impl PatternDetector for TestPatternDetector {
    fn name(&self) -> &str {
        "testing"
    }

    fn supported_extensions(&self) -> &[&str] {
        &["rs"]
    }

    fn detect(&self, content: &str, file_path: &str) -> Vec<ExtractedPattern> {
        let mut patterns = Vec::new();

        // Detect async test patterns
        for m in self.async_test_regex.find_iter(content) {
            let line_num = content[..m.start()].lines().count();
            let test_fn = self.extract_test_function(content, line_num);

            patterns.push(ExtractedPattern {
                problem: "Writing async tests with tokio::test".into(),
                solution: test_fn,
                domain: "rust.testing".into(),
                tags: vec![
                    "testing".into(),
                    "async".into(),
                    "tokio".into(),
                    "unit-test".into(),
                ],
                source_file: file_path.into(),
                line_number: line_num,
                confidence: 0.8,
                detector_name: self.name().into(),
            });

            // Limit to 2 per file
            if patterns.len() >= 2 {
                break;
            }
        }

        // Detect mock patterns
        if self.mock_regex.is_match(content) {
            if let Some(m) = self.mock_regex.find(content) {
                let line_num = content[..m.start()].lines().count();
                let lines: Vec<&str> = content.lines().collect();
                let start = line_num.saturating_sub(3);
                let end = (line_num + 20).min(lines.len());
                let context: String = lines[start..end].join("\n");

                patterns.push(ExtractedPattern {
                    problem: "Using mocks for test isolation".into(),
                    solution: context,
                    domain: "rust.testing".into(),
                    tags: vec!["testing".into(), "mock".into(), "isolation".into()],
                    source_file: file_path.into(),
                    line_number: line_num,
                    confidence: 0.8,
                    detector_name: self.name().into(),
                });
            }
        }

        // Detect property-based testing (proptest)
        for m in self.proptest_regex.find_iter(content) {
            let line_num = content[..m.start()].lines().count();
            let lines: Vec<&str> = content.lines().collect();
            let start = line_num.saturating_sub(2);
            let end = (line_num + 25).min(lines.len());
            let context: String = lines[start..end].join("\n");

            patterns.push(ExtractedPattern {
                problem: "Property-based testing with proptest".into(),
                solution: context,
                domain: "rust.testing".into(),
                tags: vec![
                    "testing".into(),
                    "proptest".into(),
                    "property-based".into(),
                    "fuzzing".into(),
                ],
                source_file: file_path.into(),
                line_number: line_num,
                confidence: 0.9,
                detector_name: self.name().into(),
            });

            break;
        }

        patterns
    }
}

/// Detects database/ORM patterns.
pub struct DatabasePatternDetector {
    sqlx_query_regex: Regex,
    transaction_regex: Regex,
    migration_regex: Regex,
    pool_regex: Regex,
}

impl Default for DatabasePatternDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl DatabasePatternDetector {
    /// Create a new database pattern detector.
    pub fn new() -> Self {
        Self {
            sqlx_query_regex: Regex::new(
                r"(?m)sqlx::query(?:_as|_scalar)?\s*(?:!\s*)?\("
            ).unwrap(),
            transaction_regex: Regex::new(
                r"(?m)\.begin\s*\(\s*\)|transaction|\.commit\s*\(\s*\)"
            ).unwrap(),
            migration_regex: Regex::new(
                r"(?m)(?:migration|migrate|MIGRATION|CREATE\s+TABLE|ALTER\s+TABLE)"
            ).unwrap(),
            pool_regex: Regex::new(
                r"(?m)(?:Pool|PgPool|SqlitePool|MySqlPool)::connect"
            ).unwrap(),
        }
    }
}

impl PatternDetector for DatabasePatternDetector {
    fn name(&self) -> &str {
        "database"
    }

    fn supported_extensions(&self) -> &[&str] {
        &["rs"]
    }

    fn detect(&self, content: &str, file_path: &str) -> Vec<ExtractedPattern> {
        let mut patterns = Vec::new();
        let lines: Vec<&str> = content.lines().collect();

        // Detect SQLx query patterns
        for m in self.sqlx_query_regex.find_iter(content) {
            let line_num = content[..m.start()].lines().count();
            let start = line_num.saturating_sub(2);
            let end = (line_num + 15).min(lines.len());
            let context: String = lines[start..end].join("\n");

            patterns.push(ExtractedPattern {
                problem: "Executing SQL queries with SQLx".into(),
                solution: context,
                domain: "rust.database".into(),
                tags: vec![
                    "database".into(),
                    "sqlx".into(),
                    "query".into(),
                    "sql".into(),
                ],
                source_file: file_path.into(),
                line_number: line_num,
                confidence: 0.8,
                detector_name: self.name().into(),
            });

            // Limit to 2 per file
            if patterns.len() >= 2 {
                break;
            }
        }

        // Detect transaction patterns
        if self.transaction_regex.is_match(content) && content.contains(".begin(") {
            if let Some(m) = content.find(".begin(") {
                let line_num = content[..m].lines().count();
                let start = line_num.saturating_sub(2);
                let end = (line_num + 20).min(lines.len());
                let context: String = lines[start..end].join("\n");

                patterns.push(ExtractedPattern {
                    problem: "Managing database transactions".into(),
                    solution: context,
                    domain: "rust.database".into(),
                    tags: vec![
                        "database".into(),
                        "transaction".into(),
                        "sqlx".into(),
                        "acid".into(),
                    ],
                    source_file: file_path.into(),
                    line_number: line_num,
                    confidence: 0.85,
                    detector_name: self.name().into(),
                });
            }
        }

        // Detect connection pool patterns
        if let Some(m) = self.pool_regex.find(content) {
            let line_num = content[..m.start()].lines().count();
            let start = line_num.saturating_sub(3);
            let end = (line_num + 15).min(lines.len());
            let context: String = lines[start..end].join("\n");

            patterns.push(ExtractedPattern {
                problem: "Setting up database connection pooling".into(),
                solution: context,
                domain: "rust.database".into(),
                tags: vec![
                    "database".into(),
                    "pool".into(),
                    "connection".into(),
                    "performance".into(),
                ],
                source_file: file_path.into(),
                line_number: line_num,
                confidence: 0.8,
                detector_name: self.name().into(),
            });
        }

        patterns
    }
}

/// Main transfusion engine.
pub struct Transfuser {
    detectors: Vec<Box<dyn PatternDetector>>,
    config: TransfusionConfig,
    exclude_regex: Vec<Regex>,
}

impl Transfuser {
    /// Create a new transfuser with default detectors.
    pub fn new(config: TransfusionConfig) -> Self {
        let exclude_regex: Vec<Regex> = config
            .exclude_patterns
            .iter()
            .filter_map(|p| Regex::new(p).ok())
            .collect();

        Self {
            detectors: vec![
                Box::new(RustErrorHandlingDetector::new()),
                Box::new(RustAsyncDetector::new()),
                Box::new(ApiPatternDetector::new()),
                Box::new(TestPatternDetector::new()),
                Box::new(DatabasePatternDetector::new()),
            ],
            config,
            exclude_regex,
        }
    }

    /// Add a custom detector.
    pub fn add_detector(mut self, detector: Box<dyn PatternDetector>) -> Self {
        self.detectors.push(detector);
        self
    }

    /// Scan a directory and extract patterns.
    pub fn transfuse(&self, path: &Path) -> crate::error::Result<TransfusionResult> {
        let mut result = TransfusionResult::default();

        for entry in WalkDir::new(path)
            .into_iter()
            .filter_entry(|e| !self.should_skip(e))
        {
            let entry = match entry {
                Ok(e) => e,
                Err(e) => {
                    result
                        .errors
                        .insert(format!("{:?}", e.path()), e.to_string());
                    continue;
                }
            };

            if !entry.file_type().is_file() {
                continue;
            }

            if !self.should_include(entry.path()) {
                continue;
            }

            // Check max files limit
            if self.config.max_files > 0 && result.files_scanned >= self.config.max_files {
                break;
            }

            result.files_scanned += 1;

            // Read file content
            let content = match fs::read_to_string(entry.path()) {
                Ok(c) => {
                    // Check file size
                    if c.len() > self.config.max_file_size {
                        continue;
                    }
                    c
                }
                Err(e) => {
                    result.errors.insert(
                        entry.path().to_string_lossy().to_string(),
                        e.to_string(),
                    );
                    continue;
                }
            };

            let file_path = entry.path().to_string_lossy().to_string();
            let extension = entry
                .path()
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or("");

            // Run all applicable detectors
            for detector in &self.detectors {
                let supported = detector.supported_extensions();
                if !supported.is_empty() && !supported.contains(&extension) {
                    continue;
                }

                let detected = detector.detect(&content, &file_path);
                for pattern in detected {
                    if pattern.confidence >= self.config.min_confidence {
                        result.patterns_extracted += 1;
                        *result
                            .by_category
                            .entry(pattern.domain.clone())
                            .or_insert(0) += 1;
                        *result
                            .by_detector
                            .entry(detector.name().to_string())
                            .or_insert(0) += 1;
                        result.patterns.push(pattern);

                        if !self.config.dry_run {
                            result.patterns_stored += 1;
                        }
                    } else {
                        result.patterns_skipped += 1;
                    }
                }
            }
        }

        Ok(result)
    }

    fn should_skip(&self, entry: &walkdir::DirEntry) -> bool {
        let name = entry.file_name().to_str().unwrap_or("");

        // Skip hidden directories
        if name.starts_with('.') && entry.file_type().is_dir() {
            return true;
        }

        // Skip excluded directories
        if entry.file_type().is_dir() {
            return self.config.exclude_dirs.iter().any(|d| name == d);
        }

        false
    }

    fn should_include(&self, path: &Path) -> bool {
        let path_str = path.to_string_lossy();

        // Check exclude patterns
        for regex in &self.exclude_regex {
            if regex.is_match(&path_str) {
                return false;
            }
        }

        // Check extension
        path.extension()
            .and_then(|e| e.to_str())
            .map(|e| self.config.include_extensions.contains(&e.to_string()))
            .unwrap_or(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transfusion_config_default() {
        let config = TransfusionConfig::default();
        assert_eq!(config.min_confidence, 0.7);
        assert!(config.include_extensions.contains(&"rs".to_string()));
        assert!(config.exclude_dirs.contains(&"target".to_string()));
    }

    #[test]
    fn test_rust_error_handling_detector() {
        let detector = RustErrorHandlingDetector::new();
        let content = r#"
use thiserror::Error;

#[derive(Error, Debug)]
pub enum MyError {
    #[error("Database error: {0}")]
    Database(String),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

impl std::error::Error for CustomError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        None
    }
}
"#;
        let patterns = detector.detect(content, "test.rs");
        assert!(!patterns.is_empty());
        assert!(patterns.iter().any(|p| p.domain == "rust.error_handling"));
    }

    #[test]
    fn test_rust_async_detector() {
        let detector = RustAsyncDetector::new();
        let content = r#"
async fn process_data() {
    tokio::spawn(async move {
        do_work().await;
    });
}

async fn multi_select() {
    tokio::select! {
        result = operation_a() => handle_a(result),
        result = operation_b() => handle_b(result),
    }
}
"#;
        let patterns = detector.detect(content, "test.rs");
        assert!(!patterns.is_empty());
        assert!(patterns.iter().any(|p| p.domain == "rust.async"));
    }

    #[test]
    fn test_api_pattern_detector() {
        let detector = ApiPatternDetector::new();
        let content = r#"
use axum::{Router, routing::get};

fn create_router() -> Router {
    Router::new()
        .route("/api/users", get(list_users))
        .route("/api/users/:id", get(get_user))
        .layer(TraceLayer::new_for_http())
}
"#;
        let patterns = detector.detect(content, "test.rs");
        assert!(!patterns.is_empty());
        assert!(patterns.iter().any(|p| p.domain == "rust.api"));
    }

    #[test]
    fn test_test_pattern_detector() {
        let detector = TestPatternDetector::new();
        let content = r#"
#[tokio::test]
async fn test_async_operation() {
    let result = async_op().await;
    assert!(result.is_ok());
}

proptest! {
    #[test]
    fn test_prop(input in ".*") {
        assert!(validate(&input).is_ok());
    }
}
"#;
        let patterns = detector.detect(content, "test.rs");
        assert!(!patterns.is_empty());
        assert!(patterns.iter().any(|p| p.domain == "rust.testing"));
    }

    #[test]
    fn test_database_pattern_detector() {
        let detector = DatabasePatternDetector::new();
        let content = r#"
use sqlx::PgPool;

async fn fetch_users(pool: &PgPool) -> Result<Vec<User>, Error> {
    let users = sqlx::query_as!(User, "SELECT * FROM users")
        .fetch_all(pool)
        .await?;
    Ok(users)
}

async fn create_pool() -> PgPool {
    PgPool::connect("postgres://localhost/db").await.unwrap()
}
"#;
        let patterns = detector.detect(content, "test.rs");
        assert!(!patterns.is_empty());
        assert!(patterns.iter().any(|p| p.domain == "rust.database"));
    }

    #[test]
    fn test_extracted_pattern_to_pattern() {
        let extracted = ExtractedPattern {
            problem: "Test problem".into(),
            solution: "Test solution".into(),
            domain: "rust.testing".into(),
            tags: vec!["test".into()],
            source_file: "/path/to/file.rs".into(),
            line_number: 42,
            confidence: 0.9,
            detector_name: "test_detector".into(),
        };

        let pattern = extracted.to_pattern();
        assert_eq!(pattern.problem(), "Test problem");
        assert_eq!(pattern.solution(), "Test solution");
        assert!((pattern.confidence() - 0.9).abs() < 0.001);
    }

    #[test]
    fn test_transfuser_should_skip() {
        let config = TransfusionConfig::default();
        let transfuser = Transfuser::new(config);

        // Create a mock entry for testing
        // In a real test we'd need actual directory entries
        // This test verifies the config is set correctly
        assert!(transfuser.config.exclude_dirs.contains(&"target".to_string()));
        assert!(transfuser.config.exclude_dirs.contains(&"node_modules".to_string()));
    }

    #[test]
    fn test_transfuser_should_include() {
        let config = TransfusionConfig::default();
        let transfuser = Transfuser::new(config);

        assert!(transfuser.should_include(Path::new("/path/to/file.rs")));
        assert!(transfuser.should_include(Path::new("/path/to/file.py")));
        assert!(!transfuser.should_include(Path::new("/path/to/file.txt")));
        assert!(!transfuser.should_include(Path::new("/path/to/file.min.js")));
    }
}
