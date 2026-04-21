//! Pattern export and import functionality.
//!
//! Supports JSON-based pattern sharing with filtering and deduplication.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Export format version.
pub const EXPORT_VERSION: &str = "1.0";

/// Exported pattern collection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternExport {
    /// Export format version
    pub version: String,
    /// When the export was created
    pub exported_at: DateTime<Utc>,
    /// Source system identifier
    pub source: String,
    /// Number of patterns in this export
    pub pattern_count: usize,
    /// The exported patterns
    pub patterns: Vec<ExportedPattern>,
}

/// A single exported pattern (portable format).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportedPattern {
    /// Problem description
    pub problem: String,
    /// Solution description
    pub solution: String,
    /// Domain/category
    pub domain: String,
    /// Optional context
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context: Option<String>,
    /// Tags
    #[serde(default)]
    pub tags: Vec<String>,
    /// Reward score
    pub reward: f32,
    /// Effectiveness score
    pub effectiveness: f32,
    /// Confidence score
    pub confidence: f32,
    /// Success rate (derived from effectiveness as a proxy)
    pub success_rate: f32,
    /// Reuse count
    pub reuse_count: u32,
    /// Pattern tier (booster, crystal, reflex)
    #[serde(default = "default_tier")]
    pub tier: String,
}

fn default_tier() -> String {
    "booster".to_string()
}

/// Import result tracking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportResult {
    /// Total patterns in the file
    pub total_in_file: usize,
    /// Patterns that were imported (new)
    pub imported: usize,
    /// Patterns that were skipped (duplicates)
    pub skipped: usize,
    /// Patterns that were updated (existing with newer data)
    pub updated: usize,
    /// Errors encountered
    pub errors: Vec<String>,
}

/// Generate a deduplication key from problem + domain.
pub fn dedup_key(problem: &str, domain: &str) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut hasher = DefaultHasher::new();
    problem.to_lowercase().trim().hash(&mut hasher);
    domain.to_lowercase().trim().hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

/// Write a pattern export to a file.
pub fn write_export(export: &PatternExport, path: &Path) -> std::io::Result<()> {
    let json = serde_json::to_string_pretty(export)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    std::fs::write(path, json)
}

/// Read a pattern export from a file.
pub fn read_export(path: &Path) -> std::io::Result<PatternExport> {
    let json = std::fs::read_to_string(path)?;
    serde_json::from_str(&json)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dedup_key_same_input() {
        let key1 = dedup_key("How to cache data", "performance");
        let key2 = dedup_key("How to cache data", "performance");
        assert_eq!(key1, key2);
    }

    #[test]
    fn test_dedup_key_case_insensitive() {
        let key1 = dedup_key("How to Cache Data", "Performance");
        let key2 = dedup_key("how to cache data", "performance");
        assert_eq!(key1, key2);
    }

    #[test]
    fn test_dedup_key_different_input() {
        let key1 = dedup_key("How to cache data", "performance");
        let key2 = dedup_key("How to test code", "testing");
        assert_ne!(key1, key2);
    }

    #[test]
    fn test_export_roundtrip() {
        let export = PatternExport {
            version: EXPORT_VERSION.to_string(),
            exported_at: Utc::now(),
            source: "test".to_string(),
            pattern_count: 1,
            patterns: vec![ExportedPattern {
                problem: "Test problem".to_string(),
                solution: "Test solution".to_string(),
                domain: "testing".to_string(),
                context: Some("Test context".to_string()),
                tags: vec!["tag1".to_string()],
                reward: 0.8,
                effectiveness: 0.9,
                confidence: 0.7,
                success_rate: 0.85,
                reuse_count: 5,
                tier: "booster".to_string(),
            }],
        };

        let json = serde_json::to_string_pretty(&export).unwrap();
        let parsed: PatternExport = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.version, EXPORT_VERSION);
        assert_eq!(parsed.pattern_count, 1);
        assert_eq!(parsed.patterns[0].problem, "Test problem");
        assert_eq!(parsed.patterns[0].reward, 0.8);
    }

    #[test]
    fn test_write_and_read_export() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test-export.json");

        let export = PatternExport {
            version: EXPORT_VERSION.to_string(),
            exported_at: Utc::now(),
            source: "test".to_string(),
            pattern_count: 1,
            patterns: vec![ExportedPattern {
                problem: "File I/O test".to_string(),
                solution: "Write then read".to_string(),
                domain: "testing".to_string(),
                context: None,
                tags: vec![],
                reward: 0.5,
                effectiveness: 0.5,
                confidence: 0.5,
                success_rate: 0.0,
                reuse_count: 0,
                tier: "booster".to_string(),
            }],
        };

        write_export(&export, &path).unwrap();
        let loaded = read_export(&path).unwrap();

        assert_eq!(loaded.version, EXPORT_VERSION);
        assert_eq!(loaded.patterns[0].problem, "File I/O test");
    }

    #[test]
    fn test_default_tier() {
        assert_eq!(default_tier(), "booster");
    }

    #[test]
    fn test_exported_pattern_serialization_skips_none_context() {
        let pattern = ExportedPattern {
            problem: "p".to_string(),
            solution: "s".to_string(),
            domain: "d".to_string(),
            context: None,
            tags: vec![],
            reward: 0.5,
            effectiveness: 0.5,
            confidence: 0.5,
            success_rate: 0.0,
            reuse_count: 0,
            tier: "booster".to_string(),
        };

        let json = serde_json::to_string(&pattern).unwrap();
        assert!(!json.contains("context"));
    }
}
