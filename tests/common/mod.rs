//! Common test utilities for Nagual unit tests.
//!
//! Provides shared fixtures, mock objects, and helper functions
//! for testing database operations, reasoning bank, ML functions,
//! and security components.

use std::path::PathBuf;
use std::sync::Arc;

use chrono::{DateTime, Duration, Utc};
use ndarray::Array1;
use rand::Rng;
use tempfile::TempDir;

// Re-export tempfile for convenience
pub use tempfile;

// ============================================================================
// Test Fixtures
// ============================================================================

/// Test fixture that manages temporary resources.
pub struct TestFixture {
    /// Temporary directory for test files.
    pub temp_dir: TempDir,
}

impl TestFixture {
    /// Create a new test fixture with a temporary directory.
    pub fn new() -> Self {
        Self {
            temp_dir: TempDir::new().expect("Failed to create temp dir"),
        }
    }

    /// Get a path within the temporary directory.
    pub fn path(&self, name: &str) -> PathBuf {
        self.temp_dir.path().join(name)
    }

    /// Get a unique database path.
    pub fn db_path(&self) -> PathBuf {
        self.path(&format!("test_{}.db", uuid::Uuid::new_v4()))
    }
}

impl Default for TestFixture {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Pattern Test Data Generators
// ============================================================================

/// Generate a random pattern ID.
pub fn random_pattern_id() -> String {
    uuid::Uuid::new_v4().to_string()
}

/// Generate test embedding vector of specified dimension.
pub fn random_embedding(dim: usize) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect()
}

/// Generate a normalized embedding vector.
pub fn normalized_embedding(dim: usize) -> Vec<f32> {
    let v = random_embedding(dim);
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > f32::EPSILON {
        v.iter().map(|x| x / norm).collect()
    } else {
        let mut result = vec![0.0; dim];
        if !result.is_empty() {
            result[0] = 1.0;
        }
        result
    }
}

/// Generate unit vector in specific direction.
pub fn unit_vector(dim: usize, axis: usize) -> Vec<f32> {
    let mut v = vec![0.0; dim];
    if axis < dim {
        v[axis] = 1.0;
    }
    v
}

/// Generate orthogonal embedding vectors.
pub fn orthogonal_embeddings(dim: usize, count: usize) -> Vec<Vec<f32>> {
    (0..count)
        .map(|i| unit_vector(dim, i % dim))
        .collect()
}

/// Generate similar embeddings (small perturbations).
pub fn similar_embeddings(base: &[f32], count: usize, noise_scale: f32) -> Vec<Vec<f32>> {
    let mut rng = rand::thread_rng();
    (0..count)
        .map(|_| {
            let perturbed: Vec<f32> = base
                .iter()
                .map(|x| x + rng.gen_range(-noise_scale..noise_scale))
                .collect();
            // Normalize
            let norm: f32 = perturbed.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > f32::EPSILON {
                perturbed.iter().map(|x| x / norm).collect()
            } else {
                perturbed
            }
        })
        .collect()
}

// ============================================================================
// Database Test Helpers
// ============================================================================

/// SQL for creating a test patterns table.
pub const CREATE_PATTERNS_SQL: &str = r#"
CREATE TABLE IF NOT EXISTS patterns (
    id TEXT PRIMARY KEY,
    problem TEXT NOT NULL,
    solution TEXT NOT NULL,
    domain TEXT NOT NULL DEFAULT 'general',
    context TEXT,
    confidence REAL DEFAULT 0.5,
    reward REAL DEFAULT 0.5,
    success INTEGER DEFAULT 1,
    reuse_count INTEGER DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_patterns_domain ON patterns(domain);
CREATE INDEX IF NOT EXISTS idx_patterns_reward ON patterns(reward);
"#;

/// SQL for creating a test embeddings table.
pub const CREATE_EMBEDDINGS_SQL: &str = r#"
CREATE TABLE IF NOT EXISTS embeddings (
    pattern_id TEXT PRIMARY KEY REFERENCES patterns(id),
    embedding BLOB NOT NULL,
    dimension INTEGER NOT NULL,
    created_at TEXT NOT NULL
);
"#;

/// Test pattern data structure.
#[derive(Debug, Clone)]
pub struct TestPattern {
    pub id: String,
    pub problem: String,
    pub solution: String,
    pub domain: String,
    pub context: Option<String>,
    pub confidence: f32,
    pub reward: f32,
    pub success: bool,
    pub reuse_count: u32,
    pub embedding: Option<Vec<f32>>,
}

impl TestPattern {
    /// Create a new test pattern with default values.
    pub fn new(problem: &str, solution: &str) -> Self {
        Self {
            id: random_pattern_id(),
            problem: problem.to_string(),
            solution: solution.to_string(),
            domain: "test".to_string(),
            context: None,
            confidence: 0.5,
            reward: 0.5,
            success: true,
            reuse_count: 0,
            embedding: None,
        }
    }

    /// Builder method for domain.
    pub fn with_domain(mut self, domain: &str) -> Self {
        self.domain = domain.to_string();
        self
    }

    /// Builder method for context.
    pub fn with_context(mut self, context: &str) -> Self {
        self.context = Some(context.to_string());
        self
    }

    /// Builder method for confidence.
    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = confidence.clamp(0.0, 1.0);
        self
    }

    /// Builder method for reward.
    pub fn with_reward(mut self, reward: f32) -> Self {
        self.reward = reward.clamp(0.0, 1.0);
        self
    }

    /// Builder method for success.
    pub fn with_success(mut self, success: bool) -> Self {
        self.success = success;
        self
    }

    /// Builder method for reuse count.
    pub fn with_reuse_count(mut self, count: u32) -> Self {
        self.reuse_count = count;
        self
    }

    /// Builder method for embedding.
    pub fn with_embedding(mut self, embedding: Vec<f32>) -> Self {
        self.embedding = Some(embedding);
        self
    }

    /// Builder method for random embedding.
    pub fn with_random_embedding(mut self, dim: usize) -> Self {
        self.embedding = Some(normalized_embedding(dim));
        self
    }
}

/// Generate multiple test patterns.
pub fn generate_test_patterns(count: usize) -> Vec<TestPattern> {
    (0..count)
        .map(|i| {
            TestPattern::new(
                &format!("Test problem {}", i),
                &format!("Test solution {}", i),
            )
            .with_domain(&format!("domain{}", i % 5))
            .with_reward(0.3 + (i as f32 * 0.1).min(0.6))
            .with_random_embedding(128)
        })
        .collect()
}

/// Generate patterns with specific domains for hierarchy testing.
pub fn generate_domain_hierarchy_patterns() -> Vec<TestPattern> {
    vec![
        TestPattern::new("Rust error handling", "Use Result type")
            .with_domain("rust.error_handling")
            .with_reward(0.9),
        TestPattern::new("Rust async", "Use tokio runtime")
            .with_domain("rust.async.tokio")
            .with_reward(0.85),
        TestPattern::new("Database timeout", "Set connection timeout")
            .with_domain("database.postgres")
            .with_reward(0.8),
        TestPattern::new("Redis caching", "Use TTL for cache")
            .with_domain("database.redis")
            .with_reward(0.75),
        TestPattern::new("Security auth", "Use JWT tokens")
            .with_domain("security.authentication")
            .with_reward(0.7),
    ]
}

// ============================================================================
// PII Test Data
// ============================================================================

/// Sample texts with various PII types for testing.
pub mod pii_samples {
    /// Text with email address.
    pub const EMAIL: &str = "Contact me at john.doe@example.com for more info.";

    /// Text with phone number.
    pub const PHONE: &str = "Call me at (555) 123-4567 or +1-555-987-6543";

    /// Text with SSN.
    pub const SSN: &str = "SSN: 456-78-9012";

    /// Text with credit card (Visa).
    pub const CREDIT_CARD: &str = "Card: 4532-1234-5678-9012";

    /// Text with IP address.
    pub const IP_ADDRESS: &str = "Server at 192.168.1.100";

    /// Text with AWS key.
    pub const AWS_KEY: &str = "AWS key: AKIAIOSFODNN7EXAMPLE";

    /// Text with GitHub token.
    pub const GITHUB_TOKEN: &str = "Token: ghp_abcdefghijklmnopqrstuvwxyz0123456789";

    /// Text with JWT.
    pub const JWT: &str = "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U";

    /// Text with private key header.
    pub const PRIVATE_KEY: &str = "-----BEGIN RSA PRIVATE KEY-----\nMIIEpQ...";

    /// Text with no PII.
    pub const CLEAN: &str = "This is a clean text with no sensitive information.";

    /// Text with multiple PII types.
    pub const MIXED: &str = "Contact john@test.com or call (555) 123-4567. SSN: 456-78-9012";

    /// Known false positives that should be filtered.
    pub const FALSE_POSITIVE_IP: &str = "Listening on 127.0.0.1:8080";
    pub const FALSE_POSITIVE_SSN: &str = "SSN: 123-45-6789";
}

// ============================================================================
// Assertion Helpers
// ============================================================================

/// Assert that two f32 values are approximately equal.
pub fn assert_approx_eq(a: f32, b: f32, tolerance: f32) {
    assert!(
        (a - b).abs() < tolerance,
        "Values not approximately equal: {} vs {} (tolerance: {})",
        a,
        b,
        tolerance
    );
}

/// Assert that two f64 values are approximately equal.
pub fn assert_approx_eq_f64(a: f64, b: f64, tolerance: f64) {
    assert!(
        (a - b).abs() < tolerance,
        "Values not approximately equal: {} vs {} (tolerance: {})",
        a,
        b,
        tolerance
    );
}

/// Assert that a vector is normalized (L2 norm is 1.0).
pub fn assert_normalized(v: &[f32], tolerance: f32) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(
        (norm - 1.0).abs() < tolerance,
        "Vector not normalized: norm = {} (expected 1.0, tolerance: {})",
        norm,
        tolerance
    );
}

/// Assert that cosine similarity is within expected range.
pub fn assert_similarity_in_range(sim: f32, min: f32, max: f32) {
    assert!(
        sim >= min && sim <= max,
        "Similarity {} not in expected range [{}, {}]",
        sim,
        min,
        max
    );
}

/// Calculate L2 norm of a vector.
pub fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

/// Calculate cosine similarity between two vectors.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Vectors must have same length");

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a = l2_norm(a);
    let norm_b = l2_norm(b);

    if norm_a > f32::EPSILON && norm_b > f32::EPSILON {
        dot / (norm_a * norm_b)
    } else {
        0.0
    }
}

/// Normalize a vector using L2 normalization.
pub fn normalize_l2(v: &[f32]) -> Vec<f32> {
    let norm = l2_norm(v);
    if norm > f32::EPSILON {
        v.iter().map(|x| x / norm).collect()
    } else {
        v.to_vec()
    }
}

// ============================================================================
// Timing Helpers
// ============================================================================

/// Measure execution time of a closure.
pub fn measure_time<F, R>(f: F) -> (R, std::time::Duration)
where
    F: FnOnce() -> R,
{
    let start = std::time::Instant::now();
    let result = f();
    let duration = start.elapsed();
    (result, duration)
}

/// Measure async execution time.
pub async fn measure_time_async<F, R, Fut>(f: F) -> (R, std::time::Duration)
where
    F: FnOnce() -> Fut,
    Fut: std::future::Future<Output = R>,
{
    let start = std::time::Instant::now();
    let result = f().await;
    let duration = start.elapsed();
    (result, duration)
}

/// Assert that an operation completes within a time limit.
pub fn assert_completes_within<F, R>(f: F, limit_ms: u64) -> R
where
    F: FnOnce() -> R,
{
    let (result, duration) = measure_time(f);
    assert!(
        duration.as_millis() < limit_ms as u128,
        "Operation took {}ms, expected < {}ms",
        duration.as_millis(),
        limit_ms
    );
    result
}

// ============================================================================
// Mock Types
// ============================================================================

/// Mock embedder that returns deterministic embeddings.
pub struct MockEmbedder {
    pub dimension: usize,
}

impl MockEmbedder {
    pub fn new(dimension: usize) -> Self {
        Self { dimension }
    }

    /// Generate a deterministic embedding based on text hash.
    pub fn embed(&self, text: &str) -> Vec<f32> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        text.hash(&mut hasher);
        let seed = hasher.finish();

        // Generate deterministic values from seed
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let v: Vec<f32> = (0..self.dimension)
            .map(|_| rng.gen_range(-1.0..1.0))
            .collect();

        normalize_l2(&v)
    }

    /// Batch embed multiple texts.
    pub fn embed_batch(&self, texts: &[&str]) -> Vec<Vec<f32>> {
        texts.iter().map(|t| self.embed(t)).collect()
    }
}

use rand::SeedableRng;

// ============================================================================
// Test Configuration
// ============================================================================

/// Common test dimensions.
pub mod dimensions {
    pub const SMALL: usize = 64;
    pub const STANDARD: usize = 128;
    pub const LARGE: usize = 384;
}

/// Test database configuration.
#[derive(Debug, Clone)]
pub struct TestDbConfig {
    pub use_encryption: bool,
    pub encryption_key: Option<String>,
    pub enable_fts: bool,
}

impl Default for TestDbConfig {
    fn default() -> Self {
        Self {
            use_encryption: false,
            encryption_key: None,
            enable_fts: true,
        }
    }
}

impl TestDbConfig {
    /// Create config for encrypted database.
    pub fn encrypted(key: &str) -> Self {
        Self {
            use_encryption: true,
            encryption_key: Some(key.to_string()),
            enable_fts: true,
        }
    }

    /// Create config without FTS.
    pub fn without_fts() -> Self {
        Self {
            use_encryption: false,
            encryption_key: None,
            enable_fts: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_random_embedding() {
        let emb = random_embedding(128);
        assert_eq!(emb.len(), 128);
    }

    #[test]
    fn test_normalized_embedding() {
        let emb = normalized_embedding(128);
        assert_eq!(emb.len(), 128);
        assert_normalized(&emb, 1e-5);
    }

    #[test]
    fn test_unit_vector() {
        let v = unit_vector(4, 2);
        assert_eq!(v, vec![0.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn test_orthogonal_embeddings() {
        let embeddings = orthogonal_embeddings(4, 4);
        assert_eq!(embeddings.len(), 4);

        // First 4 should be orthogonal
        let sim = cosine_similarity(&embeddings[0], &embeddings[1]);
        assert_approx_eq(sim, 0.0, 1e-6);
    }

    #[test]
    fn test_similar_embeddings() {
        let base = normalized_embedding(128);
        // Use smaller noise for higher similarity guarantee
        let similar = similar_embeddings(&base, 5, 0.05);

        for s in &similar {
            let sim = cosine_similarity(&base, s);
            // With small perturbations, similarity should be high but not necessarily > 0.9
            assert!(sim > 0.7, "Similar embeddings should have high similarity, got {}", sim);
        }
    }

    #[test]
    fn test_mock_embedder() {
        let embedder = MockEmbedder::new(128);
        let emb1 = embedder.embed("test text");
        let emb2 = embedder.embed("test text");
        let emb3 = embedder.embed("different text");

        // Same text should produce same embedding
        assert_eq!(emb1, emb2);

        // Different text should produce different embedding
        assert_ne!(emb1, emb3);

        // Should be normalized
        assert_normalized(&emb1, 1e-5);
    }

    #[test]
    fn test_test_pattern() {
        let pattern = TestPattern::new("problem", "solution")
            .with_domain("test.domain")
            .with_reward(0.9)
            .with_random_embedding(128);

        assert_eq!(pattern.problem, "problem");
        assert_eq!(pattern.domain, "test.domain");
        assert_approx_eq(pattern.reward, 0.9, 1e-6);
        assert!(pattern.embedding.is_some());
    }

    #[test]
    fn test_measure_time() {
        let (result, duration) = measure_time(|| {
            std::thread::sleep(std::time::Duration::from_millis(10));
            42
        });

        assert_eq!(result, 42);
        assert!(duration.as_millis() >= 10);
    }
}
