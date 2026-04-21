//! Sliding window deduplication for the OSpipe pipeline.
//!
//! Uses cosine similarity on embeddings to detect duplicate content
//! within a configurable time window.

use std::collections::VecDeque;
use std::time::Duration;

use chrono::{DateTime, Utc};
use ndarray::Array1;
use serde::{Deserialize, Serialize};

use crate::ml::cosine_similarity_normalized;

/// Result of deduplication check.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DedupResult {
    /// Whether the content is considered a duplicate.
    pub is_duplicate: bool,
    /// Highest similarity score found against recent embeddings.
    pub max_similarity: f32,
    /// Index of the most similar recent embedding (if any).
    pub most_similar_index: Option<usize>,
    /// Timestamp of the most similar recent item.
    pub most_similar_timestamp: Option<DateTime<Utc>>,
}

impl DedupResult {
    /// Create a result indicating this is a duplicate.
    pub fn duplicate(similarity: f32, index: usize, timestamp: DateTime<Utc>) -> Self {
        Self {
            is_duplicate: true,
            max_similarity: similarity,
            most_similar_index: Some(index),
            most_similar_timestamp: Some(timestamp),
        }
    }

    /// Create a result indicating this is not a duplicate.
    pub fn unique(max_similarity: f32) -> Self {
        Self {
            is_duplicate: false,
            max_similarity,
            most_similar_index: None,
            most_similar_timestamp: None,
        }
    }

    /// Create a result for when there are no recent embeddings to compare.
    pub fn first_entry() -> Self {
        Self {
            is_duplicate: false,
            max_similarity: 0.0,
            most_similar_index: None,
            most_similar_timestamp: None,
        }
    }
}

/// Entry in the sliding window.
#[derive(Debug, Clone)]
struct WindowEntry {
    /// Timestamp of the entry.
    timestamp: DateTime<Utc>,
    /// Embedding vector (assumed to be normalized).
    embedding: Vec<f32>,
    /// Optional content hash for exact duplicate detection.
    content_hash: Option<u64>,
}

/// Sliding window deduplication engine.
///
/// Maintains a time-limited window of recent embeddings and checks new
/// content for similarity against them.
pub struct SlidingWindowDedup {
    /// Time window size. Entries older than this are evicted.
    window_size: Duration,
    /// Cosine similarity threshold for considering content as duplicate.
    similarity_threshold: f32,
    /// Recent embeddings within the window.
    recent_entries: VecDeque<WindowEntry>,
    /// Maximum number of entries to keep (prevents memory bloat).
    max_entries: usize,
    /// Statistics
    stats: DedupStats,
}

/// Statistics for the deduplication engine.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DedupStats {
    /// Total items checked.
    pub total_checked: u64,
    /// Total duplicates found.
    pub duplicates_found: u64,
    /// Total unique items.
    pub unique_items: u64,
    /// Current window size.
    pub window_entries: usize,
    /// Total evictions due to age.
    pub age_evictions: u64,
    /// Total evictions due to capacity.
    pub capacity_evictions: u64,
}

impl Default for SlidingWindowDedup {
    fn default() -> Self {
        Self::new(Duration::from_secs(5 * 60), 0.9)
    }
}

impl SlidingWindowDedup {
    /// Create a new sliding window deduplicator.
    ///
    /// # Arguments
    ///
    /// * `window_size` - Time duration to keep entries in the window.
    /// * `similarity_threshold` - Cosine similarity threshold (0.0-1.0) for duplicates.
    pub fn new(window_size: Duration, similarity_threshold: f32) -> Self {
        Self {
            window_size,
            similarity_threshold: similarity_threshold.clamp(0.0, 1.0),
            recent_entries: VecDeque::with_capacity(1000),
            max_entries: 10_000,
            stats: DedupStats::default(),
        }
    }

    /// Create with a custom maximum entry count.
    pub fn with_max_entries(mut self, max_entries: usize) -> Self {
        self.max_entries = max_entries;
        self
    }

    /// Set the similarity threshold.
    pub fn set_threshold(&mut self, threshold: f32) {
        self.similarity_threshold = threshold.clamp(0.0, 1.0);
    }

    /// Set the window size.
    pub fn set_window_size(&mut self, window_size: Duration) {
        self.window_size = window_size;
    }

    /// Evict old entries from the window.
    fn evict_old_entries(&mut self, now: DateTime<Utc>) {
        let cutoff = now - chrono::Duration::from_std(self.window_size).unwrap_or_default();

        while let Some(entry) = self.recent_entries.front() {
            if entry.timestamp < cutoff {
                self.recent_entries.pop_front();
                self.stats.age_evictions += 1;
            } else {
                break;
            }
        }
    }

    /// Evict entries if we're at capacity.
    fn evict_if_at_capacity(&mut self) {
        while self.recent_entries.len() >= self.max_entries {
            self.recent_entries.pop_front();
            self.stats.capacity_evictions += 1;
        }
    }

    /// Check if an embedding is a duplicate of recent content.
    ///
    /// # Arguments
    ///
    /// * `embedding` - The embedding vector (should be normalized).
    /// * `timestamp` - The timestamp of the content.
    ///
    /// # Returns
    ///
    /// A `DedupResult` indicating whether this is a duplicate.
    pub fn is_duplicate(&mut self, embedding: &[f32], timestamp: DateTime<Utc>) -> DedupResult {
        self.stats.total_checked += 1;

        // Evict old entries first
        self.evict_old_entries(timestamp);

        // If no recent entries, this is unique
        if self.recent_entries.is_empty() {
            self.stats.unique_items += 1;
            return DedupResult::first_entry();
        }

        // Convert to ndarray for efficient similarity computation
        let query = Array1::from_vec(embedding.to_vec());

        let mut max_similarity = 0.0f32;
        let mut most_similar_index = None;
        let mut most_similar_timestamp = None;

        // Compare against all recent entries
        for (idx, entry) in self.recent_entries.iter().enumerate() {
            if entry.embedding.len() != embedding.len() {
                continue; // Skip mismatched dimensions
            }

            let entry_arr = Array1::from_vec(entry.embedding.clone());
            let similarity = cosine_similarity_normalized(&query.view(), &entry_arr.view());

            if similarity > max_similarity {
                max_similarity = similarity;
                most_similar_index = Some(idx);
                most_similar_timestamp = Some(entry.timestamp);
            }
        }

        // Check if it exceeds threshold
        if max_similarity >= self.similarity_threshold {
            self.stats.duplicates_found += 1;
            DedupResult::duplicate(
                max_similarity,
                most_similar_index.unwrap_or(0),
                most_similar_timestamp.unwrap_or(timestamp),
            )
        } else {
            self.stats.unique_items += 1;
            DedupResult::unique(max_similarity)
        }
    }

    /// Check for duplicate and add to the window if unique.
    ///
    /// # Arguments
    ///
    /// * `embedding` - The embedding vector (should be normalized).
    /// * `timestamp` - The timestamp of the content.
    ///
    /// # Returns
    ///
    /// A `DedupResult` indicating whether this is a duplicate.
    pub fn check_and_add(&mut self, embedding: Vec<f32>, timestamp: DateTime<Utc>) -> DedupResult {
        let result = self.is_duplicate(&embedding, timestamp);

        // Only add if not a duplicate
        if !result.is_duplicate {
            self.add(embedding, timestamp);
        }

        result
    }

    /// Add an embedding to the window without checking for duplicates.
    ///
    /// Use this when you've already verified the content is unique.
    pub fn add(&mut self, embedding: Vec<f32>, timestamp: DateTime<Utc>) {
        self.evict_if_at_capacity();

        self.recent_entries.push_back(WindowEntry {
            timestamp,
            embedding,
            content_hash: None,
        });

        self.stats.window_entries = self.recent_entries.len();
    }

    /// Add an embedding with a content hash for exact duplicate detection.
    pub fn add_with_hash(&mut self, embedding: Vec<f32>, timestamp: DateTime<Utc>, hash: u64) {
        self.evict_if_at_capacity();

        self.recent_entries.push_back(WindowEntry {
            timestamp,
            embedding,
            content_hash: Some(hash),
        });

        self.stats.window_entries = self.recent_entries.len();
    }

    /// Check for exact duplicate by content hash.
    pub fn is_exact_duplicate(&self, hash: u64) -> bool {
        self.recent_entries
            .iter()
            .any(|e| e.content_hash == Some(hash))
    }

    /// Get the current window size (number of entries).
    pub fn window_len(&self) -> usize {
        self.recent_entries.len()
    }

    /// Get statistics.
    pub fn stats(&self) -> DedupStats {
        let mut stats = self.stats.clone();
        stats.window_entries = self.recent_entries.len();
        stats
    }

    /// Reset statistics.
    pub fn reset_stats(&mut self) {
        self.stats = DedupStats::default();
        self.stats.window_entries = self.recent_entries.len();
    }

    /// Clear the window.
    pub fn clear(&mut self) {
        self.recent_entries.clear();
        self.stats.window_entries = 0;
    }

    /// Get the similarity threshold.
    pub fn threshold(&self) -> f32 {
        self.similarity_threshold
    }

    /// Get the window duration.
    pub fn window_duration(&self) -> Duration {
        self.window_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_normalized_embedding(dim: usize, seed: u64) -> Vec<f32> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut embedding: Vec<f32> = (0..dim)
            .map(|i| {
                let mut hasher = DefaultHasher::new();
                (seed, i).hash(&mut hasher);
                (hasher.finish() as f32 / u64::MAX as f32) * 2.0 - 1.0
            })
            .collect();

        // Normalize
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            embedding.iter_mut().for_each(|x| *x /= norm);
        }

        embedding
    }

    #[test]
    fn test_dedup_first_entry() {
        let mut dedup = SlidingWindowDedup::new(Duration::from_secs(300), 0.9);
        let embedding = create_normalized_embedding(128, 1);
        let now = Utc::now();

        let result = dedup.is_duplicate(&embedding, now);

        assert!(!result.is_duplicate);
        assert_eq!(result.max_similarity, 0.0);
    }

    #[test]
    fn test_dedup_exact_duplicate() {
        let mut dedup = SlidingWindowDedup::new(Duration::from_secs(300), 0.9);
        let embedding = create_normalized_embedding(128, 1);
        let now = Utc::now();

        // Add first entry
        dedup.add(embedding.clone(), now);

        // Check same embedding
        let result = dedup.is_duplicate(&embedding, now);

        assert!(result.is_duplicate);
        assert!((result.max_similarity - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_dedup_different_content() {
        let mut dedup = SlidingWindowDedup::new(Duration::from_secs(300), 0.9);
        let embedding1 = create_normalized_embedding(128, 1);
        let embedding2 = create_normalized_embedding(128, 999); // Very different
        let now = Utc::now();

        dedup.add(embedding1, now);
        let result = dedup.is_duplicate(&embedding2, now);

        assert!(!result.is_duplicate);
        assert!(result.max_similarity < 0.9);
    }

    #[test]
    fn test_dedup_check_and_add() {
        let mut dedup = SlidingWindowDedup::new(Duration::from_secs(300), 0.9);
        let embedding1 = create_normalized_embedding(128, 1);
        let embedding2 = create_normalized_embedding(128, 2);
        let now = Utc::now();

        // First entry - should be added
        let result1 = dedup.check_and_add(embedding1.clone(), now);
        assert!(!result1.is_duplicate);
        assert_eq!(dedup.window_len(), 1);

        // Duplicate - should not be added
        let result2 = dedup.check_and_add(embedding1.clone(), now);
        assert!(result2.is_duplicate);
        assert_eq!(dedup.window_len(), 1);

        // Different - should be added
        let result3 = dedup.check_and_add(embedding2, now);
        assert!(!result3.is_duplicate);
        assert_eq!(dedup.window_len(), 2);
    }

    #[test]
    fn test_dedup_window_eviction() {
        let mut dedup = SlidingWindowDedup::new(Duration::from_secs(60), 0.9);
        let embedding1 = create_normalized_embedding(128, 1);
        let embedding2 = create_normalized_embedding(128, 2);

        let old_time = Utc::now() - chrono::Duration::seconds(120);
        let now = Utc::now();

        // Add old entry
        dedup.add(embedding1, old_time);
        assert_eq!(dedup.window_len(), 1);

        // Check new entry - should trigger eviction of old entry
        let _ = dedup.is_duplicate(&embedding2, now);
        assert_eq!(dedup.window_len(), 0);
    }

    #[test]
    fn test_dedup_capacity_eviction() {
        let mut dedup = SlidingWindowDedup::new(Duration::from_secs(300), 0.9)
            .with_max_entries(3);

        let now = Utc::now();

        for i in 0..5 {
            let embedding = create_normalized_embedding(128, i as u64);
            dedup.add(embedding, now);
        }

        assert_eq!(dedup.window_len(), 3);
        assert_eq!(dedup.stats().capacity_evictions, 2);
    }

    #[test]
    fn test_dedup_exact_hash_duplicate() {
        let mut dedup = SlidingWindowDedup::new(Duration::from_secs(300), 0.9);
        let embedding = create_normalized_embedding(128, 1);
        let now = Utc::now();
        let hash = 12345u64;

        dedup.add_with_hash(embedding, now, hash);

        assert!(dedup.is_exact_duplicate(hash));
        assert!(!dedup.is_exact_duplicate(99999));
    }

    #[test]
    fn test_dedup_stats() {
        let mut dedup = SlidingWindowDedup::new(Duration::from_secs(300), 0.9);
        let embedding1 = create_normalized_embedding(128, 1);
        let embedding2 = create_normalized_embedding(128, 999);
        let now = Utc::now();

        dedup.check_and_add(embedding1.clone(), now);
        dedup.check_and_add(embedding1.clone(), now); // Duplicate
        dedup.check_and_add(embedding2, now);

        let stats = dedup.stats();
        assert_eq!(stats.total_checked, 3);
        assert_eq!(stats.duplicates_found, 1);
        assert_eq!(stats.unique_items, 2);
        assert_eq!(stats.window_entries, 2);
    }

    #[test]
    fn test_dedup_threshold_adjustment() {
        let mut dedup = SlidingWindowDedup::new(Duration::from_secs(300), 0.5);

        // Create an embedding
        let embedding1 = create_normalized_embedding(128, 1);

        // Create a second embedding that is similar but not identical
        // Modify it to get a moderate similarity (~0.7)
        let mut embedding2 = create_normalized_embedding(128, 2);
        // Blend 70% of embedding1 into embedding2
        for i in 0..embedding2.len() {
            embedding2[i] = embedding1[i] * 0.7 + embedding2[i] * 0.3;
        }
        // Re-normalize
        let norm: f32 = embedding2.iter().map(|x| x * x).sum::<f32>().sqrt();
        embedding2.iter_mut().for_each(|x| *x /= norm);

        let now = Utc::now();
        dedup.add(embedding1, now);

        // With low threshold 0.5, this should be detected as duplicate
        let result1 = dedup.is_duplicate(&embedding2, now);

        // Now set very high threshold
        dedup.set_threshold(0.99);
        let result2 = dedup.is_duplicate(&embedding2, now);

        // With low threshold (0.5), should be duplicate
        // With high threshold (0.99), should NOT be duplicate
        assert!(result1.is_duplicate, "With threshold 0.5, should be duplicate");
        assert!(!result2.is_duplicate, "With threshold 0.99, should not be duplicate");
    }

    #[test]
    fn test_dedup_clear() {
        let mut dedup = SlidingWindowDedup::new(Duration::from_secs(300), 0.9);
        let embedding = create_normalized_embedding(128, 1);
        let now = Utc::now();

        dedup.add(embedding, now);
        assert_eq!(dedup.window_len(), 1);

        dedup.clear();
        assert_eq!(dedup.window_len(), 0);
    }
}
