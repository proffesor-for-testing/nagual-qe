//! Hash-based embedding generator (no external model files required).
//!
//! Produces 128-dimensional embeddings using SHAKE-256 structured hashing.
//! Ported from mcp-brain's embed.rs with three subspaces:
//! - Unigram:  dims [0..42)   — 42 dims (33%)
//! - Bigram:   dims [42..84)  — 42 dims (33%)
//! - Trigram:  dims [84..128) — 44 dims (34%)
//!
//! Each n-gram level uses signed hashing (bucket index + sign) to reduce
//! collision bias. The result is L2-normalized to unit length.
//!
//! This embedder is deterministic, fast, and requires zero external files.

use sha3::{
    digest::{ExtendableOutput, Update, XofReader},
    Shake256,
};

use super::{EmbeddingResult, MlResult};

/// Embedding dimension (128 f32s = 512 bytes).
const EMBEDDING_DIM: usize = 128;

/// Subspace allocation for multi-granularity hashing.
const UNIGRAM_START: usize = 0;
const UNIGRAM_END: usize = 42;
const BIGRAM_START: usize = 42;
const BIGRAM_END: usize = 84;
const TRIGRAM_START: usize = 84;
const TRIGRAM_END: usize = EMBEDDING_DIM;

/// Hash-based embedder that produces 128-dimensional vectors without any
/// external model files. Uses SHAKE-256 structured hashing with disjoint
/// subspaces for unigram, bigram, and trigram features.
pub struct HashEmbedder {
    dim: usize,
}

impl HashEmbedder {
    /// Create a new hash embedder (always 128-dimensional).
    pub fn new() -> Self {
        Self { dim: EMBEDDING_DIM }
    }

    /// Embed a single text into a 128-dimensional vector.
    pub fn embed(&self, text: &str) -> MlResult<EmbeddingResult> {
        let features = generate_structured_hash_features(text);
        Ok(EmbeddingResult {
            embedding: features,
            normalized: true,
            token_count: text.split_whitespace().count(),
            truncated: false,
        })
    }

    /// Embed a batch of texts.
    pub fn embed_batch(&self, texts: &[&str]) -> MlResult<Vec<EmbeddingResult>> {
        texts.iter().map(|t| self.embed(t)).collect()
    }

    /// Return the embedding dimension (always 128).
    pub fn embedding_dim(&self) -> usize {
        self.dim
    }
}

impl Default for HashEmbedder {
    fn default() -> Self {
        Self::new()
    }
}

/// Generate structured multi-granularity hash features.
///
/// Splits text into unigram, bigram, and trigram tokens. Each n-gram level
/// hashes into a disjoint subspace of the embedding vector using signed
/// hashing (hash determines both the bucket index AND the sign, reducing
/// collision bias).
///
/// For short texts (1-2 words), character trigrams are also added to the
/// trigram subspace at reduced weight.
fn generate_structured_hash_features(text: &str) -> Vec<f32> {
    let mut features = vec![0.0f32; EMBEDDING_DIM];
    let lower = text.to_lowercase();
    let words: Vec<&str> = lower.split_whitespace().collect();

    // Unigram features: each word hashes into dims [0..42)
    let unigram_dim = UNIGRAM_END - UNIGRAM_START;
    for word in &words {
        let (bucket, sign) = signed_hash(word.as_bytes(), b"uni", unigram_dim);
        features[UNIGRAM_START + bucket] += sign;
    }

    // Bigram features: consecutive word pairs hash into dims [42..84)
    let bigram_dim = BIGRAM_END - BIGRAM_START;
    for pair in words.windows(2) {
        let key = format!("{} {}", pair[0], pair[1]);
        let (bucket, sign) = signed_hash(key.as_bytes(), b"bi", bigram_dim);
        features[BIGRAM_START + bucket] += sign;
    }

    // Trigram features: consecutive word triples hash into dims [84..128)
    let trigram_dim = TRIGRAM_END - TRIGRAM_START;
    for triple in words.windows(3) {
        let key = format!("{} {} {}", triple[0], triple[1], triple[2]);
        let (bucket, sign) = signed_hash(key.as_bytes(), b"tri", trigram_dim);
        features[TRIGRAM_START + bucket] += sign;
    }

    // Character trigrams for short texts (1-2 words)
    if words.len() <= 2 {
        let chars: Vec<char> = lower.chars().filter(|c| c.is_alphanumeric()).collect();
        for window in chars.windows(3) {
            let key: String = window.iter().collect();
            let (bucket, sign) = signed_hash(key.as_bytes(), b"ctri", trigram_dim);
            features[TRIGRAM_START + bucket] += sign * 0.5;
        }
    }

    // L2 normalize
    let norm: f32 = features.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-10 {
        for v in &mut features {
            *v = (*v / norm).clamp(-1.0, 1.0);
        }
    }

    features
}

/// Signed hash: returns (bucket_index, +1.0 or -1.0).
///
/// Uses SHAKE-256 for uniform distribution. The first 4 bytes determine
/// the bucket, the 5th byte determines the sign.
fn signed_hash(data: &[u8], salt: &[u8], num_buckets: usize) -> (usize, f32) {
    let mut hasher = Shake256::default();
    hasher.update(b"ruvector-shf:");
    hasher.update(salt);
    hasher.update(b":");
    hasher.update(data);
    let mut reader = hasher.finalize_xof();
    let mut buf = [0u8; 5];
    reader.read(&mut buf);

    let bucket = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]) as usize % num_buckets;
    let sign = if buf[4] & 1 == 0 { 1.0f32 } else { -1.0f32 };
    (bucket, sign)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_embedding_generation() {
        let embedder = HashEmbedder::new();
        let result = embedder.embed("hello world").unwrap();
        assert_eq!(result.embedding.len(), 128);
        assert!(result.normalized);
        assert_eq!(result.token_count, 2);
        assert!(!result.truncated);
    }

    #[test]
    fn test_empty_input() {
        let embedder = HashEmbedder::new();
        let result = embedder.embed("").unwrap();
        assert_eq!(result.embedding.len(), 128);
        // Empty input produces a zero vector (no features activated)
        assert!(result.embedding.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_whitespace_only_input() {
        let embedder = HashEmbedder::new();
        let result = embedder.embed("   \t\n  ").unwrap();
        assert_eq!(result.embedding.len(), 128);
        assert!(result.embedding.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_dimension_always_128() {
        let embedder = HashEmbedder::new();
        assert_eq!(embedder.embedding_dim(), 128);

        for text in &["a", "hello world", "the quick brown fox jumps over the lazy dog"] {
            let result = embedder.embed(text).unwrap();
            assert_eq!(result.embedding.len(), 128);
        }
    }

    #[test]
    fn test_l2_normalized() {
        let embedder = HashEmbedder::new();
        let result = embedder.embed("a longer text for normalization testing").unwrap();
        let norm: f32 = result.embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 0.01,
            "Expected norm ~1.0, got {norm}"
        );
    }

    #[test]
    fn test_deterministic() {
        let embedder = HashEmbedder::new();
        let a = embedder.embed("deterministic test input").unwrap();
        let b = embedder.embed("deterministic test input").unwrap();
        assert_eq!(a.embedding, b.embedding);
    }

    #[test]
    fn test_different_inputs_different_outputs() {
        let embedder = HashEmbedder::new();
        let a = embedder.embed("rust programming language").unwrap();
        let b = embedder.embed("cooking recipes for dinner").unwrap();
        assert_ne!(a.embedding, b.embedding);
    }

    #[test]
    fn test_similar_texts_closer() {
        let embedder = HashEmbedder::new();
        let e1 = embedder.embed("rust programming language features").unwrap();
        let e2 = embedder.embed("rust programming language syntax").unwrap();
        let e3 = embedder.embed("cooking recipes for dinner tonight").unwrap();

        let sim12 = cosine_sim(&e1.embedding, &e2.embedding);
        let sim13 = cosine_sim(&e1.embedding, &e3.embedding);
        assert!(
            sim12 > sim13,
            "Similar texts should be closer: sim12={sim12}, sim13={sim13}"
        );
    }

    #[test]
    fn test_batch_embedding() {
        let embedder = HashEmbedder::new();
        let texts = vec!["hello world", "foo bar", "rust is great"];
        let results = embedder.embed_batch(&texts).unwrap();
        assert_eq!(results.len(), 3);
        for r in &results {
            assert_eq!(r.embedding.len(), 128);
        }
        // Each should match individual embedding
        for (i, text) in texts.iter().enumerate() {
            let single = embedder.embed(text).unwrap();
            assert_eq!(results[i].embedding, single.embedding);
        }
    }

    #[test]
    fn test_single_word_activates_unigram() {
        // Single word should activate unigram subspace and possibly char trigrams
        let features = generate_structured_hash_features("hello");
        let uni_energy: f32 = features[UNIGRAM_START..UNIGRAM_END]
            .iter()
            .map(|x| x * x)
            .sum();
        assert!(uni_energy > 0.0, "Unigram subspace should be active");

        // Bigram subspace should be zero (single word = no pairs)
        let bi_energy: f32 = features[BIGRAM_START..BIGRAM_END]
            .iter()
            .map(|x| x * x)
            .sum();
        assert_eq!(bi_energy, 0.0, "Bigram subspace should be inactive for single word");
    }

    #[test]
    fn test_signed_hash_distribution() {
        let mut pos = 0;
        let mut neg = 0;
        for i in 0..100 {
            let key = format!("test-{i}");
            let (_, sign) = signed_hash(key.as_bytes(), b"test", 42);
            if sign > 0.0 {
                pos += 1;
            } else {
                neg += 1;
            }
        }
        // Both signs should appear in 100 trials
        assert!(pos > 10 && neg > 10, "pos={pos}, neg={neg}");
    }

    /// Helper: compute cosine similarity between two vectors.
    fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm_a < 1e-10 || norm_b < 1e-10 {
            return 0.0;
        }
        dot / (norm_a * norm_b)
    }
}
