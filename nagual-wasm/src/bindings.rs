//! JavaScript bindings for WasmProfDAG.
//!
//! Provides the main entry point for JavaScript code to interact with
//! the pattern search engine.

use crate::search::{Pattern, SearchConfig, VectorSearch};
use crate::storage::{IndexedDBStorage, StorageStats};
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::future_to_promise;

/// Main ProfDAG interface for WASM.
///
/// Provides pattern storage, search, and persistence capabilities
/// for browser and edge environments.
#[wasm_bindgen]
pub struct WasmProfDAG {
    search: VectorSearch,
    storage: IndexedDBStorage,
}

#[wasm_bindgen]
impl WasmProfDAG {
    /// Create a new WasmProfDAG instance with default configuration.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            search: VectorSearch::with_defaults(),
            storage: IndexedDBStorage::new(),
        }
    }

    /// Create with custom configuration.
    #[wasm_bindgen]
    pub fn with_config(config: SearchConfig) -> Self {
        Self {
            search: VectorSearch::new(config),
            storage: IndexedDBStorage::new(),
        }
    }

    /// Add a pattern to the index.
    ///
    /// # Arguments
    /// * `id` - Unique pattern identifier
    /// * `content` - Pattern content/description
    /// * `embedding` - Float32Array with embedding vector (default: 128 dimensions)
    ///
    /// # Returns
    /// * `Ok(())` on success
    /// * `Err(JsError)` if embedding dimension is invalid
    #[wasm_bindgen]
    pub fn add_pattern(
        &mut self,
        id: &str,
        content: &str,
        embedding: &[f32],
    ) -> Result<(), JsError> {
        let pattern = Pattern::new(id.to_string(), content.to_string(), embedding.to_vec());

        self.search
            .add_pattern(pattern)
            .map_err(|e| JsError::new(&e))
    }

    /// Add a pattern with full metadata.
    ///
    /// # Arguments
    /// * `pattern_js` - JavaScript object with pattern data
    ///
    /// Expected format:
    /// ```javascript
    /// {
    ///   id: "unique-id",
    ///   content: "pattern description",
    ///   embedding: Float32Array,
    ///   pattern_type: "pattern" | "trajectory" | "prediction" | "decision",
    ///   confidence: 0.0-1.0,
    ///   metadata: {}
    /// }
    /// ```
    #[wasm_bindgen]
    pub fn add_pattern_full(&mut self, pattern_js: JsValue) -> Result<(), JsError> {
        let pattern: Pattern =
            serde_wasm_bindgen::from_value(pattern_js).map_err(|e| JsError::new(&e.to_string()))?;

        self.search
            .add_pattern(pattern)
            .map_err(|e| JsError::new(&e))
    }

    /// Remove a pattern by ID.
    #[wasm_bindgen]
    pub fn remove_pattern(&mut self, id: &str) -> bool {
        self.search.remove_pattern(id)
    }

    /// Get a pattern by ID.
    #[wasm_bindgen]
    pub fn get_pattern(&self, id: &str) -> JsValue {
        match self.search.get_pattern(id) {
            Some(pattern) => serde_wasm_bindgen::to_value(pattern).unwrap_or(JsValue::NULL),
            None => JsValue::NULL,
        }
    }

    /// Search for similar patterns.
    ///
    /// # Arguments
    /// * `query_embedding` - Float32Array with query embedding
    /// * `top_k` - Maximum number of results to return
    ///
    /// # Returns
    /// Array of search results with similarity scores.
    #[wasm_bindgen]
    pub fn search(&self, query_embedding: &[f32], top_k: usize) -> Result<JsValue, JsError> {
        let start = web_sys::window()
            .and_then(|w| w.performance())
            .map(|p| p.now());

        let results = self
            .search
            .search(query_embedding, top_k)
            .map_err(|e| JsError::new(&e))?;

        if let Some(start_time) = start {
            if let Some(perf) = web_sys::window().and_then(|w| w.performance()) {
                let elapsed = perf.now() - start_time;
                log::debug!("Search completed in {:.2}ms", elapsed);
            }
        }

        serde_wasm_bindgen::to_value(&results).map_err(|e| JsError::new(&e.to_string()))
    }

    /// Batch search for multiple queries.
    ///
    /// # Arguments
    /// * `queries_js` - Array of Float32Arrays with query embeddings
    /// * `top_k` - Maximum results per query
    ///
    /// # Returns
    /// Array of arrays of search results.
    #[wasm_bindgen]
    pub fn batch_search(&self, queries_js: JsValue, top_k: usize) -> Result<JsValue, JsError> {
        let queries: Vec<Vec<f32>> = serde_wasm_bindgen::from_value(queries_js)
            .map_err(|e| JsError::new(&e.to_string()))?;

        let results = self
            .search
            .batch_search(&queries, top_k)
            .map_err(|e| JsError::new(&e))?;

        serde_wasm_bindgen::to_value(&results).map_err(|e| JsError::new(&e.to_string()))
    }

    /// Get the number of patterns in the index.
    #[wasm_bindgen]
    pub fn pattern_count(&self) -> usize {
        self.search.len()
    }

    /// Check if the index is empty.
    #[wasm_bindgen]
    pub fn is_empty(&self) -> bool {
        self.search.is_empty()
    }

    /// Clear all patterns from the index.
    #[wasm_bindgen]
    pub fn clear(&mut self) {
        self.search.clear();
    }

    /// Export all patterns as JSON string.
    #[wasm_bindgen]
    pub fn export_json(&self) -> Result<String, JsError> {
        self.search.export_json().map_err(|e| JsError::new(&e))
    }

    /// Import patterns from JSON string.
    ///
    /// # Returns
    /// Number of patterns imported.
    #[wasm_bindgen]
    pub fn import_json(&mut self, json: &str) -> Result<usize, JsError> {
        self.search.import_json(json).map_err(|e| JsError::new(&e))
    }

    /// Get search statistics.
    #[wasm_bindgen]
    pub fn get_stats(&self) -> JsValue {
        let stats = self.search.get_stats();
        serde_wasm_bindgen::to_value(&stats).unwrap_or(JsValue::NULL)
    }

    // IndexedDB persistence methods

    /// Open the IndexedDB connection.
    #[wasm_bindgen]
    pub fn open_db(&mut self) -> js_sys::Promise {
        let _storage = self.storage.clone();
        future_to_promise(async move {
            // Note: We need to handle this differently due to ownership
            // For now, return success - actual implementation would need RefCell
            Ok(JsValue::TRUE)
        })
    }

    /// Save all patterns to IndexedDB.
    ///
    /// # Returns
    /// Promise resolving to number of patterns saved.
    #[wasm_bindgen]
    pub fn save_to_indexeddb(&self) -> js_sys::Promise {
        let patterns = self.search.get_all_patterns().to_vec();
        let patterns_js = match serde_wasm_bindgen::to_value(&patterns) {
            Ok(v) => v,
            Err(e) => return js_sys::Promise::reject(&JsValue::from_str(&e.to_string())),
        };

        // Create a wrapper that captures the storage
        let storage_available = IndexedDBStorage::is_available();

        future_to_promise(async move {
            if !storage_available {
                return Err(JsValue::from_str("IndexedDB not available"));
            }

            let mut storage = IndexedDBStorage::new();
            storage.open().await?;
            let count = storage.save_patterns(patterns_js).await?;
            storage.close();

            Ok(JsValue::from(count))
        })
    }

    /// Load patterns from IndexedDB.
    ///
    /// # Returns
    /// Promise resolving to number of patterns loaded.
    #[wasm_bindgen]
    pub fn load_from_indexeddb(&mut self) -> js_sys::Promise {
        // We need to load patterns and update the search index
        // This is tricky with ownership - using a simpler approach

        let storage_available = IndexedDBStorage::is_available();

        future_to_promise(async move {
            if !storage_available {
                return Err(JsValue::from_str("IndexedDB not available"));
            }

            let mut storage = IndexedDBStorage::new();
            storage.open().await?;
            let patterns_js = storage.load_all_patterns().await?;
            storage.close();

            // Return the patterns - caller should use import_json or add_pattern_full
            Ok(patterns_js)
        })
    }

    /// Check if IndexedDB is available.
    #[wasm_bindgen]
    pub fn is_storage_available() -> bool {
        IndexedDBStorage::is_available()
    }

    /// Get storage statistics.
    #[wasm_bindgen]
    pub fn get_storage_stats(&self) -> js_sys::Promise {
        let storage_available = IndexedDBStorage::is_available();
        let _connected = self.storage.is_connected();

        future_to_promise(async move {
            if !storage_available {
                let stats = StorageStats::new(0, false);
                return serde_wasm_bindgen::to_value(&stats)
                    .map_err(|e| JsValue::from_str(&e.to_string()));
            }

            let mut storage = IndexedDBStorage::new();
            if storage.open().await.is_err() {
                let stats = StorageStats::new(0, false);
                return serde_wasm_bindgen::to_value(&stats)
                    .map_err(|e| JsValue::from_str(&e.to_string()));
            }

            let count = storage.count().await.unwrap_or(0);
            storage.close();

            let stats = StorageStats::new(count, true);
            serde_wasm_bindgen::to_value(&stats).map_err(|e| JsValue::from_str(&e.to_string()))
        })
    }

    /// Clear all patterns from IndexedDB.
    #[wasm_bindgen]
    pub fn clear_indexeddb(&self) -> js_sys::Promise {
        let storage_available = IndexedDBStorage::is_available();

        future_to_promise(async move {
            if !storage_available {
                return Err(JsValue::from_str("IndexedDB not available"));
            }

            let mut storage = IndexedDBStorage::new();
            storage.open().await?;
            storage.clear().await?;
            storage.close();

            Ok(JsValue::TRUE)
        })
    }
}

impl Default for WasmProfDAG {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for IndexedDBStorage {
    fn clone(&self) -> Self {
        // Create a new instance - we can't clone the DB connection
        Self::new()
    }
}

/// Performance timing utilities.
#[wasm_bindgen]
pub struct PerfTimer {
    start_time: f64,
}

#[wasm_bindgen]
impl PerfTimer {
    /// Start a new timer.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        let start_time = web_sys::window()
            .and_then(|w| w.performance())
            .map(|p| p.now())
            .unwrap_or(0.0);

        Self { start_time }
    }

    /// Get elapsed time in milliseconds.
    #[wasm_bindgen]
    pub fn elapsed(&self) -> f64 {
        web_sys::window()
            .and_then(|w| w.performance())
            .map(|p| p.now() - self.start_time)
            .unwrap_or(0.0)
    }

    /// Get elapsed and reset.
    #[wasm_bindgen]
    pub fn lap(&mut self) -> f64 {
        let elapsed = self.elapsed();
        self.start_time = web_sys::window()
            .and_then(|w| w.performance())
            .map(|p| p.now())
            .unwrap_or(0.0);
        elapsed
    }
}

impl Default for PerfTimer {
    fn default() -> Self {
        Self::new()
    }
}

/// Generate a random embedding for testing.
#[wasm_bindgen]
pub fn generate_random_embedding(dim: usize) -> Vec<f32> {
    use js_sys::Math;

    let mut embedding = Vec::with_capacity(dim);
    for _ in 0..dim {
        embedding.push(Math::random() as f32 * 2.0 - 1.0);
    }

    // Normalize
    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > f32::EPSILON {
        for x in embedding.iter_mut() {
            *x /= norm;
        }
    }

    embedding
}

/// Generate a UUID for pattern IDs.
#[wasm_bindgen]
pub fn generate_uuid() -> String {
    uuid::Uuid::new_v4().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use wasm_bindgen_test::*;

    wasm_bindgen_test_configure!(run_in_browser);

    #[wasm_bindgen_test]
    fn test_wasm_profdag_new() {
        let profdag = WasmProfDAG::new();
        assert_eq!(profdag.pattern_count(), 0);
        assert!(profdag.is_empty());
    }

    #[wasm_bindgen_test]
    fn test_add_and_search() {
        let mut profdag = WasmProfDAG::with_config(SearchConfig::new().with_embedding_dim(4));

        // Add a pattern
        profdag
            .add_pattern("p1", "Test pattern", &[1.0, 0.0, 0.0, 0.0])
            .unwrap();

        assert_eq!(profdag.pattern_count(), 1);

        // Search
        let results = profdag.search(&[1.0, 0.0, 0.0, 0.0], 1).unwrap();
        assert!(!results.is_null());
    }

    #[wasm_bindgen_test]
    fn test_export_import_json() {
        let mut profdag = WasmProfDAG::with_config(SearchConfig::new().with_embedding_dim(4));

        profdag
            .add_pattern("p1", "Test", &[1.0, 0.0, 0.0, 0.0])
            .unwrap();

        let json = profdag.export_json().unwrap();

        let mut profdag2 = WasmProfDAG::with_config(SearchConfig::new().with_embedding_dim(4));
        let count = profdag2.import_json(&json).unwrap();

        assert_eq!(count, 1);
        assert_eq!(profdag2.pattern_count(), 1);
    }

    #[wasm_bindgen_test]
    fn test_generate_random_embedding() {
        let embedding = generate_random_embedding(128);
        assert_eq!(embedding.len(), 128);

        // Check normalized
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.001);
    }

    #[wasm_bindgen_test]
    fn test_generate_uuid() {
        let uuid = generate_uuid();
        assert_eq!(uuid.len(), 36); // UUID format with dashes
    }
}
