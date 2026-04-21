//! Storage for LoRA adapter files.
//!
//! Saves and loads LoRA adapters as JSON files on disk,
//! organized by domain name.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use super::adapter::LoraAdapter;
use crate::ml::{MlError, MlResult};

/// Metadata about a stored adapter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredAdapter {
    /// Domain the adapter was trained for.
    pub domain: String,
    /// Path to the adapter file on disk.
    pub path: PathBuf,
    /// Size of the adapter file in bytes.
    pub size_bytes: u64,
    /// When the adapter was trained.
    pub trained_at: String,
    /// Number of training iterations completed.
    pub iterations: u32,
    /// Final training loss.
    pub final_loss: f32,
}

/// Storage manager for LoRA adapters.
///
/// Persists adapters as JSON files in a directory, with one file per domain.
pub struct LoraStorage {
    base_dir: PathBuf,
}

impl LoraStorage {
    /// Create a new storage manager.
    ///
    /// The base directory will be created if it does not exist.
    pub fn new(base_dir: impl Into<PathBuf>) -> Self {
        Self {
            base_dir: base_dir.into(),
        }
    }

    /// Save an adapter to disk as JSON.
    ///
    /// Returns the path to the saved file.
    pub fn save(&self, adapter: &LoraAdapter) -> MlResult<PathBuf> {
        // Ensure directory exists
        std::fs::create_dir_all(&self.base_dir)?;

        let path = self.adapter_path(&adapter.domain);
        let json = serde_json::to_string_pretty(adapter).map_err(|e| {
            MlError::Migration(format!("Failed to serialize adapter: {}", e))
        })?;

        std::fs::write(&path, json)?;
        Ok(path)
    }

    /// Load an adapter for a domain.
    pub fn load(&self, domain: &str) -> MlResult<LoraAdapter> {
        let path = self.adapter_path(domain);
        if !path.exists() {
            return Err(MlError::ModelLoad {
                path: path.display().to_string(),
                reason: format!("No adapter found for domain '{}'", domain),
            });
        }

        let json = std::fs::read_to_string(&path)?;
        let adapter: LoraAdapter = serde_json::from_str(&json).map_err(|e| {
            MlError::Migration(format!("Failed to deserialize adapter: {}", e))
        })?;

        Ok(adapter)
    }

    /// List all stored adapters.
    pub fn list(&self) -> MlResult<Vec<StoredAdapter>> {
        if !self.base_dir.exists() {
            return Ok(Vec::new());
        }

        let mut adapters = Vec::new();

        for entry in std::fs::read_dir(&self.base_dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.extension().and_then(|s| s.to_str()) != Some("json") {
                continue;
            }

            let filename = path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("");

            if !filename.starts_with("lora_") {
                continue;
            }

            // Try to load the adapter to get metadata
            let json = match std::fs::read_to_string(&path) {
                Ok(j) => j,
                Err(_) => continue,
            };

            let adapter: LoraAdapter = match serde_json::from_str(&json) {
                Ok(a) => a,
                Err(_) => continue,
            };

            let metadata = entry.metadata()?;

            adapters.push(StoredAdapter {
                domain: adapter.domain,
                path: path.clone(),
                size_bytes: metadata.len(),
                trained_at: adapter.trained_at,
                iterations: adapter.iterations,
                final_loss: adapter.final_loss,
            });
        }

        // Sort by domain for consistent ordering
        adapters.sort_by(|a, b| a.domain.cmp(&b.domain));

        Ok(adapters)
    }

    /// Check if an adapter exists for a domain.
    pub fn exists(&self, domain: &str) -> bool {
        self.adapter_path(domain).exists()
    }

    /// Delete an adapter for a domain.
    pub fn delete(&self, domain: &str) -> MlResult<()> {
        let path = self.adapter_path(domain);
        if path.exists() {
            std::fs::remove_file(&path)?;
        }
        Ok(())
    }

    /// Get the file path for a domain's adapter.
    ///
    /// Dots in domain names are replaced with underscores:
    /// `"rust.async"` -> `"lora_rust_async.json"`
    fn adapter_path(&self, domain: &str) -> PathBuf {
        let sanitized = domain.replace('.', "_").replace(' ', "_");
        self.base_dir.join(format!("lora_{}.json", sanitized))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ml::lora::LoraConfig;

    fn make_test_adapter(domain: &str) -> LoraAdapter {
        let config = LoraConfig {
            base_dim: 8,
            rank: 2,
            ..Default::default()
        };
        let mut adapter = LoraAdapter::new(domain, config);
        adapter.iterations = 10;
        adapter.final_loss = 0.25;
        adapter.trained_at = "2026-02-07T00:00:00Z".to_string();
        adapter
    }

    #[test]
    fn test_storage_save_and_load() {
        let dir = tempfile::tempdir().unwrap();
        let storage = LoraStorage::new(dir.path());

        let adapter = make_test_adapter("rust");
        let saved_path = storage.save(&adapter).unwrap();

        assert!(saved_path.exists(), "saved file should exist");

        let loaded = storage.load("rust").unwrap();
        assert_eq!(loaded.domain, "rust");
        assert_eq!(loaded.iterations, 10);
        assert!((loaded.final_loss - 0.25).abs() < f32::EPSILON);
        assert_eq!(loaded.matrix_a.len(), adapter.matrix_a.len());
        assert_eq!(loaded.matrix_b.len(), adapter.matrix_b.len());
    }

    #[test]
    fn test_storage_exists() {
        let dir = tempfile::tempdir().unwrap();
        let storage = LoraStorage::new(dir.path());

        assert!(!storage.exists("rust"), "should not exist before save");

        let adapter = make_test_adapter("rust");
        storage.save(&adapter).unwrap();

        assert!(storage.exists("rust"), "should exist after save");
        assert!(
            !storage.exists("python"),
            "different domain should not exist"
        );
    }

    #[test]
    fn test_storage_list() {
        let dir = tempfile::tempdir().unwrap();
        let storage = LoraStorage::new(dir.path());

        // Empty list initially
        let list = storage.list().unwrap();
        assert!(list.is_empty());

        // Save two adapters
        storage.save(&make_test_adapter("rust")).unwrap();
        storage.save(&make_test_adapter("python")).unwrap();

        let list = storage.list().unwrap();
        assert_eq!(list.len(), 2);

        let domains: Vec<&str> = list.iter().map(|a| a.domain.as_str()).collect();
        assert!(domains.contains(&"rust"));
        assert!(domains.contains(&"python"));
    }

    #[test]
    fn test_storage_delete() {
        let dir = tempfile::tempdir().unwrap();
        let storage = LoraStorage::new(dir.path());

        let adapter = make_test_adapter("rust");
        storage.save(&adapter).unwrap();
        assert!(storage.exists("rust"));

        storage.delete("rust").unwrap();
        assert!(!storage.exists("rust"));

        // Deleting a nonexistent adapter should not error
        let result = storage.delete("nonexistent");
        assert!(result.is_ok());
    }

    #[test]
    fn test_storage_load_nonexistent() {
        let dir = tempfile::tempdir().unwrap();
        let storage = LoraStorage::new(dir.path());

        let result = storage.load("nonexistent");
        assert!(result.is_err(), "loading nonexistent adapter should fail");
    }

    #[test]
    fn test_adapter_path_sanitization() {
        let dir = tempfile::tempdir().unwrap();
        let storage = LoraStorage::new(dir.path());

        // Domain with dots should be sanitized
        let adapter = make_test_adapter("rust.async");
        let path = storage.save(&adapter).unwrap();

        assert!(
            path.file_name()
                .unwrap()
                .to_str()
                .unwrap()
                .contains("lora_rust_async"),
            "dots should be replaced with underscores"
        );

        // Should be loadable by original domain name
        let loaded = storage.load("rust.async").unwrap();
        assert_eq!(loaded.domain, "rust.async");
    }
}
