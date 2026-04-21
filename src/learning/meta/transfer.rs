//! Transfer Learning Engine
//!
//! Enables knowledge transfer between related domains by identifying
//! applicable patterns and adapting them to new contexts.

use std::collections::HashMap;
use std::sync::Arc;

use parking_lot::RwLock;
use tracing::info;

use super::types::{DomainTransfer, MetaLearningStats, PatternMapping};

/// Transfer learning engine for cross-domain pattern adaptation
pub struct TransferEngine {
    /// Domain transfer mappings (source_domain:target_domain -> DomainTransfer)
    transfers: Arc<RwLock<HashMap<String, DomainTransfer>>>,
    /// Statistics
    stats: Arc<RwLock<MetaLearningStats>>,
}

impl TransferEngine {
    /// Create a new transfer engine
    pub fn new() -> Self {
        Self {
            transfers: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(MetaLearningStats::default())),
        }
    }

    /// Create a key for the transfer map
    fn transfer_key(source: &str, target: &str) -> String {
        format!("{}:{}", source, target)
    }

    /// Get or create a domain transfer record
    pub fn get_or_create_transfer(&self, source: &str, target: &str) -> DomainTransfer {
        let key = Self::transfer_key(source, target);
        let mut transfers = self.transfers.write();

        transfers
            .entry(key)
            .or_insert_with(|| DomainTransfer::new(source, target))
            .clone()
    }

    /// Get transfer coefficient between domains
    pub fn get_transfer_coefficient(&self, source: &str, target: &str) -> f64 {
        let key = Self::transfer_key(source, target);
        self.transfers
            .read()
            .get(&key)
            .map(|t| t.transfer_coefficient)
            .unwrap_or(0.5) // Default prior
    }

    /// Record a transfer outcome
    pub fn record_transfer(
        &self,
        source: &str,
        target: &str,
        success: bool,
        source_pattern_id: Option<&str>,
        target_pattern_id: Option<&str>,
        similarity: Option<f64>,
    ) {
        let key = Self::transfer_key(source, target);
        let mut transfers = self.transfers.write();

        let transfer = transfers
            .entry(key)
            .or_insert_with(|| DomainTransfer::new(source, target));

        let mapping = match (source_pattern_id, target_pattern_id, similarity) {
            (Some(src), Some(tgt), Some(sim)) => Some(PatternMapping::new(src, tgt, sim)),
            _ => None,
        };

        transfer.record_transfer(success, mapping);

        // Update stats
        let mut stats = self.stats.write();
        if success {
            stats.successful_transfers += 1;
        } else {
            stats.failed_transfers += 1;
        }

        info!(
            source = %source,
            target = %target,
            success = success,
            new_coefficient = transfer.transfer_coefficient,
            "Recorded domain transfer"
        );
    }

    /// Find domains similar to a given domain
    pub fn find_related_domains(&self, domain: &str) -> Vec<(String, f64)> {
        let transfers = self.transfers.read();
        let mut related = Vec::new();

        for transfer in transfers.values() {
            if transfer.source_domain == domain {
                related.push((transfer.target_domain.clone(), transfer.transfer_coefficient));
            }
            if transfer.target_domain == domain {
                related.push((transfer.source_domain.clone(), transfer.transfer_coefficient));
            }
        }

        // Sort by transfer coefficient (highest first)
        related.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Deduplicate
        let mut seen = std::collections::HashSet::new();
        related.retain(|(d, _)| seen.insert(d.clone()));

        related
    }

    /// Suggest patterns from related domains that might apply
    pub fn suggest_transfer_candidates(
        &self,
        target_domain: &str,
        min_coefficient: f64,
    ) -> Vec<(String, f64)> {
        let related = self.find_related_domains(target_domain);

        related
            .into_iter()
            .filter(|(_, coef)| *coef >= min_coefficient)
            .collect()
    }

    /// Calculate domain similarity based on name hierarchy
    ///
    /// Example: "rust.async" and "rust.error" share "rust" prefix
    pub fn calculate_domain_similarity(domain1: &str, domain2: &str) -> f64 {
        if domain1 == domain2 {
            return 1.0;
        }

        let parts1: Vec<&str> = domain1.split('.').collect();
        let parts2: Vec<&str> = domain2.split('.').collect();

        let common_prefix = parts1
            .iter()
            .zip(parts2.iter())
            .take_while(|(a, b)| a == b)
            .count();

        let max_len = parts1.len().max(parts2.len());

        if max_len == 0 {
            0.0
        } else {
            common_prefix as f64 / max_len as f64
        }
    }

    /// Initialize common domain relationships
    pub fn initialize_common_transfers(&self) {
        // Programming language transfers
        let lang_pairs = [
            ("rust", "cpp", 0.6),
            ("rust", "go", 0.5),
            ("python", "javascript", 0.4),
            ("typescript", "javascript", 0.9),
            ("java", "kotlin", 0.8),
            ("java", "scala", 0.7),
        ];

        for (src, tgt, coef) in lang_pairs {
            let mut transfers = self.transfers.write();
            let key = Self::transfer_key(src, tgt);
            if !transfers.contains_key(&key) {
                let mut transfer = DomainTransfer::new(src, tgt);
                transfer.transfer_coefficient = coef;
                transfers.insert(key.clone(), transfer);

                // Also add reverse direction with same coefficient
                let reverse_key = Self::transfer_key(tgt, src);
                if !transfers.contains_key(&reverse_key) {
                    let mut reverse = DomainTransfer::new(tgt, src);
                    reverse.transfer_coefficient = coef;
                    transfers.insert(reverse_key, reverse);
                }
            }
        }

        // Concept transfers (apply across languages)
        let concept_pairs = [
            ("async", "concurrent", 0.7),
            ("error", "exception", 0.8),
            ("testing", "validation", 0.6),
            ("api", "interface", 0.7),
        ];

        for (src, tgt, coef) in concept_pairs {
            let mut transfers = self.transfers.write();
            let key = Self::transfer_key(src, tgt);
            if !transfers.contains_key(&key) {
                let mut transfer = DomainTransfer::new(src, tgt);
                transfer.transfer_coefficient = coef;
                transfers.insert(key, transfer);
            }
        }

        info!("Initialized common domain transfer relationships");
    }

    /// Get all transfers
    pub fn all_transfers(&self) -> Vec<DomainTransfer> {
        self.transfers.read().values().cloned().collect()
    }

    /// Get statistics
    pub fn stats(&self) -> MetaLearningStats {
        self.stats.read().clone()
    }

    /// Load transfer data
    pub fn load_transfers(&self, data: Vec<DomainTransfer>) {
        let mut transfers = self.transfers.write();
        for transfer in data {
            let key = Self::transfer_key(&transfer.source_domain, &transfer.target_domain);
            transfers.insert(key, transfer);
        }
    }

    /// Export transfer data
    pub fn export_transfers(&self) -> Vec<DomainTransfer> {
        self.transfers.read().values().cloned().collect()
    }
}

impl Default for TransferEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_domain_similarity() {
        assert_eq!(TransferEngine::calculate_domain_similarity("rust", "rust"), 1.0);
        assert!(TransferEngine::calculate_domain_similarity("rust.async", "rust.error") > 0.0);
        assert_eq!(TransferEngine::calculate_domain_similarity("rust", "python"), 0.0);
    }

    #[test]
    fn test_transfer_recording() {
        let engine = TransferEngine::new();

        engine.record_transfer("rust", "go", true, None, None, None);
        engine.record_transfer("rust", "go", true, None, None, None);
        engine.record_transfer("rust", "go", false, None, None, None);

        let coef = engine.get_transfer_coefficient("rust", "go");
        // 2 success, 1 failure -> (2+1)/(2+1+1+1) = 3/5 = 0.6
        assert!(coef > 0.5 && coef < 0.8);
    }

    #[test]
    fn test_find_related_domains() {
        let engine = TransferEngine::new();
        engine.initialize_common_transfers();

        let related = engine.find_related_domains("rust");
        assert!(!related.is_empty());
        assert!(related.iter().any(|(d, _)| d == "cpp" || d == "go"));
    }
}
