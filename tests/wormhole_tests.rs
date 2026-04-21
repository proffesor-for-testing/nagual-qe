//! Wormhole Tests - Phase 3 (Learning Layer)
//!
//! Comprehensive test suite for wormhole edge functionality.
//! Wormholes are non-local connections that bridge distant graph regions
//! based on co-access patterns, enabling faster traversal.
//!
//! # Wormhole Mechanics
//! - Created when co-access threshold is exceeded
//! - Strength = frequency / (frequency + decay_constant)
//! - Decay after 30 days unused
//! - Max wormholes per node limit enforced
//! - Provides >50% traversal reduction for distant patterns
//!
//! # Test Categories
//! - Creation based on co-access threshold
//! - Strength calculation formula verification
//! - Decay mechanics after 30 days
//! - Max wormholes per node limit
//! - Traversal savings calculation
//! - Audit logging verification
//! - Integration with AutoEdgeCreator

use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};

use chrono::{DateTime, Duration as ChronoDuration, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

mod common;
use common::{cosine_similarity, normalized_embedding, similar_embeddings};

// ============================================================================
// Wormhole Configuration
// ============================================================================

/// Configuration for wormhole creation and management.
#[derive(Debug, Clone)]
pub struct WormholeConfig {
    /// Minimum co-access count before wormhole creation.
    pub co_access_threshold: u32,
    /// Decay constant for strength calculation (default: 10.0).
    pub decay_constant: f64,
    /// Days after which unused wormholes decay.
    pub decay_days: u32,
    /// Maximum wormholes per node.
    pub max_wormholes_per_node: usize,
    /// Minimum traversal savings required (0.0-1.0).
    pub min_traversal_savings: f64,
    /// Base strength for new wormholes.
    pub base_strength: f64,
}

impl Default for WormholeConfig {
    fn default() -> Self {
        Self {
            co_access_threshold: 3,
            decay_constant: 10.0,
            decay_days: 30,
            max_wormholes_per_node: 5,
            min_traversal_savings: 0.5,
            base_strength: 0.7,
        }
    }
}

// ============================================================================
// Wormhole Types
// ============================================================================

/// A wormhole edge connecting two distant nodes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Wormhole {
    pub id: String,
    pub source_id: String,
    pub target_id: String,
    pub strength: f64,
    pub co_access_count: u32,
    pub reason: String,
    pub created_at: DateTime<Utc>,
    pub last_used_at: DateTime<Utc>,
    pub traversal_count: u32,
    pub traversal_savings_ms: u64,
    pub metadata: serde_json::Value,
}

impl Wormhole {
    pub fn new(
        source_id: impl Into<String>,
        target_id: impl Into<String>,
        co_access_count: u32,
        reason: impl Into<String>,
        config: &WormholeConfig,
    ) -> Self {
        let strength = Self::calculate_strength(co_access_count, config);
        Self {
            id: Uuid::new_v4().to_string(),
            source_id: source_id.into(),
            target_id: target_id.into(),
            strength,
            co_access_count,
            reason: reason.into(),
            created_at: Utc::now(),
            last_used_at: Utc::now(),
            traversal_count: 0,
            traversal_savings_ms: 0,
            metadata: serde_json::json!({}),
        }
    }

    /// Calculate wormhole strength: frequency / (frequency + decay_constant).
    pub fn calculate_strength(frequency: u32, config: &WormholeConfig) -> f64 {
        let freq = frequency as f64;
        freq / (freq + config.decay_constant)
    }

    /// Update strength based on new co-access count.
    pub fn update_strength(&mut self, new_count: u32, config: &WormholeConfig) {
        self.co_access_count = new_count;
        self.strength = Self::calculate_strength(new_count, config);
    }

    /// Record a traversal and update statistics.
    pub fn record_traversal(&mut self, savings_ms: u64) {
        self.traversal_count += 1;
        self.traversal_savings_ms += savings_ms;
        self.last_used_at = Utc::now();
    }

    /// Check if wormhole should decay (unused for decay_days).
    pub fn should_decay(&self, config: &WormholeConfig) -> bool {
        let days_since_use = (Utc::now() - self.last_used_at).num_days();
        days_since_use >= config.decay_days as i64
    }

    /// Apply decay to wormhole strength.
    pub fn apply_decay(&mut self, decay_factor: f64) {
        self.strength *= decay_factor;
        self.strength = self.strength.clamp(0.0, 1.0);
    }

    /// Get average traversal savings.
    pub fn avg_traversal_savings(&self) -> f64 {
        if self.traversal_count > 0 {
            self.traversal_savings_ms as f64 / self.traversal_count as f64
        } else {
            0.0
        }
    }

    /// Check if wormhole is strong (>= 0.7).
    pub fn is_strong(&self) -> bool {
        self.strength >= 0.7
    }

    /// Check if wormhole is weak (< 0.3).
    pub fn is_weak(&self) -> bool {
        self.strength < 0.3
    }
}

/// Co-access record for tracking pattern pairs.
#[derive(Debug, Clone)]
pub struct CoAccessRecord {
    pub pattern_a: String,
    pub pattern_b: String,
    pub count: u32,
    pub first_accessed: DateTime<Utc>,
    pub last_accessed: DateTime<Utc>,
    pub sessions: HashSet<String>,
}

impl CoAccessRecord {
    pub fn new(pattern_a: impl Into<String>, pattern_b: impl Into<String>) -> Self {
        let now = Utc::now();
        Self {
            pattern_a: pattern_a.into(),
            pattern_b: pattern_b.into(),
            count: 0,  // Start at 0, increment will add 1
            first_accessed: now,
            last_accessed: now,
            sessions: HashSet::new(),
        }
    }

    pub fn increment(&mut self, session_id: Option<String>) {
        self.count += 1;
        self.last_accessed = Utc::now();
        if let Some(sid) = session_id {
            self.sessions.insert(sid);
        }
    }
}

/// Audit log entry for wormhole operations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WormholeAuditEntry {
    pub id: String,
    pub wormhole_id: String,
    pub operation: String,
    pub old_strength: Option<f64>,
    pub new_strength: Option<f64>,
    pub reason: String,
    pub timestamp: DateTime<Utc>,
}

/// Wormhole manager for creating and maintaining wormholes.
#[derive(Debug)]
pub struct WormholeManager {
    config: WormholeConfig,
    wormholes: HashMap<String, Wormhole>,
    co_access_records: HashMap<(String, String), CoAccessRecord>,
    wormholes_by_node: HashMap<String, Vec<String>>,
    audit_log: Vec<WormholeAuditEntry>,
}

impl WormholeManager {
    pub fn new(config: WormholeConfig) -> Self {
        Self {
            config,
            wormholes: HashMap::new(),
            co_access_records: HashMap::new(),
            wormholes_by_node: HashMap::new(),
            audit_log: Vec::new(),
        }
    }

    pub fn with_default_config() -> Self {
        Self::new(WormholeConfig::default())
    }

    /// Record a co-access between two patterns.
    pub fn record_co_access(
        &mut self,
        pattern_a: &str,
        pattern_b: &str,
        session_id: Option<String>,
    ) -> Option<Wormhole> {
        // Ensure consistent ordering
        let (ordered_a, ordered_b) = if pattern_a < pattern_b {
            (pattern_a.to_string(), pattern_b.to_string())
        } else {
            (pattern_b.to_string(), pattern_a.to_string())
        };

        let key = (ordered_a.clone(), ordered_b.clone());

        let record = self.co_access_records.entry(key.clone()).or_insert_with(|| {
            CoAccessRecord::new(&ordered_a, &ordered_b)
        });

        record.increment(session_id);

        // Extract values before further mutable borrows
        let current_count = record.count;
        let threshold = self.config.co_access_threshold;

        // Check if threshold is reached and no wormhole exists yet
        if current_count >= threshold {
            let wormhole_key = format!("{}_{}", ordered_a, ordered_b);
            if !self.wormholes.contains_key(&wormhole_key) {
                return self.create_wormhole(&ordered_a, &ordered_b, current_count);
            } else {
                // Update existing wormhole
                let (wh_id, old_strength, new_strength) = {
                    if let Some(wh) = self.wormholes.get_mut(&wormhole_key) {
                        let old_strength = wh.strength;
                        wh.update_strength(current_count, &self.config);
                        (wh.id.clone(), old_strength, wh.strength)
                    } else {
                        return None;
                    }
                };
                self.log_audit(&wh_id, "strength_updated", Some(old_strength), Some(new_strength), "Co-access count increased");
            }
        }

        None
    }

    /// Create a new wormhole between two patterns.
    fn create_wormhole(
        &mut self,
        source_id: &str,
        target_id: &str,
        co_access_count: u32,
    ) -> Option<Wormhole> {
        // Check max wormholes limit for source
        if let Some(source_wormholes) = self.wormholes_by_node.get(source_id) {
            if source_wormholes.len() >= self.config.max_wormholes_per_node {
                return None;
            }
        }

        // Check max wormholes limit for target
        if let Some(target_wormholes) = self.wormholes_by_node.get(target_id) {
            if target_wormholes.len() >= self.config.max_wormholes_per_node {
                return None;
            }
        }

        let reason = format!(
            "Co-accessed {} times (threshold: {})",
            co_access_count, self.config.co_access_threshold
        );

        let wormhole = Wormhole::new(source_id, target_id, co_access_count, &reason, &self.config);
        let wormhole_id = wormhole.id.clone();
        let wormhole_key = format!("{}_{}", source_id, target_id);

        // Store wormhole
        self.wormholes.insert(wormhole_key.clone(), wormhole.clone());

        // Track by node
        self.wormholes_by_node
            .entry(source_id.to_string())
            .or_insert_with(Vec::new)
            .push(wormhole_id.clone());

        self.wormholes_by_node
            .entry(target_id.to_string())
            .or_insert_with(Vec::new)
            .push(wormhole_id.clone());

        // Log creation
        self.log_audit(&wormhole_id, "created", None, Some(wormhole.strength), &reason);

        Some(wormhole)
    }

    /// Get wormholes for a node.
    pub fn get_wormholes_for_node(&self, node_id: &str) -> Vec<&Wormhole> {
        self.wormholes
            .values()
            .filter(|wh| wh.source_id == node_id || wh.target_id == node_id)
            .collect()
    }

    /// Record a traversal on a wormhole.
    pub fn record_traversal(&mut self, wormhole_key: &str, savings_ms: u64) -> bool {
        if let Some(wormhole) = self.wormholes.get_mut(wormhole_key) {
            wormhole.record_traversal(savings_ms);
            true
        } else {
            false
        }
    }

    /// Apply decay to unused wormholes.
    pub fn apply_decay(&mut self, decay_factor: f64) -> usize {
        let mut decayed_count = 0;
        let mut to_remove = Vec::new();

        for (key, wormhole) in self.wormholes.iter_mut() {
            if wormhole.should_decay(&self.config) {
                let old_strength = wormhole.strength;
                wormhole.apply_decay(decay_factor);
                self.audit_log.push(WormholeAuditEntry {
                    id: Uuid::new_v4().to_string(),
                    wormhole_id: wormhole.id.clone(),
                    operation: "decay_applied".to_string(),
                    old_strength: Some(old_strength),
                    new_strength: Some(wormhole.strength),
                    reason: format!("Unused for {} days", self.config.decay_days),
                    timestamp: Utc::now(),
                });
                decayed_count += 1;

                // Mark for removal if too weak
                if wormhole.strength < 0.1 {
                    to_remove.push(key.clone());
                }
            }
        }

        // Remove very weak wormholes
        for key in to_remove {
            if let Some(wh) = self.wormholes.remove(&key) {
                self.log_audit(&wh.id, "removed", Some(wh.strength), None, "Strength below threshold after decay");
            }
        }

        decayed_count
    }

    /// Get audit log.
    pub fn audit_log(&self) -> &[WormholeAuditEntry] {
        &self.audit_log
    }

    /// Log an audit entry.
    fn log_audit(
        &mut self,
        wormhole_id: &str,
        operation: &str,
        old_strength: Option<f64>,
        new_strength: Option<f64>,
        reason: &str,
    ) {
        self.audit_log.push(WormholeAuditEntry {
            id: Uuid::new_v4().to_string(),
            wormhole_id: wormhole_id.to_string(),
            operation: operation.to_string(),
            old_strength,
            new_strength,
            reason: reason.to_string(),
            timestamp: Utc::now(),
        });
    }

    /// Get statistics.
    pub fn stats(&self) -> WormholeStats {
        let total_wormholes = self.wormholes.len();
        let strong_wormholes = self.wormholes.values().filter(|w| w.is_strong()).count();
        let weak_wormholes = self.wormholes.values().filter(|w| w.is_weak()).count();
        let total_traversals: u32 = self.wormholes.values().map(|w| w.traversal_count).sum();
        let total_savings_ms: u64 = self.wormholes.values().map(|w| w.traversal_savings_ms).sum();

        WormholeStats {
            total_wormholes,
            strong_wormholes,
            weak_wormholes,
            total_traversals,
            total_savings_ms,
            avg_strength: if total_wormholes > 0 {
                self.wormholes.values().map(|w| w.strength).sum::<f64>() / total_wormholes as f64
            } else {
                0.0
            },
        }
    }
}

/// Wormhole statistics.
#[derive(Debug, Clone)]
pub struct WormholeStats {
    pub total_wormholes: usize,
    pub strong_wormholes: usize,
    pub weak_wormholes: usize,
    pub total_traversals: u32,
    pub total_savings_ms: u64,
    pub avg_strength: f64,
}

// ============================================================================
// Graph Traversal Simulator
// ============================================================================

/// Simple graph for traversal simulation.
#[derive(Debug, Default)]
pub struct SimulatedGraph {
    nodes: HashSet<String>,
    edges: HashMap<String, Vec<(String, f64)>>,
    distances: HashMap<(String, String), usize>,
}

impl SimulatedGraph {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_node(&mut self, id: &str) {
        self.nodes.insert(id.to_string());
    }

    pub fn add_edge(&mut self, from: &str, to: &str, weight: f64) {
        self.nodes.insert(from.to_string());
        self.nodes.insert(to.to_string());
        self.edges
            .entry(from.to_string())
            .or_insert_with(Vec::new)
            .push((to.to_string(), weight));
    }

    /// Calculate shortest path distance using BFS.
    pub fn shortest_path_distance(&self, from: &str, to: &str) -> Option<usize> {
        if from == to {
            return Some(0);
        }

        let mut visited = HashSet::new();
        let mut queue = vec![(from.to_string(), 0usize)];

        while let Some((current, dist)) = queue.pop() {
            if current == to {
                return Some(dist);
            }

            if visited.contains(&current) {
                continue;
            }
            visited.insert(current.clone());

            if let Some(neighbors) = self.edges.get(&current) {
                for (neighbor, _) in neighbors {
                    if !visited.contains(neighbor) {
                        queue.insert(0, (neighbor.clone(), dist + 1));
                    }
                }
            }
        }

        None
    }

    /// Calculate traversal savings with a wormhole.
    pub fn calculate_traversal_savings(&self, wormhole: &Wormhole) -> f64 {
        let without_wormhole = self.shortest_path_distance(&wormhole.source_id, &wormhole.target_id);
        let with_wormhole = 1; // Wormhole provides direct connection

        if let Some(normal_dist) = without_wormhole {
            if normal_dist > 1 {
                let savings = (normal_dist - with_wormhole) as f64 / normal_dist as f64;
                return savings;
            }
        }

        0.0
    }
}

// ============================================================================
// Wormhole Creation Tests
// ============================================================================

mod wormhole_creation_tests {
    use super::*;

    #[test]
    fn test_wormhole_created_on_threshold() {
        let mut manager = WormholeManager::with_default_config();

        // Record co-accesses below threshold
        assert!(manager.record_co_access("pat_1", "pat_2", None).is_none());
        assert!(manager.record_co_access("pat_1", "pat_2", None).is_none());

        // Third access should trigger wormhole creation
        let wormhole = manager.record_co_access("pat_1", "pat_2", None);
        assert!(wormhole.is_some());

        let wh = wormhole.unwrap();
        assert_eq!(wh.source_id, "pat_1");
        assert_eq!(wh.target_id, "pat_2");
        assert!(wh.strength > 0.0);
    }

    #[test]
    fn test_wormhole_not_created_below_threshold() {
        let config = WormholeConfig {
            co_access_threshold: 5,
            ..Default::default()
        };
        let mut manager = WormholeManager::new(config);

        for _ in 0..4 {
            assert!(manager.record_co_access("pat_a", "pat_b", None).is_none());
        }

        // Should not have any wormholes yet
        assert_eq!(manager.stats().total_wormholes, 0);
    }

    #[test]
    fn test_wormhole_with_custom_threshold() {
        let config = WormholeConfig {
            co_access_threshold: 2,
            ..Default::default()
        };
        let mut manager = WormholeManager::new(config);

        // First access - no wormhole
        assert!(manager.record_co_access("pat_x", "pat_y", None).is_none());

        // Second access - should create wormhole
        let wormhole = manager.record_co_access("pat_x", "pat_y", None);
        assert!(wormhole.is_some());
    }

    #[test]
    fn test_multiple_wormholes_from_different_pairs() {
        let mut manager = WormholeManager::with_default_config();

        // Create wormhole between pat_1 and pat_2
        for _ in 0..3 {
            manager.record_co_access("pat_1", "pat_2", None);
        }

        // Create wormhole between pat_1 and pat_3
        for _ in 0..3 {
            manager.record_co_access("pat_1", "pat_3", None);
        }

        let wormholes = manager.get_wormholes_for_node("pat_1");
        assert_eq!(wormholes.len(), 2);
    }

    #[test]
    fn test_wormhole_consistent_ordering() {
        let mut manager = WormholeManager::with_default_config();

        // Access in reverse order should still create same wormhole
        manager.record_co_access("pat_b", "pat_a", None);
        manager.record_co_access("pat_a", "pat_b", None);
        let wormhole = manager.record_co_access("pat_b", "pat_a", None);

        assert!(wormhole.is_some());
        let wh = wormhole.unwrap();
        // Should be consistently ordered
        assert!(wh.source_id < wh.target_id);
    }
}

// ============================================================================
// Strength Calculation Tests
// ============================================================================

mod strength_calculation_tests {
    use super::*;

    #[test]
    fn test_strength_formula() {
        let config = WormholeConfig::default();

        // Test: strength = frequency / (frequency + decay_constant)
        // With decay_constant = 10.0

        // frequency = 3, strength = 3 / (3 + 10) = 0.2307...
        let strength_3 = Wormhole::calculate_strength(3, &config);
        assert!((strength_3 - 0.2307692).abs() < 0.0001);

        // frequency = 10, strength = 10 / (10 + 10) = 0.5
        let strength_10 = Wormhole::calculate_strength(10, &config);
        assert!((strength_10 - 0.5).abs() < 0.0001);

        // frequency = 20, strength = 20 / (20 + 10) = 0.6666...
        let strength_20 = Wormhole::calculate_strength(20, &config);
        assert!((strength_20 - 0.6666666).abs() < 0.0001);
    }

    #[test]
    fn test_strength_increases_with_frequency() {
        let config = WormholeConfig::default();

        let s1 = Wormhole::calculate_strength(1, &config);
        let s5 = Wormhole::calculate_strength(5, &config);
        let s10 = Wormhole::calculate_strength(10, &config);
        let s100 = Wormhole::calculate_strength(100, &config);

        assert!(s1 < s5);
        assert!(s5 < s10);
        assert!(s10 < s100);
    }

    #[test]
    fn test_strength_bounded_below_one() {
        let config = WormholeConfig::default();

        // Even with very high frequency, strength should approach but not exceed 1.0
        let strength = Wormhole::calculate_strength(1000000, &config);
        assert!(strength < 1.0);
        assert!(strength > 0.99);
    }

    #[test]
    fn test_strength_with_different_decay_constants() {
        let freq = 10;

        // Lower decay constant = higher strength
        let config_low = WormholeConfig {
            decay_constant: 5.0,
            ..Default::default()
        };
        let strength_low = Wormhole::calculate_strength(freq, &config_low);

        // Higher decay constant = lower strength
        let config_high = WormholeConfig {
            decay_constant: 20.0,
            ..Default::default()
        };
        let strength_high = Wormhole::calculate_strength(freq, &config_high);

        assert!(strength_low > strength_high);
    }

    #[test]
    fn test_strength_update() {
        let config = WormholeConfig::default();
        let mut wormhole = Wormhole::new("a", "b", 3, "test", &config);

        let initial_strength = wormhole.strength;

        // Increase co-access count
        wormhole.update_strength(10, &config);

        assert!(wormhole.strength > initial_strength);
        assert_eq!(wormhole.co_access_count, 10);
    }
}

// ============================================================================
// Decay Tests
// ============================================================================

mod decay_tests {
    use super::*;

    #[test]
    fn test_wormhole_should_decay_after_30_days() {
        let config = WormholeConfig::default();
        let mut wormhole = Wormhole::new("a", "b", 5, "test", &config);

        // Just created - should not decay
        assert!(!wormhole.should_decay(&config));

        // Simulate 31 days ago
        wormhole.last_used_at = Utc::now() - ChronoDuration::days(31);
        assert!(wormhole.should_decay(&config));
    }

    #[test]
    fn test_wormhole_no_decay_if_used_recently() {
        let config = WormholeConfig::default();
        let mut wormhole = Wormhole::new("a", "b", 5, "test", &config);

        // Set last_used to 29 days ago (just under threshold)
        wormhole.last_used_at = Utc::now() - ChronoDuration::days(29);
        assert!(!wormhole.should_decay(&config));
    }

    #[test]
    fn test_decay_application() {
        let config = WormholeConfig::default();
        let mut wormhole = Wormhole::new("a", "b", 10, "test", &config);

        let initial_strength = wormhole.strength;

        // Apply 50% decay
        wormhole.apply_decay(0.5);

        assert!((wormhole.strength - initial_strength * 0.5).abs() < 0.001);
    }

    #[test]
    fn test_manager_applies_decay() {
        let config = WormholeConfig::default();
        let mut manager = WormholeManager::new(config);

        // Create a wormhole
        for _ in 0..3 {
            manager.record_co_access("pat_1", "pat_2", None);
        }

        // Get the wormhole key and manually set last_used to 31 days ago
        let wormhole_key = "pat_1_pat_2".to_string();
        if let Some(wh) = manager.wormholes.get_mut(&wormhole_key) {
            wh.last_used_at = Utc::now() - ChronoDuration::days(31);
        }

        let initial_strength = manager.wormholes.get(&wormhole_key).unwrap().strength;

        // Apply decay
        let decayed_count = manager.apply_decay(0.5);

        assert_eq!(decayed_count, 1);
        assert!(manager.wormholes.get(&wormhole_key).unwrap().strength < initial_strength);
    }

    #[test]
    fn test_weak_wormholes_removed_after_decay() {
        let config = WormholeConfig::default();
        let mut manager = WormholeManager::new(config);

        // Create a wormhole
        for _ in 0..3 {
            manager.record_co_access("pat_1", "pat_2", None);
        }

        let wormhole_key = "pat_1_pat_2".to_string();

        // Make it old and weak
        if let Some(wh) = manager.wormholes.get_mut(&wormhole_key) {
            wh.last_used_at = Utc::now() - ChronoDuration::days(31);
            wh.strength = 0.15; // Weak but not below removal threshold
        }

        // Apply aggressive decay - should remove the wormhole
        manager.apply_decay(0.1); // Brings strength to 0.015

        assert!(manager.wormholes.get(&wormhole_key).is_none());
    }

    #[test]
    fn test_traversal_resets_decay_timer() {
        let config = WormholeConfig::default();
        let mut wormhole = Wormhole::new("a", "b", 5, "test", &config);

        // Set to 35 days ago
        wormhole.last_used_at = Utc::now() - ChronoDuration::days(35);
        assert!(wormhole.should_decay(&config));

        // Record a traversal
        wormhole.record_traversal(100);

        // Should no longer decay (just used)
        assert!(!wormhole.should_decay(&config));
    }
}

// ============================================================================
// Max Wormholes Per Node Tests
// ============================================================================

mod max_wormholes_tests {
    use super::*;

    #[test]
    fn test_max_wormholes_limit_enforced() {
        let config = WormholeConfig {
            max_wormholes_per_node: 2,
            co_access_threshold: 1,
            ..Default::default()
        };
        let mut manager = WormholeManager::new(config);

        // Create first wormhole
        assert!(manager.record_co_access("central", "node_1", None).is_some());

        // Create second wormhole
        assert!(manager.record_co_access("central", "node_2", None).is_some());

        // Third should fail - limit reached for "central"
        assert!(manager.record_co_access("central", "node_3", None).is_none());

        // But wormhole between other nodes should work
        assert!(manager.record_co_access("node_1", "node_2", None).is_some());
    }

    #[test]
    fn test_limit_applies_to_both_endpoints() {
        let config = WormholeConfig {
            max_wormholes_per_node: 2,
            co_access_threshold: 1,
            ..Default::default()
        };
        let mut manager = WormholeManager::new(config);

        // Fill up node_a
        manager.record_co_access("node_a", "x1", None);
        manager.record_co_access("node_a", "x2", None);

        // node_a is full, so this should fail even though node_b has room
        assert!(manager.record_co_access("node_a", "node_b", None).is_none());

        // But node_b can still form wormholes with other nodes
        assert!(manager.record_co_access("node_b", "y1", None).is_some());
    }

    #[test]
    fn test_count_wormholes_correctly() {
        let config = WormholeConfig {
            max_wormholes_per_node: 3,
            co_access_threshold: 1,
            ..Default::default()
        };
        let mut manager = WormholeManager::new(config);

        manager.record_co_access("hub", "spoke1", None);
        manager.record_co_access("hub", "spoke2", None);
        manager.record_co_access("hub", "spoke3", None);

        let hub_wormholes = manager.get_wormholes_for_node("hub");
        assert_eq!(hub_wormholes.len(), 3);
    }
}

// ============================================================================
// Traversal Savings Tests
// ============================================================================

mod traversal_savings_tests {
    use super::*;

    #[test]
    fn test_traversal_savings_calculation() {
        let mut graph = SimulatedGraph::new();

        // Create a linear chain: A -> B -> C -> D -> E
        graph.add_edge("A", "B", 1.0);
        graph.add_edge("B", "C", 1.0);
        graph.add_edge("C", "D", 1.0);
        graph.add_edge("D", "E", 1.0);

        // Wormhole from A to E
        let config = WormholeConfig::default();
        let wormhole = Wormhole::new("A", "E", 5, "test", &config);

        let savings = graph.calculate_traversal_savings(&wormhole);

        // Normal path: A -> B -> C -> D -> E = 4 hops
        // With wormhole: A -> E = 1 hop
        // Savings: (4 - 1) / 4 = 0.75 = 75%
        assert!((savings - 0.75).abs() < 0.01);
        assert!(savings >= 0.5, "Should provide >50% traversal reduction");
    }

    #[test]
    fn test_traversal_savings_greater_than_50_percent() {
        let mut graph = SimulatedGraph::new();

        // Create a longer chain
        for i in 0..10 {
            graph.add_edge(&format!("node_{}", i), &format!("node_{}", i + 1), 1.0);
        }

        let config = WormholeConfig::default();
        let wormhole = Wormhole::new("node_0", "node_10", 5, "test", &config);

        let savings = graph.calculate_traversal_savings(&wormhole);

        // 10 hops to 1 hop = 90% savings
        assert!(savings > 0.5, "Wormhole should provide >50% savings");
    }

    #[test]
    fn test_no_savings_for_adjacent_nodes() {
        let mut graph = SimulatedGraph::new();
        graph.add_edge("A", "B", 1.0);

        let config = WormholeConfig::default();
        let wormhole = Wormhole::new("A", "B", 5, "test", &config);

        let savings = graph.calculate_traversal_savings(&wormhole);

        // Already adjacent - no savings
        assert!((savings - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_record_traversal_statistics() {
        let config = WormholeConfig::default();
        let mut wormhole = Wormhole::new("A", "E", 5, "test", &config);

        assert_eq!(wormhole.traversal_count, 0);
        assert_eq!(wormhole.traversal_savings_ms, 0);

        wormhole.record_traversal(100);
        wormhole.record_traversal(150);
        wormhole.record_traversal(50);

        assert_eq!(wormhole.traversal_count, 3);
        assert_eq!(wormhole.traversal_savings_ms, 300);
        assert!((wormhole.avg_traversal_savings() - 100.0).abs() < 0.001);
    }
}

// ============================================================================
// Audit Logging Tests
// ============================================================================

mod audit_logging_tests {
    use super::*;

    #[test]
    fn test_wormhole_creation_logged() {
        let mut manager = WormholeManager::with_default_config();

        for _ in 0..3 {
            manager.record_co_access("pat_1", "pat_2", None);
        }

        let audit_log = manager.audit_log();
        assert!(!audit_log.is_empty());

        let creation_entry = audit_log.iter().find(|e| e.operation == "created");
        assert!(creation_entry.is_some());

        let entry = creation_entry.unwrap();
        assert!(entry.old_strength.is_none());
        assert!(entry.new_strength.is_some());
        assert!(entry.reason.contains("Co-accessed"));
    }

    #[test]
    fn test_strength_update_logged() {
        let mut manager = WormholeManager::with_default_config();

        // Create wormhole
        for _ in 0..3 {
            manager.record_co_access("pat_a", "pat_b", None);
        }

        // Trigger strength update
        for _ in 0..2 {
            manager.record_co_access("pat_a", "pat_b", None);
        }

        let audit_log = manager.audit_log();
        let update_entries: Vec<_> = audit_log
            .iter()
            .filter(|e| e.operation == "strength_updated")
            .collect();

        assert!(!update_entries.is_empty());
        for entry in update_entries {
            assert!(entry.old_strength.is_some());
            assert!(entry.new_strength.is_some());
        }
    }

    #[test]
    fn test_decay_logged() {
        let config = WormholeConfig::default();
        let mut manager = WormholeManager::new(config);

        // Create wormhole
        for _ in 0..3 {
            manager.record_co_access("pat_1", "pat_2", None);
        }

        // Make it old
        let wormhole_key = "pat_1_pat_2".to_string();
        if let Some(wh) = manager.wormholes.get_mut(&wormhole_key) {
            wh.last_used_at = Utc::now() - ChronoDuration::days(31);
        }

        // Apply decay
        manager.apply_decay(0.5);

        let audit_log = manager.audit_log();
        let decay_entry = audit_log.iter().find(|e| e.operation == "decay_applied");
        assert!(decay_entry.is_some());
    }

    #[test]
    fn test_removal_logged() {
        let config = WormholeConfig::default();
        let mut manager = WormholeManager::new(config);

        // Create wormhole
        for _ in 0..3 {
            manager.record_co_access("pat_1", "pat_2", None);
        }

        let wormhole_key = "pat_1_pat_2".to_string();
        if let Some(wh) = manager.wormholes.get_mut(&wormhole_key) {
            wh.last_used_at = Utc::now() - ChronoDuration::days(31);
            wh.strength = 0.05; // Will be removed after decay
        }

        manager.apply_decay(0.5);

        let audit_log = manager.audit_log();
        let removal_entry = audit_log.iter().find(|e| e.operation == "removed");
        assert!(removal_entry.is_some());
    }
}

// ============================================================================
// Integration Tests
// ============================================================================

mod integration_tests {
    use super::*;

    #[test]
    fn test_full_wormhole_lifecycle() {
        let config = WormholeConfig::default();
        let mut manager = WormholeManager::new(config);

        // 1. Accumulate co-accesses (using names that sort as expected)
        assert!(manager.record_co_access("pat_start", "pat_end", Some("session-1".to_string())).is_none());
        assert!(manager.record_co_access("pat_start", "pat_end", Some("session-2".to_string())).is_none());

        // 2. Threshold reached - wormhole created
        let wormhole = manager.record_co_access("pat_start", "pat_end", Some("session-3".to_string()));
        assert!(wormhole.is_some());

        // Key is ordered: pat_end < pat_start
        let wormhole_key = "pat_end_pat_start".to_string();

        // 3. Record some traversals
        manager.record_traversal(&wormhole_key, 100);
        manager.record_traversal(&wormhole_key, 200);

        // 4. More co-accesses strengthen the wormhole
        let initial_strength = manager.wormholes.get(&wormhole_key).unwrap().strength;
        for _ in 0..5 {
            manager.record_co_access("pat_start", "pat_end", None);
        }
        let new_strength = manager.wormholes.get(&wormhole_key).unwrap().strength;
        assert!(new_strength > initial_strength);

        // 5. Verify stats
        let stats = manager.stats();
        assert_eq!(stats.total_wormholes, 1);
        assert_eq!(stats.total_traversals, 2);
        assert_eq!(stats.total_savings_ms, 300);
    }

    #[test]
    fn test_graph_traversal_with_wormholes() {
        let config = WormholeConfig::default();
        let mut manager = WormholeManager::new(config);
        let mut graph = SimulatedGraph::new();

        // Build a complex graph
        // A -- B -- C -- D -- E
        //      |         |
        //      F -- G -- H
        graph.add_edge("A", "B", 1.0);
        graph.add_edge("B", "C", 1.0);
        graph.add_edge("C", "D", 1.0);
        graph.add_edge("D", "E", 1.0);
        graph.add_edge("B", "F", 1.0);
        graph.add_edge("F", "G", 1.0);
        graph.add_edge("G", "H", 1.0);
        graph.add_edge("H", "D", 1.0);

        // Normal path from A to E = 4 hops
        let normal_distance = graph.shortest_path_distance("A", "E").unwrap();
        assert_eq!(normal_distance, 4);

        // Create wormhole from A to E
        for _ in 0..3 {
            manager.record_co_access("A", "E", None);
        }

        // With wormhole, effective distance is 1
        let wormhole = manager.get_wormholes_for_node("A")[0];
        let savings = graph.calculate_traversal_savings(wormhole);

        // 4 hops to 1 hop = 75% savings
        assert!(savings >= 0.5, "Wormhole should provide >50% savings");
    }

    #[test]
    fn test_wormhole_network() {
        let config = WormholeConfig {
            max_wormholes_per_node: 10,
            co_access_threshold: 2,
            ..Default::default()
        };
        let mut manager = WormholeManager::new(config);

        // Create a hub-and-spoke pattern of wormholes
        let spokes = vec!["spoke_1", "spoke_2", "spoke_3", "spoke_4", "spoke_5"];

        for spoke in &spokes {
            for _ in 0..2 {
                manager.record_co_access("hub", spoke, None);
            }
        }

        // All spokes should be connected to hub via wormholes
        let hub_wormholes = manager.get_wormholes_for_node("hub");
        assert_eq!(hub_wormholes.len(), 5);

        let stats = manager.stats();
        assert_eq!(stats.total_wormholes, 5);
    }
}

// ============================================================================
// Property-Based Tests
// ============================================================================

#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        /// Property: Wormhole strength is always in (0, 1).
        #[test]
        fn prop_strength_bounded(frequency in 1u32..1000u32, decay in 1.0f64..100.0f64) {
            let config = WormholeConfig {
                decay_constant: decay,
                ..Default::default()
            };

            let strength = Wormhole::calculate_strength(frequency, &config);

            prop_assert!(strength > 0.0);
            prop_assert!(strength < 1.0);
        }

        /// Property: Higher frequency always yields higher strength (monotonic).
        #[test]
        fn prop_strength_monotonic(freq1 in 1u32..500u32, delta in 1u32..500u32) {
            let config = WormholeConfig::default();
            let freq2 = freq1 + delta;

            let s1 = Wormhole::calculate_strength(freq1, &config);
            let s2 = Wormhole::calculate_strength(freq2, &config);

            prop_assert!(s2 > s1, "Higher frequency should yield higher strength");
        }

        /// Property: Decay factor reduces strength proportionally.
        #[test]
        fn prop_decay_reduces_strength(
            frequency in 3u32..100u32,
            decay_factor in 0.1f64..0.9f64
        ) {
            let config = WormholeConfig::default();
            let mut wormhole = Wormhole::new("a", "b", frequency, "test", &config);

            let initial_strength = wormhole.strength;
            wormhole.apply_decay(decay_factor);

            prop_assert!(wormhole.strength < initial_strength);
            prop_assert!(
                (wormhole.strength - initial_strength * decay_factor).abs() < 0.001,
                "Decay should reduce strength by decay factor"
            );
        }

        /// Property: Co-access threshold must be reached before wormhole creation.
        #[test]
        fn prop_threshold_required(threshold in 2u32..10u32) {
            let config = WormholeConfig {
                co_access_threshold: threshold,
                ..Default::default()
            };
            let mut manager = WormholeManager::new(config);

            // Access threshold - 1 times
            for _ in 0..(threshold - 1) {
                prop_assert!(manager.record_co_access("x", "y", None).is_none());
            }

            // Final access should create wormhole
            prop_assert!(manager.record_co_access("x", "y", None).is_some());
        }
    }
}

// ============================================================================
// Performance Tests
// ============================================================================

mod performance_tests {
    use super::*;

    #[test]
    fn test_many_co_accesses_performance() {
        let config = WormholeConfig::default();
        let mut manager = WormholeManager::new(config);

        let start = Instant::now();

        // Record 10,000 co-accesses across 100 pattern pairs
        for i in 0..100 {
            for j in 0..100 {
                manager.record_co_access(
                    &format!("pat_{}", i),
                    &format!("pat_{}", j),
                    Some(format!("session_{}", i * 100 + j)),
                );
            }
        }

        let duration = start.elapsed();

        assert!(
            duration.as_millis() < 2000,
            "10,000 co-accesses should complete in < 2s, took {:?}",
            duration
        );
    }

    #[test]
    fn test_wormhole_lookup_performance() {
        let config = WormholeConfig {
            co_access_threshold: 1,
            max_wormholes_per_node: 100,
            ..Default::default()
        };
        let mut manager = WormholeManager::new(config);

        // Create many wormholes
        for i in 0..50 {
            manager.record_co_access("hub", &format!("spoke_{}", i), None);
        }

        let start = Instant::now();

        // Look up wormholes 1000 times
        for _ in 0..1000 {
            let _ = manager.get_wormholes_for_node("hub");
        }

        let duration = start.elapsed();

        assert!(
            duration.as_millis() < 100,
            "1000 lookups should complete in < 100ms, took {:?}",
            duration
        );
    }
}
