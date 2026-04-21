//! Meta-Learning Types
//!
//! Core types for the meta-learning system including EWC configuration,
//! pattern importance, domain transfer, and learning rate adaptation.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Elastic Weight Consolidation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EwcConfig {
    /// Lambda: regularization strength (higher = more protection)
    pub lambda: f64,
    /// Minimum importance threshold to protect a pattern
    pub importance_threshold: f64,
    /// Number of recent outcomes to consider for Fisher information
    pub fisher_sample_size: usize,
    /// Decay factor for old Fisher information (0.0-1.0)
    pub fisher_decay: f64,
}

impl Default for EwcConfig {
    fn default() -> Self {
        Self {
            lambda: 1000.0,
            importance_threshold: 0.3,
            fisher_sample_size: 100,
            fisher_decay: 0.95,
        }
    }
}

/// Importance weight for a pattern (Fisher information proxy)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternImportance {
    /// Pattern ID
    pub pattern_id: String,
    /// Importance score (0.0 - 1.0)
    pub importance: f64,
    /// Fisher information estimate
    pub fisher_info: f64,
    /// Number of successful applications
    pub success_count: u32,
    /// Total applications
    pub total_count: u32,
    /// Last updated timestamp
    pub updated_at: DateTime<Utc>,
}

impl PatternImportance {
    /// Create a new pattern importance record
    pub fn new(pattern_id: impl Into<String>) -> Self {
        Self {
            pattern_id: pattern_id.into(),
            importance: 0.0,
            fisher_info: 0.0,
            success_count: 0,
            total_count: 0,
            updated_at: Utc::now(),
        }
    }

    /// Success rate for this pattern
    pub fn success_rate(&self) -> f64 {
        if self.total_count == 0 {
            0.5 // Prior
        } else {
            self.success_count as f64 / self.total_count as f64
        }
    }

    /// Update with a new outcome
    pub fn record_outcome(&mut self, success: bool) {
        self.total_count += 1;
        if success {
            self.success_count += 1;
        }
        self.updated_at = Utc::now();
    }
}

/// Domain transfer mapping
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DomainTransfer {
    /// Source domain
    pub source_domain: String,
    /// Target domain
    pub target_domain: String,
    /// Transfer coefficient (0.0-1.0, how applicable patterns are)
    pub transfer_coefficient: f64,
    /// Number of successful transfers
    pub successful_transfers: u32,
    /// Number of failed transfers
    pub failed_transfers: u32,
    /// Specific pattern mappings that worked
    pub pattern_mappings: Vec<PatternMapping>,
    /// Last updated
    pub updated_at: DateTime<Utc>,
}

impl DomainTransfer {
    /// Create a new domain transfer mapping
    pub fn new(source: impl Into<String>, target: impl Into<String>) -> Self {
        Self {
            source_domain: source.into(),
            target_domain: target.into(),
            transfer_coefficient: 0.5, // Initial prior
            successful_transfers: 0,
            failed_transfers: 0,
            pattern_mappings: Vec::new(),
            updated_at: Utc::now(),
        }
    }

    /// Record a transfer outcome and update coefficient
    pub fn record_transfer(&mut self, success: bool, mapping: Option<PatternMapping>) {
        if success {
            self.successful_transfers += 1;
        } else {
            self.failed_transfers += 1;
        }

        // Update transfer coefficient using Bayesian update
        let total = self.successful_transfers + self.failed_transfers;
        if total > 0 {
            // Beta distribution posterior mean
            let alpha = self.successful_transfers as f64 + 1.0;
            let beta = self.failed_transfers as f64 + 1.0;
            self.transfer_coefficient = alpha / (alpha + beta);
        }

        if let Some(m) = mapping {
            self.pattern_mappings.push(m);
        }

        self.updated_at = Utc::now();
    }
}

/// Mapping between patterns in different domains
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternMapping {
    pub source_pattern_id: String,
    pub target_pattern_id: String,
    pub similarity: f64,
    pub transfer_success_rate: f64,
    pub transfer_count: u32,
}

impl PatternMapping {
    pub fn new(source: impl Into<String>, target: impl Into<String>, similarity: f64) -> Self {
        Self {
            source_pattern_id: source.into(),
            target_pattern_id: target.into(),
            similarity,
            transfer_success_rate: 0.5,
            transfer_count: 0,
        }
    }
}

/// Learning rate configuration per domain
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningRateConfig {
    /// Domain name
    pub domain: String,
    /// Base learning rate (reward adjustment multiplier)
    pub base_rate: f64,
    /// Current adapted rate
    pub current_rate: f64,
    /// Performance history (recent success rates)
    pub performance_history: Vec<f64>,
    /// Bayesian prior (alpha, beta for Beta distribution)
    pub prior: (f64, f64),
    /// Maximum history size
    pub max_history: usize,
}

impl LearningRateConfig {
    pub fn new(domain: impl Into<String>) -> Self {
        Self {
            domain: domain.into(),
            base_rate: 0.1,
            current_rate: 0.1,
            performance_history: Vec::new(),
            prior: (1.0, 1.0), // Uniform prior
            max_history: 50,
        }
    }

    /// Record a performance measurement and adapt the learning rate
    pub fn record_performance(&mut self, success_rate: f64) {
        self.performance_history.push(success_rate);

        // Keep history bounded
        if self.performance_history.len() > self.max_history {
            self.performance_history.remove(0);
        }

        // Adapt learning rate based on recent performance
        self.adapt_rate();
    }

    /// Adapt learning rate based on performance history
    fn adapt_rate(&mut self) {
        if self.performance_history.is_empty() {
            return;
        }

        let recent: Vec<_> = self.performance_history.iter().rev().take(10).collect();
        let avg_performance: f64 = recent.iter().copied().sum::<f64>() / recent.len() as f64;

        // If performing well, increase learning rate (learn faster)
        // If performing poorly, decrease learning rate (be more conservative)
        let adjustment = if avg_performance > 0.7 {
            1.2 // Increase by 20%
        } else if avg_performance < 0.3 {
            0.8 // Decrease by 20%
        } else {
            1.0 // No change
        };

        self.current_rate = (self.current_rate * adjustment).clamp(0.01, 0.5);
    }
}

/// Generalized pattern template
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternTemplate {
    /// Template ID
    pub id: String,
    /// Abstracted problem description (with placeholders)
    pub problem_template: String,
    /// Abstracted solution template
    pub solution_template: String,
    /// Source pattern IDs this was generalized from
    pub source_patterns: Vec<String>,
    /// Domain applicability
    pub domains: Vec<String>,
    /// Confidence in the generalization
    pub confidence: f64,
    /// Number of times instantiated
    pub instantiation_count: u32,
    /// Created timestamp
    pub created_at: DateTime<Utc>,
}

impl PatternTemplate {
    pub fn new(
        id: impl Into<String>,
        problem: impl Into<String>,
        solution: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            problem_template: problem.into(),
            solution_template: solution.into(),
            source_patterns: Vec::new(),
            domains: Vec::new(),
            confidence: 0.5,
            instantiation_count: 0,
            created_at: Utc::now(),
        }
    }
}

/// Meta-learning statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MetaLearningStats {
    /// Total patterns protected by EWC
    pub protected_patterns: u32,
    /// Catastrophic forgetting events prevented
    pub forgetting_prevented: u32,
    /// Successful cross-domain transfers
    pub successful_transfers: u32,
    /// Failed cross-domain transfers
    pub failed_transfers: u32,
    /// Learning rate adjustments made
    pub rate_adjustments: u32,
    /// Patterns generalized
    pub patterns_generalized: u32,
    /// Templates created
    pub templates_created: u32,
    /// Last optimization run
    pub last_optimization: Option<DateTime<Utc>>,
}

/// Configuration for the entire meta-learning system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetaLearningConfig {
    /// EWC configuration
    pub ewc: EwcConfig,
    /// Whether transfer learning is enabled
    pub transfer_enabled: bool,
    /// Minimum similarity for pattern generalization
    pub generalization_threshold: f64,
    /// Whether to auto-optimize during dream cycle
    pub auto_optimize: bool,
}

impl Default for MetaLearningConfig {
    fn default() -> Self {
        Self {
            ewc: EwcConfig::default(),
            transfer_enabled: true,
            generalization_threshold: 0.85,
            auto_optimize: true,
        }
    }
}
