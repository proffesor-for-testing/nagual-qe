//! Learning and self-improvement CLI commands.
//!
//! This module provides commands for:
//! - Recording outcomes for pattern applications
//! - Running improvement cycles on pattern domains
//! - Viewing domain insights and statistics
//! - Triggering pattern consolidation
//! - Viewing improvement recommendations
//! - Generating embeddings for patterns
//! - Managing strategy cache (EGUR-inspired)
//! - Fine-tuning LoRA adapters
//! - Finding and merging duplicate patterns
//! - Managing validation scenarios
//! - Running meta-learning analysis
//!
//! # Module Organization
//!
//! The learning commands are split into focused submodules:
//! - `record` - Outcome recording
//! - `improve` - Self-improvement cycles
//! - `insights` - Domain analytics
//! - `consolidate` - Pattern consolidation
//! - `recommendations` - Improvement recommendations
//! - `embed` - Embedding generation
//! - `strategy` - Strategy cache management
//! - `finetune` - LoRA fine-tuning
//! - `dedup` - Duplicate detection
//! - `scenario` - Validation scenarios
//! - `meta` - Meta-learning
//! - `common` - Shared utilities

mod auto_promote;
mod common;
mod consolidate;
pub mod daily;
mod dedup;
mod drift;
mod embed;
mod finetune;
mod improve;
mod insights;
mod meta;
mod recommendations;
mod record;
mod scenario;
mod strange_loop_cmd;
mod strategy;
mod transfer_cmd;

use clap::{Args, Subcommand};

use crate::error::Result;

// Re-export Args types for CLI parsing
pub use auto_promote::AutoPromoteArgs;
pub use consolidate::ConsolidateArgs;
pub use daily::DailyArgs;
pub use dedup::DedupArgs;
pub use drift::DriftArgs;
pub use embed::EmbedArgs;
pub use finetune::FinetuneArgs;
pub use improve::ImproveArgs;
pub use insights::InsightsArgs;
pub use meta::MetaArgs;
pub use recommendations::RecommendationsArgs;
pub use record::RecordArgs;
pub use scenario::ScenarioArgs;
pub use strange_loop_cmd::StrangeLoopArgs;
pub use strategy::StrategyArgs;
pub use transfer_cmd::TransferArgs;

/// Learning and self-improvement commands.
///
/// Provides tools for analyzing pattern performance, generating
/// improvement recommendations, and managing pattern consolidation.
#[derive(Args, Debug)]
pub struct LearnCommand {
    #[command(subcommand)]
    pub subcommand: LearnSubcommand,
}

/// Learning subcommands.
#[derive(Subcommand, Debug)]
pub enum LearnSubcommand {
    /// Record an outcome for a pattern application.
    ///
    /// Records whether a pattern was successfully applied and
    /// updates its effectiveness metrics accordingly.
    Record(RecordArgs),

    /// Run a self-improvement cycle on patterns.
    ///
    /// Analyzes patterns in the specified domain and generates
    /// recommendations for consolidation, archiving, or improvement.
    Improve(ImproveArgs),

    /// Show domain insights and statistics.
    ///
    /// Displays aggregated metrics, trends, and top patterns
    /// for a given domain with time-windowed analysis.
    Insights(InsightsArgs),

    /// Trigger pattern consolidation.
    ///
    /// Consolidates similar patterns, archives low-performers,
    /// and cleans up stale entries based on configured triggers.
    Consolidate(ConsolidateArgs),

    /// Show improvement recommendations.
    ///
    /// Lists all pending recommendations from the last improvement
    /// cycle, sorted by priority.
    Recommendations(RecommendationsArgs),

    /// Generate embeddings for all patterns.
    ///
    /// Uses the ONNX MiniLM model to generate 128-dimensional embeddings
    /// for patterns that don't have them yet. This enables similarity
    /// search, consolidation, and the full learning loop.
    Embed(EmbedArgs),

    /// Manage strategy cache (EGUR-inspired).
    ///
    /// Stores, searches, and lists successful strategies derived from
    /// pattern clusters. Strategies are cached for fast lookup by
    /// problem category.
    Strategy(StrategyArgs),

    /// Fine-tune a LoRA adapter for a specific domain.
    ///
    /// Trains a lightweight Low-Rank Adaptation (LoRA) adapter using
    /// contrastive learning on patterns from the specified domain.
    /// The adapter improves retrieval accuracy for domain-specific queries.
    Finetune(FinetuneArgs),

    /// Find and merge duplicate patterns.
    ///
    /// Uses BLAKE3 content hashes for exact duplicates and embedding
    /// cosine similarity for near-duplicates. Keeps the pattern with
    /// highest reward as canonical and aggregates reuse counts.
    Dedup(DedupArgs),

    /// Manage validation scenarios (holdout set).
    ///
    /// Scenarios are test cases for pattern validation. They prevent
    /// overfitting by evaluating patterns against scenarios they
    /// haven't "seen" during training.
    Scenario(ScenarioArgs),

    /// Run meta-learning analysis (ADR-035).
    ///
    /// Analyzes patterns across domains using EWC++ for catastrophic
    /// forgetting prevention and transfer learning for cross-domain
    /// knowledge adaptation.
    Meta(MetaArgs),

    /// Manage daily log files (memory/YYYY-MM-DD.md).
    ///
    /// Daily logs are human-readable Markdown files that serve as a
    /// staging area before entries are promoted to the pattern store.
    Daily(DailyArgs),

    /// Show embedding drift analysis per domain.
    ///
    /// Monitors how pattern embeddings change over time within each domain.
    /// High drift may indicate data quality issues or concept evolution.
    /// Stagnation may indicate the domain needs fresh contributions.
    Drift(DriftArgs),

    /// Auto-promote patterns meeting recurrence thresholds.
    ///
    /// Patterns seen 3+ times across 2+ distinct sessions within 30 days
    /// are promoted one tier (booster->crystal, crystal->reflex).
    Promote(AutoPromoteArgs),

    /// Show meta-cognitive evaluation status (strange-loop).
    ///
    /// Displays quality assessments from the Hofstadter-inspired recursive
    /// self-critique loop that evaluates learning pipeline health on every
    /// SONA outcome recording.
    #[command(name = "strange-loop")]
    StrangeLoop(StrangeLoopArgs),

    /// Cross-domain transfer learning (Meta Thompson Sampling).
    ///
    /// View expansion status, list registered domains, initiate prior
    /// transfer between domains, and check for learning plateaus.
    /// Powered by ruvector-domain-expansion.
    Transfer(TransferArgs),
}

impl LearnCommand {
    /// Execute the learn command.
    pub async fn run(&self) -> Result<()> {
        match &self.subcommand {
            LearnSubcommand::Record(args) => record::run(args).await,
            LearnSubcommand::Improve(args) => improve::run(args).await,
            LearnSubcommand::Insights(args) => insights::run(args).await,
            LearnSubcommand::Consolidate(args) => consolidate::run(args).await,
            LearnSubcommand::Recommendations(args) => recommendations::run(args).await,
            LearnSubcommand::Embed(args) => embed::run(args).await,
            LearnSubcommand::Strategy(args) => strategy::run(args).await,
            LearnSubcommand::Finetune(args) => finetune::run(args).await,
            LearnSubcommand::Dedup(args) => dedup::run(args).await,
            LearnSubcommand::Scenario(args) => scenario::run(args).await,
            LearnSubcommand::Meta(args) => meta::run(args).await,
            LearnSubcommand::Daily(args) => daily::run(args).await,
            LearnSubcommand::Drift(args) => drift::run(args).await,
            LearnSubcommand::Promote(args) => auto_promote::run(args).await,
            LearnSubcommand::StrangeLoop(args) => strange_loop_cmd::run(args).await,
            LearnSubcommand::Transfer(args) => transfer_cmd::run(args).await,
        }
    }
}
