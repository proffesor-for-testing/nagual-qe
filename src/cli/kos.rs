//! KOS (Knowledge Operating System) CLI commands.
//!
//! Provides unified access to all KOS subsystems:
//! - Lineage (P0): Pattern derivation tracking
//! - Witness (P1): Tamper-evident mutation log
//! - Delta (P2): Field-level event sourcing
//! - Coherence Scoring: Contradiction detection
//! - Domain Transfer (P4): Cross-domain knowledge transfer
//! - Epochs (P5): Knowledge versioning
//! - Tiering: Hot/warm/cold access tiering
//! - Agent Views: Per-agent pattern visibility
//! - EWC: Elastic weight consolidation
//! - Routing Ladder: Reflex/retrieval/heavy routing
//! - Hyperbolic Index: Hyperbolic-space similarity search

use clap::{Args, Subcommand};
use std::path::PathBuf;

/// KOS (Knowledge Operating System) unified CLI
#[derive(Args, Debug)]
pub struct KosCommand {
    #[command(subcommand)]
    pub subcommand: KosSubcommand,

    /// Path to the SQLite database
    #[arg(long, default_value = "nagual.db", global = true)]
    pub db_path: PathBuf,
}

#[derive(Subcommand, Debug)]
pub enum KosSubcommand {
    /// Show KOS subsystem status overview
    Status,

    // -- P0: Lineage ----------------------------------------------------------

    /// Query pattern lineage (derivation tree)
    Lineage {
        #[command(subcommand)]
        cmd: LineageCmd,
    },

    // -- P1: Witness Chains ---------------------------------------------------

    /// Witness chain operations (tamper-evident log)
    Witness {
        #[command(subcommand)]
        cmd: WitnessCmd,
    },

    // -- P2: Delta Event Sourcing ---------------------------------------------

    /// Delta event sourcing (field-level change tracking)
    Delta {
        #[command(subcommand)]
        cmd: DeltaCmd,
    },

    // -- Coherence Scoring ----------------------------------------------------

    /// Coherence scoring (contradiction detection)
    Scoring {
        #[command(subcommand)]
        cmd: ScoringCmd,
    },

    // -- P4: Domain Transfer --------------------------------------------------

    /// Cross-domain knowledge transfer
    Transfer {
        #[command(subcommand)]
        cmd: TransferCmd,
    },

    // -- P5: Epochs -----------------------------------------------------------

    /// Knowledge epoch management (versioning)
    Epoch {
        #[command(subcommand)]
        cmd: EpochCmd,
    },

    // -- Tiering --------------------------------------------------------------

    /// Hot/warm/cold pattern tiering
    Tiering {
        #[command(subcommand)]
        cmd: TieringCmd,
    },

    // -- Agent Views ----------------------------------------------------------

    /// Per-agent pattern visibility management
    Views {
        #[command(subcommand)]
        cmd: ViewsCmd,
    },

    // -- EWC ------------------------------------------------------------------

    /// Elastic Weight Consolidation
    Ewc {
        #[command(subcommand)]
        cmd: EwcCmd,
    },

    // -- Routing Ladder -------------------------------------------------------

    /// Compute routing ladder (reflex/retrieval/heavy)
    Ladder {
        #[command(subcommand)]
        cmd: LadderCmd,
    },

    // -- Hyperbolic Index -----------------------------------------------------

    /// Hyperbolic-space similarity index
    Hyperbolic {
        #[command(subcommand)]
        cmd: HyperbolicCmd,
    },
}

// =============================================================================
// Lineage subcommands
// =============================================================================

#[derive(Subcommand, Debug)]
pub enum LineageCmd {
    /// Show lineage record for a pattern
    Get {
        /// Pattern ID
        pattern_id: String,
    },
    /// List ancestors (parents up to root)
    Ancestors {
        /// Pattern ID
        pattern_id: String,
    },
    /// List descendants (children down to leaves)
    Descendants {
        /// Pattern ID
        pattern_id: String,
    },
    /// Show children of a pattern
    Children {
        /// Pattern ID
        pattern_id: String,
    },
    /// Show depth distribution across all patterns
    Distribution,
}

// =============================================================================
// Witness subcommands
// =============================================================================

#[derive(Subcommand, Debug)]
pub enum WitnessCmd {
    /// Verify the entire witness chain integrity
    Verify,
    /// Verify witness entries for a specific pattern
    VerifyPattern {
        /// Pattern ID
        pattern_id: String,
    },
    /// Show audit trail for a pattern
    Audit {
        /// Pattern ID
        pattern_id: String,
    },
    /// Show total witness entry count
    Count,
}

// =============================================================================
// Delta subcommands
// =============================================================================

#[derive(Subcommand, Debug)]
pub enum DeltaCmd {
    /// Show full change history for a pattern
    History {
        /// Pattern ID
        pattern_id: String,
    },
    /// Show change summary statistics for a pattern
    Summary {
        /// Pattern ID
        pattern_id: String,
    },
}

// =============================================================================
// Coherence Scoring subcommands
// =============================================================================

#[derive(Subcommand, Debug)]
pub enum ScoringCmd {
    /// Scan a domain for contradictions
    Scan {
        /// Domain to scan
        domain: String,
    },
    /// Show system-wide coherence health
    Health,
}

// =============================================================================
// Transfer subcommands
// =============================================================================

#[derive(Subcommand, Debug)]
pub enum TransferCmd {
    /// Find transfer candidates between domains
    Candidates {
        /// Source domain
        source: String,
        /// Target domain
        target: String,
        /// Maximum number of candidates
        #[arg(long, default_value = "10")]
        limit: usize,
    },
    /// Show transfer statistics
    Stats,
}

// =============================================================================
// Epoch subcommands
// =============================================================================

#[derive(Subcommand, Debug)]
pub enum EpochCmd {
    /// Create a new epoch snapshot
    Create {
        /// Epoch name
        name: String,
        /// Optional description
        #[arg(long)]
        description: Option<String>,
    },
    /// Branch from an existing epoch
    Branch {
        /// Parent epoch name
        parent: String,
        /// New branch name
        branch_name: String,
    },
    /// Diff two epochs
    Diff {
        /// First epoch name
        epoch_a: String,
        /// Second epoch name
        epoch_b: String,
    },
    /// List all epochs
    List,
    /// Rollback to an epoch
    Rollback {
        /// Epoch name to rollback to
        name: String,
    },
}

// =============================================================================
// Tiering subcommands
// =============================================================================

#[derive(Subcommand, Debug)]
pub enum TieringCmd {
    /// Show tiering statistics
    Stats,
    /// Show the tier for a specific pattern
    Get {
        /// Pattern ID
        pattern_id: String,
    },
    /// Reclassify all patterns based on access history
    Reclassify,
    /// List hot (frequently accessed) patterns
    Hot {
        /// Maximum patterns to show
        #[arg(long, default_value = "20")]
        limit: usize,
    },
    /// List cold (rarely accessed) patterns
    Cold {
        /// Maximum patterns to show
        #[arg(long, default_value = "20")]
        limit: usize,
    },
}

// =============================================================================
// Agent Views subcommands
// =============================================================================

#[derive(Subcommand, Debug)]
pub enum ViewsCmd {
    /// List all registered agents and their view modes
    List,
    /// Show view statistics
    Stats,
    /// Register a new agent with a view mode
    Register {
        /// Agent identifier
        agent_id: String,
        /// View mode: include, exclude, or all
        #[arg(long, default_value = "all")]
        mode: String,
    },
    /// Show a specific agent's view
    Get {
        /// Agent identifier
        agent_id: String,
    },
}

// =============================================================================
// EWC subcommands
// =============================================================================

#[derive(Subcommand, Debug)]
pub enum EwcCmd {
    /// Show EWC statistics
    Stats,
    /// Compute Fisher information for a domain
    Fisher {
        /// Domain name
        domain: String,
    },
    /// Check protection status for a pattern in a domain
    Protection {
        /// Pattern ID
        pattern_id: String,
        /// Domain name
        domain: String,
    },
    /// Run EWC consolidation for a domain
    Consolidate {
        /// Domain name
        domain: String,
    },
}

// =============================================================================
// Routing Ladder subcommands
// =============================================================================

#[derive(Subcommand, Debug)]
pub enum LadderCmd {
    /// Show routing ladder statistics
    Stats,
    /// Route a query and show which compute lane it maps to
    Route {
        /// Query text
        query: String,
        /// Estimated complexity (0.0 - 1.0)
        #[arg(long, default_value = "0.5")]
        complexity: f64,
    },
    /// Check if a query hits the reflex cache
    Reflex {
        /// Query text
        query: String,
    },
}

// =============================================================================
// Hyperbolic Index subcommands
// =============================================================================

#[derive(Subcommand, Debug)]
pub enum HyperbolicCmd {
    /// Show hyperbolic index statistics (in-memory demo)
    Stats,
}

// =============================================================================
// Execution
// =============================================================================

impl KosCommand {
    pub async fn run(self) -> crate::error::Result<()> {
        // Destructure to avoid partial-move of `self` in match arms.
        let KosCommand { subcommand, db_path } = self;
        // Reconstruct a "shell" KosCommand with Status placeholder so we can call
        // &self helper methods that only need db_path.
        let shell = KosCommand {
            subcommand: KosSubcommand::Status,
            db_path,
        };

        match subcommand {
            KosSubcommand::Status => shell.run_status().await,

            KosSubcommand::Lineage { cmd } => shell.run_lineage(cmd).await,

            KosSubcommand::Witness { cmd } => shell.run_witness(cmd).await,

            KosSubcommand::Delta { cmd } => shell.run_delta(cmd).await,

            KosSubcommand::Scoring { cmd } => shell.run_scoring(cmd).await,

            KosSubcommand::Transfer { cmd } => shell.run_transfer(cmd).await,

            KosSubcommand::Epoch { cmd } => shell.run_epoch(cmd).await,

            KosSubcommand::Tiering { cmd } => shell.run_tiering(cmd).await,

            KosSubcommand::Views { cmd } => shell.run_views(cmd).await,

            KosSubcommand::Ewc { cmd } => shell.run_ewc(cmd).await,

            KosSubcommand::Ladder { cmd } => shell.run_ladder(cmd).await,

            KosSubcommand::Hyperbolic { cmd } => shell.run_hyperbolic(cmd).await,
        }
    }

    // -------------------------------------------------------------------------
    // Helpers
    // -------------------------------------------------------------------------

    /// Open the SQLite database and return an Arc<SqliteDb>.
    async fn open_db(&self) -> crate::error::Result<std::sync::Arc<crate::db::SqliteDb>> {
        let storage = super::common::init_storage_sqlite_only(&self.db_path).await?;
        Ok(storage.adapter().sqlite().clone())
    }

    // -------------------------------------------------------------------------
    // Status
    // -------------------------------------------------------------------------

    async fn run_status(&self) -> crate::error::Result<()> {
        println!("=== Nagual KOS (Knowledge Operating System) ===\n");

        let features: Vec<(&str, bool)> = vec![
            ("lineage", true),
            ("witness", true),
            ("delta", true),
            ("coherence-scoring", true),
            ("domain-transfer", true),
            ("epochs", true),
            ("tiering", true),
            ("agent-views", true),
            ("ewc", true),
            ("routing-ladder", true),
            ("hyperbolic-index", true),
        ];

        println!("Feature Flags:");
        for (name, enabled) in &features {
            let icon = if *enabled { "[x]" } else { "[ ]" };
            println!("  {} {}", icon, name);
        }

        let enabled_count = features.iter().filter(|(_, e)| *e).count();
        println!("\n{}/{} subsystems enabled", enabled_count, features.len());

        // If we can open the DB, show live stats from enabled subsystems
        match self.open_db().await {
            Ok(db) => {
                println!("\nLive Statistics (from {}):", self.db_path.display());

                {
                    let w = crate::witness::WitnessChain::new(db.clone());
                    match w.count().await {
                        Ok(c) => println!("  Witness entries:    {}", c),
                        Err(e) => println!("  Witness entries:    (error: {})", e),
                    }
                }

                {
                    match crate::tiering::TieringManager::new(
                        db.clone(),
                        crate::tiering::TieringConfig::default(),
                    )
                    .await
                    {
                        Ok(tm) => match tm.stats().await {
                            Ok(s) => {
                                println!(
                                    "  Tiering:            hot={} warm={} cold={}",
                                    s.hot_count, s.warm_count, s.cold_count
                                );
                            }
                            Err(e) => println!("  Tiering:            (error: {})", e),
                        },
                        Err(e) => println!("  Tiering:            (init error: {})", e),
                    }
                }

                {
                    match crate::learning::ewc::EwcManager::new(
                        db.clone(),
                        crate::learning::ewc::EwcEngineConfig::default(),
                    )
                    .await
                    {
                        Ok(ewc) => match ewc.stats().await {
                            Ok(s) => {
                                println!("  EWC domains:        {}", s.domains_tracked);
                                println!("  EWC boundaries:     {}", s.total_boundaries_detected);
                                println!("  EWC consolidations: {}", s.total_consolidations);
                                println!("  EWC protected:      {}", s.patterns_protected);
                            }
                            Err(e) => println!("  EWC:                (error: {})", e),
                        },
                        Err(e) => println!("  EWC:                (init error: {})", e),
                    }
                }

                {
                    match crate::router::ladder::RoutingLadder::new(
                        db.clone(),
                        crate::router::ladder::LadderConfig::default(),
                    )
                    .await
                    {
                        Ok(ladder) => match ladder.stats().await {
                            Ok(s) => {
                                println!("  Ladder cache size:  {}", s.cache_size);
                                println!("  Ladder hit rate:    {:.1}%", s.reflex_hit_rate * 100.0);
                                println!("  Ladder requests:    {}", s.total_requests);
                            }
                            Err(e) => println!("  Ladder:             (error: {})", e),
                        },
                        Err(e) => println!("  Ladder:             (init error: {})", e),
                    }
                }

                {
                    match crate::agent_views::ViewManager::new(
                        db.clone(),
                        crate::agent_views::ViewConfig::default(),
                    )
                    .await
                    {
                        Ok(vm) => match vm.stats().await {
                            Ok(s) => println!("  Agent views:        {} agents", s.total_agents),
                            Err(e) => println!("  Agent views:        (error: {})", e),
                        },
                        Err(e) => println!("  Agent views:        (init error: {})", e),
                    }
                }

                let _ = db;
            }
            Err(e) => {
                println!("\n(Could not open database: {})", e);
            }
        }

        Ok(())
    }

    // -------------------------------------------------------------------------
    // Lineage (P0)
    // -------------------------------------------------------------------------

    async fn run_lineage(&self, cmd: LineageCmd) -> crate::error::Result<()> {
        use crate::lineage::LineageQuery;
        use crate::reasoning_bank::pattern::PatternId;

        let db = self.open_db().await?;
        let lq = LineageQuery::new(db);

        match cmd {
            LineageCmd::Get { pattern_id } => {
                let pid = PatternId::from_string(pattern_id);
                match lq.get(&pid).await? {
                    Some(rec) => {
                        println!("Pattern:     {}", rec.pattern_id.as_str());
                        println!(
                            "Parent:      {}",
                            rec.parent_id
                                .as_ref()
                                .map(|p| p.as_str())
                                .unwrap_or("(none)")
                        );
                        println!("Derivation:  {}", rec.derivation_type);
                        println!("Depth:       {}", rec.lineage_depth);
                        println!("Created:     {}", rec.created_at);
                    }
                    None => println!("No lineage record found for pattern"),
                }
            }
            LineageCmd::Ancestors { pattern_id } => {
                let pid = PatternId::from_string(pattern_id);
                let ancestors = lq.ancestors(&pid).await?;
                if ancestors.is_empty() {
                    println!("No ancestors found (pattern is an original or not tracked)");
                } else {
                    println!("Ancestors ({} total):", ancestors.len());
                    for (i, a) in ancestors.iter().enumerate() {
                        println!(
                            "  {}. {} (depth={}, via={})",
                            i + 1,
                            a.pattern_id.as_str(),
                            a.lineage_depth,
                            a.derivation_type
                        );
                    }
                }
            }
            LineageCmd::Descendants { pattern_id } => {
                let pid = PatternId::from_string(pattern_id);
                let desc = lq.descendants(&pid).await?;
                if desc.is_empty() {
                    println!("No descendants found");
                } else {
                    println!("Descendants ({} total):", desc.len());
                    for (i, d) in desc.iter().enumerate() {
                        println!(
                            "  {}. {} (depth={}, via={})",
                            i + 1,
                            d.pattern_id.as_str(),
                            d.lineage_depth,
                            d.derivation_type
                        );
                    }
                }
            }
            LineageCmd::Children { pattern_id } => {
                let pid = PatternId::from_string(pattern_id);
                let children = lq.children(&pid).await?;
                if children.is_empty() {
                    println!("No children found");
                } else {
                    println!("Children ({} total):", children.len());
                    for c in &children {
                        println!(
                            "  - {} (via={}, depth={})",
                            c.pattern_id.as_str(),
                            c.derivation_type,
                            c.lineage_depth
                        );
                    }
                }
            }
            LineageCmd::Distribution => {
                let dist = lq.depth_distribution().await?;
                if dist.is_empty() {
                    println!("No lineage data available");
                } else {
                    println!("Lineage Depth Distribution:");
                    println!("  {:>5}  {:>8}", "Depth", "Count");
                    println!("  {:->5}  {:->8}", "", "");
                    for (depth, count) in &dist {
                        println!("  {:>5}  {:>8}", depth, count);
                    }
                }
            }
        }
        Ok(())
    }

    // -------------------------------------------------------------------------
    // Witness (P1)
    // -------------------------------------------------------------------------

    async fn run_witness(&self, cmd: WitnessCmd) -> crate::error::Result<()> {
        use crate::witness::WitnessChain;

        let db = self.open_db().await?;
        let wc = WitnessChain::new(db);
        wc.init_schema().await?;

        match cmd {
            WitnessCmd::Verify => {
                let result = wc.verify().await?;
                if result.valid {
                    println!(
                        "Witness chain VALID ({} entries checked, chain length {})",
                        result.entries_checked, result.chain_length
                    );
                } else {
                    println!(
                        "Witness chain BROKEN at seq {} ({} entries checked)",
                        result
                            .first_broken_seq
                            .map(|s| s.to_string())
                            .unwrap_or_else(|| "unknown".to_string()),
                        result.entries_checked,
                    );
                }
            }
            WitnessCmd::VerifyPattern { pattern_id } => {
                let result = wc.verify_pattern(&pattern_id).await?;
                if result.valid {
                    println!(
                        "Witness chain for pattern '{}' is VALID ({} entries)",
                        pattern_id, result.entries_checked
                    );
                } else {
                    println!(
                        "Witness chain for pattern '{}' is BROKEN at seq {}",
                        pattern_id,
                        result
                            .first_broken_seq
                            .map(|s| s.to_string())
                            .unwrap_or_else(|| "unknown".to_string()),
                    );
                }
            }
            WitnessCmd::Audit { pattern_id } => {
                let trail = wc.audit_trail(&pattern_id).await?;
                if trail.is_empty() {
                    println!("No witness entries for pattern '{}'", pattern_id);
                } else {
                    println!(
                        "Audit trail for '{}' ({} entries):",
                        pattern_id,
                        trail.len()
                    );
                    for entry in &trail {
                        println!(
                            "  seq={:<6} op={:<8} type={:<14} ts={} hash={}",
                            entry.seq,
                            entry.operation,
                            entry.witness_type,
                            entry.timestamp.format("%Y-%m-%d %H:%M:%S"),
                            hex::encode(&entry.entry_hash[..8]),
                        );
                    }
                }
            }
            WitnessCmd::Count => {
                let count = wc.count().await?;
                println!("Total witness entries: {}", count);
            }
        }
        Ok(())
    }

    // -------------------------------------------------------------------------
    // Delta (P2)
    // -------------------------------------------------------------------------

    async fn run_delta(&self, cmd: DeltaCmd) -> crate::error::Result<()> {
        use crate::delta::DeltaStore;

        let db = self.open_db().await?;
        let ds = DeltaStore::with_defaults(db);
        ds.init().await?;

        match cmd {
            DeltaCmd::History { pattern_id } => {
                let deltas = ds.history(&pattern_id).await?;
                if deltas.is_empty() {
                    println!("No deltas recorded for pattern '{}'", pattern_id);
                } else {
                    println!(
                        "Delta history for '{}' ({} records):",
                        pattern_id,
                        deltas.len()
                    );
                    for d in &deltas {
                        println!(
                            "  seq={:<4} op={:<8} ts={} fields={}{}",
                            d.seq,
                            d.operation,
                            d.timestamp.format("%Y-%m-%d %H:%M:%S"),
                            d.field_diffs.len(),
                            if d.snapshot.is_some() {
                                " [snapshot]"
                            } else {
                                ""
                            },
                        );
                        for fd in &d.field_diffs {
                            println!("    {} : {} -> {}", fd.field, fd.old_value, fd.new_value);
                        }
                    }
                }
            }
            DeltaCmd::Summary { pattern_id } => {
                let summary = ds.summary(&pattern_id).await?;
                println!("Delta Summary for '{}':", pattern_id);
                println!("  Total deltas:    {}", summary.total_deltas);
                println!("  Creates:         {}", summary.creates);
                println!("  Updates:         {}", summary.updates);
                println!("  Deletes:         {}", summary.deletes);
                println!("  Snapshots:       {}", summary.snapshot_count);
                println!(
                    "  First change:    {}",
                    summary
                        .first_change
                        .map(|t| t.to_string())
                        .unwrap_or_else(|| "(none)".to_string())
                );
                println!(
                    "  Last change:     {}",
                    summary
                        .last_change
                        .map(|t| t.to_string())
                        .unwrap_or_else(|| "(none)".to_string())
                );
            }
        }
        Ok(())
    }

    // -------------------------------------------------------------------------
    // Coherence Scoring
    // -------------------------------------------------------------------------

    async fn run_scoring(&self, cmd: ScoringCmd) -> crate::error::Result<()> {
        use crate::coherence::scoring::{CoherenceScorer, ScoringConfig};

        let db = self.open_db().await?;
        let scorer = CoherenceScorer::new(db, ScoringConfig::default());
        scorer.init_schema().await?;

        match cmd {
            ScoringCmd::Scan { domain } => {
                let scores = scorer.scan_domain(&domain).await?;
                if scores.is_empty() {
                    println!("No coherence issues found in domain '{}'", domain);
                } else {
                    println!(
                        "Coherence scan for '{}' ({} pairs):",
                        domain,
                        scores.len()
                    );
                    for s in &scores {
                        println!(
                            "  {} <-> {}  sim={:.3} contra={:.3} type={}",
                            s.pattern_a.as_str(),
                            s.pattern_b.as_str(),
                            s.similarity,
                            s.contradiction,
                            s.coherence_type,
                        );
                    }
                }
            }
            ScoringCmd::Health => {
                let health = scorer.system_health().await?;
                println!("Coherence Health:");
                println!("  Total pairs checked:      {}", health.total_pairs_checked);
                println!("  Contradictions found:     {}", health.contradictions_found);
                println!(
                    "  Contradiction rate:       {:.1}%",
                    health.contradiction_rate * 100.0
                );
                println!(
                    "  Entailment consistency:   {:.3}",
                    health.entailment_consistency
                );
                println!("  Domains scanned:          {}", health.domains_scanned.len());
                if let Some((domain, rate)) = &health.worst_domain {
                    println!("  Worst domain:             {} ({:.1}%)", domain, rate * 100.0);
                }
            }
        }
        Ok(())
    }

    // -------------------------------------------------------------------------
    // Domain Transfer (P4)
    // -------------------------------------------------------------------------

    async fn run_transfer(&self, cmd: TransferCmd) -> crate::error::Result<()> {
        use crate::learning::transfer::{DomainTransferEngine, TransferConfig};

        let db = self.open_db().await?;
        let engine = DomainTransferEngine::new(db, TransferConfig::default());
        engine.init_schema().await?;

        match cmd {
            TransferCmd::Candidates {
                source,
                target,
                limit,
            } => {
                let candidates = engine.find_candidates(&source, &target, limit).await?;
                if candidates.is_empty() {
                    println!(
                        "No transfer candidates found from '{}' to '{}'",
                        source, target
                    );
                } else {
                    println!(
                        "Transfer candidates {} -> {} ({} found):",
                        source,
                        target,
                        candidates.len()
                    );
                    for (i, c) in candidates.iter().enumerate() {
                        println!(
                            "  {}. {} (reward={:.3}, relevance={:.3}, expected={:.3})",
                            i + 1,
                            c.source_pattern_id,
                            c.source_reward,
                            c.relevance_score,
                            c.expected_reward,
                        );
                    }
                }
            }
            TransferCmd::Stats => {
                let stats = engine.stats();
                println!("Domain Transfer Statistics:");
                println!("  Total transfers:    {}", stats.total_transfers);
                println!("  Successful:         {}", stats.successful_transfers);
                println!(
                    "  Success rate:       {:.1}%",
                    if stats.total_transfers > 0 {
                        stats.successful_transfers as f64 / stats.total_transfers as f64 * 100.0
                    } else {
                        0.0
                    }
                );
                println!("  Domain pairs:       {}", stats.domain_pairs);
                if let Some((src, tgt, rate)) = &stats.best_pair {
                    println!(
                        "  Best pair:          {} -> {} ({:.1}% success)",
                        src,
                        tgt,
                        rate * 100.0
                    );
                }
            }
        }
        Ok(())
    }

    // -------------------------------------------------------------------------
    // Epochs (P5)
    // -------------------------------------------------------------------------

    async fn run_epoch(&self, cmd: EpochCmd) -> crate::error::Result<()> {
        use crate::epoch::EpochManager;

        let db = self.open_db().await?;
        let em = EpochManager::new(db);
        em.init_schema().await?;

        match cmd {
            EpochCmd::Create { name, description } => {
                let epoch = em
                    .create_epoch(&name, description.as_deref())
                    .await?;
                println!("Created epoch '{}':", epoch.name);
                println!("  ID:           {}", epoch.id);
                println!("  Patterns:     {}", epoch.pattern_count);
                println!("  Domains:      {}", epoch.domain_count);
                println!("  Created:      {}", epoch.created_at);
            }
            EpochCmd::Branch {
                parent,
                branch_name,
            } => {
                let epoch = em.branch(&parent, &branch_name).await?;
                println!(
                    "Branched '{}' from '{}' -> '{}'",
                    branch_name, parent, epoch.name
                );
                println!("  Patterns:     {}", epoch.pattern_count);
            }
            EpochCmd::Diff { epoch_a, epoch_b } => {
                let diff = em.diff(&epoch_a, &epoch_b).await?;
                println!("Diff: '{}' vs '{}':", epoch_a, epoch_b);
                println!("  Added:      {}", diff.added.len());
                println!("  Removed:    {}", diff.removed.len());
                println!("  Common:     {}", diff.common);
                if !diff.added.is_empty() {
                    println!("  Added patterns:");
                    for id in diff.added.iter().take(10) {
                        println!("    + {}", id);
                    }
                    if diff.added.len() > 10 {
                        println!("    ... and {} more", diff.added.len() - 10);
                    }
                }
            }
            EpochCmd::List => {
                let epochs = em.list().await?;
                if epochs.is_empty() {
                    println!("No epochs found");
                } else {
                    println!("Epochs ({} total):", epochs.len());
                    println!(
                        "  {:<20} {:>8} {:>8} {:>10}  {}",
                        "Name", "Patterns", "Domains", "Parent", "Created"
                    );
                    println!("  {:-<20} {:->8} {:->8} {:->10}  {:-<19}", "", "", "", "", "");
                    for e in &epochs {
                        println!(
                            "  {:<20} {:>8} {:>8} {:>10}  {}",
                            e.name,
                            e.pattern_count,
                            e.domain_count,
                            e.parent_epoch
                                .as_deref()
                                .unwrap_or("-"),
                            e.created_at.format("%Y-%m-%d %H:%M:%S"),
                        );
                    }
                }
            }
            EpochCmd::Rollback { name } => {
                let result = em.rollback(&name).await?;
                println!("Rollback to epoch '{}':", name);
                println!("  Patterns restored:  {}", result.patterns_restored);
                println!("  Patterns removed:   {}", result.patterns_removed);
            }
        }
        Ok(())
    }

    // -------------------------------------------------------------------------
    // Tiering
    // -------------------------------------------------------------------------

    async fn run_tiering(&self, cmd: TieringCmd) -> crate::error::Result<()> {
        use crate::tiering::{TieringConfig, TieringManager};

        let db = self.open_db().await?;
        let tm = TieringManager::new(db, TieringConfig::default()).await?;

        match cmd {
            TieringCmd::Stats => {
                let stats = tm.stats().await?;
                println!("Tiering Statistics:");
                println!("  Hot:                {}", stats.hot_count);
                println!("  Warm:               {}", stats.warm_count);
                println!("  Cold:               {}", stats.cold_count);
                println!(
                    "  Total tracked:      {}",
                    stats.hot_count + stats.warm_count + stats.cold_count
                );
                println!("  Total accesses:     {}", stats.total_accesses);
                println!("  Avg frequency:      {:.2}", stats.avg_access_frequency);
            }
            TieringCmd::Get { pattern_id } => {
                let tier = tm.get_tier(&pattern_id).await?;
                println!("Pattern '{}' tier: {}", pattern_id, tier.as_str());
            }
            TieringCmd::Reclassify => {
                let result = tm.reclassify_all().await?;
                println!("Reclassification complete:");
                println!(
                    "  Patterns checked:   {}",
                    result.promoted.len() + result.demoted.len() + result.unchanged as usize
                );
                println!("  Promotions:         {}", result.promoted.len());
                println!("  Demotions:          {}", result.demoted.len());
                println!("  Unchanged:          {}", result.unchanged);
            }
            TieringCmd::Hot { limit } => {
                let patterns = tm.get_hot_patterns(limit).await?;
                if patterns.is_empty() {
                    println!("No hot patterns found");
                } else {
                    println!("Hot patterns ({}):", patterns.len());
                    for p in &patterns {
                        println!(
                            "  {} (accesses={}, last={})",
                            p.pattern_id,
                            p.access_count,
                            p.last_accessed.format("%Y-%m-%d %H:%M:%S"),
                        );
                    }
                }
            }
            TieringCmd::Cold { limit } => {
                let patterns = tm.get_cold_patterns(limit).await?;
                if patterns.is_empty() {
                    println!("No cold patterns found");
                } else {
                    println!("Cold patterns ({}):", patterns.len());
                    for p in &patterns {
                        println!(
                            "  {} (accesses={}, last={})",
                            p.pattern_id,
                            p.access_count,
                            p.last_accessed.format("%Y-%m-%d %H:%M:%S"),
                        );
                    }
                }
            }
        }
        Ok(())
    }

    // -------------------------------------------------------------------------
    // Agent Views
    // -------------------------------------------------------------------------

    async fn run_views(&self, cmd: ViewsCmd) -> crate::error::Result<()> {
        use crate::agent_views::{ViewConfig, ViewManager, ViewMode};

        let db = self.open_db().await?;
        let vm = ViewManager::new(db, ViewConfig::default()).await?;

        match cmd {
            ViewsCmd::List => {
                let agents = vm.list_agents().await?;
                if agents.is_empty() {
                    println!("No agents registered");
                } else {
                    println!("Registered agents ({}):", agents.len());
                    for a in &agents {
                        println!(
                            "  {} (mode={}, grants={}, excludes={})",
                            a.agent_id,
                            a.view_mode.as_str(),
                            a.pattern_grants.len(),
                            a.pattern_excludes.len(),
                        );
                    }
                }
            }
            ViewsCmd::Stats => {
                let stats = vm.stats().await?;
                println!("Agent View Statistics:");
                println!("  Total agents:       {}", stats.total_agents);
                println!("  Total grants:       {}", stats.total_grants);
                println!("  Total excludes:     {}", stats.total_excludes);
                println!("  Modes:");
                for (mode, count) in &stats.agents_by_mode {
                    println!("    {}: {}", mode, count);
                }
            }
            ViewsCmd::Register { agent_id, mode } => {
                let view_mode = match mode.to_lowercase().as_str() {
                    "include" => ViewMode::Include,
                    "exclude" => ViewMode::Exclude,
                    "all" => ViewMode::All,
                    _ => {
                        return Err(crate::error::NagualError::internal(format!(
                            "Invalid view mode '{}'. Use: include, exclude, all",
                            mode
                        )));
                    }
                };
                vm.register_agent(&agent_id, view_mode).await?;
                println!(
                    "Registered agent '{}' with mode '{}'",
                    agent_id, mode
                );
            }
            ViewsCmd::Get { agent_id } => {
                match vm.get_view(&agent_id).await? {
                    Some(view) => {
                        println!("Agent '{}' view:", agent_id);
                        println!("  Mode:         {}", view.view_mode.as_str());
                        println!("  Grants:       {}", view.pattern_grants.len());
                        println!("  Excludes:     {}", view.pattern_excludes.len());
                        if !view.domain_filters.is_empty() {
                            println!("  Domains:      {:?}", view.domain_filters);
                        }
                    }
                    None => {
                        println!("Agent '{}' not found", agent_id);
                    }
                }
            }
        }
        Ok(())
    }

    // -------------------------------------------------------------------------
    // EWC
    // -------------------------------------------------------------------------

    async fn run_ewc(&self, cmd: EwcCmd) -> crate::error::Result<()> {
        use crate::learning::ewc::{EwcEngineConfig, EwcManager};

        let db = self.open_db().await?;
        let ewc = EwcManager::new(db, EwcEngineConfig::default()).await?;

        match cmd {
            EwcCmd::Stats => {
                let stats = ewc.stats().await?;
                println!("EWC Statistics:");
                println!("  Domains tracked:    {}", stats.domains_tracked);
                println!("  Boundaries found:   {}", stats.total_boundaries_detected);
                println!("  Consolidations:     {}", stats.total_consolidations);
                println!("  Avg importance:     {:.6}", stats.avg_fisher_importance);
                println!("  Patterns protected: {}", stats.patterns_protected);
            }
            EwcCmd::Fisher { domain } => {
                let fisher = ewc.compute_fisher(&domain).await?;
                println!("Fisher Information for domain '{}':", domain);
                println!("  Domain:           {}", fisher.domain);
                println!("  Patterns scored:  {}", fisher.importance_scores.len());
                println!("  Computed at:      {}", fisher.computed_at);
                println!("  Samples:          {}", fisher.sample_count);
                if !fisher.importance_scores.is_empty() {
                    let max_importance = fisher
                        .importance_scores
                        .iter()
                        .map(|(_, v)| *v)
                        .fold(0.0_f64, f64::max);
                    let avg_importance: f64 = fisher.importance_scores.values().sum::<f64>()
                        / fisher.importance_scores.len() as f64;
                    println!("  Max importance:   {:.6}", max_importance);
                    println!("  Avg importance:   {:.6}", avg_importance);
                }
            }
            EwcCmd::Protection { pattern_id, domain } => {
                let decision = ewc.check_protection(&pattern_id, &domain).await?;
                println!(
                    "Protection for pattern '{}' in domain '{}':",
                    pattern_id, domain
                );
                println!("  Protected:    {}", decision.should_protect);
                println!("  Importance:   {:.6}", decision.importance);
                println!("  Lambda:       {:.6}", decision.lambda);
                println!("  Penalty:      {:.6}", decision.penalty);
            }
            EwcCmd::Consolidate { domain } => {
                let record = ewc.consolidate(&domain).await?;
                println!("EWC Consolidation for domain '{}':", domain);
                println!("  Domain:            {}", record.domain);
                println!("  Lambda adjusted:   {:.4}", record.lambda_adjusted);
                println!("  Patterns:          {}", record.patterns_consolidated);
                println!("  Fisher computed:   {}", record.fisher_computed);
                println!("  Consolidated at:   {}", record.consolidated_at);
            }
        }
        Ok(())
    }

    // -------------------------------------------------------------------------
    // Routing Ladder
    // -------------------------------------------------------------------------

    async fn run_ladder(&self, cmd: LadderCmd) -> crate::error::Result<()> {
        use crate::router::ladder::{LadderConfig, RoutingLadder};

        let db = self.open_db().await?;
        let ladder = RoutingLadder::new(db, LadderConfig::default()).await?;

        match cmd {
            LadderCmd::Stats => {
                let stats = ladder.stats().await?;
                println!("Routing Ladder Statistics:");
                println!("  Cache size:         {}", stats.cache_size);
                println!("  Reflex hit rate:    {:.1}%", stats.reflex_hit_rate * 100.0);
                println!("  Total requests:     {}", stats.total_requests);
                println!("  Reflex hits:        {}", stats.reflex_hits);
                println!("  Reflex misses:      {}", stats.reflex_misses);
                println!("  Retrieval count:    {}", stats.retrieval_count);
                println!("  Heavy count:        {}", stats.heavy_count);
                println!("  Human count:        {}", stats.human_count);
                if !stats.avg_latency_by_lane.is_empty() {
                    println!("  Avg latency by lane:");
                    for (lane, avg_ms) in &stats.avg_latency_by_lane {
                        println!("    {}: {:.1}ms", lane, avg_ms);
                    }
                }
            }
            LadderCmd::Route { query, complexity } => {
                let decision = ladder.route(&query, complexity)?;
                println!("Routing Decision:");
                println!("  Query:        {}", query);
                println!("  Complexity:   {:.2}", complexity);
                println!("  Lane:         {:?}", decision.lane);
                println!("  Confidence:   {:.3}", decision.confidence);
                println!("  Reasoning:    {}", decision.reasoning);
            }
            LadderCmd::Reflex { query } => {
                match ladder.check_reflex(&query) {
                    Some(entry) => {
                        println!("Reflex HIT for query:");
                        println!("  Response:     {}", entry.response);
                        println!("  Confidence:   {:.3}", entry.confidence);
                        println!("  Hit count:    {}", entry.hit_count);
                    }
                    None => {
                        println!("No reflex entry for query '{}'", query);
                    }
                }
            }
        }
        Ok(())
    }

    // -------------------------------------------------------------------------
    // Hyperbolic Index
    // -------------------------------------------------------------------------

    async fn run_hyperbolic(&self, _cmd: HyperbolicCmd) -> crate::error::Result<()> {
        use crate::ml::hyperbolic::{HyperbolicConfig, HyperbolicIndex, HyperbolicIndexConfig};

        // HyperbolicIndex is purely in-memory, no DB dependency.
        // Show the configuration defaults and explain usage.
        let hyper_config = HyperbolicConfig::default();
        let index_config = HyperbolicIndexConfig::default();
        let index = HyperbolicIndex::new(hyper_config.clone(), index_config.clone());

        println!("Hyperbolic Index (in-memory):");
        println!("  Curvature:      {:.2}", hyper_config.curvature);
        println!("  Dimensions:     {}", hyper_config.dimension);
        println!("  Max layers:     {}", index_config.max_layers);
        println!("  ef_construction: {}", index_config.ef_construction);
        println!("  M connections:  {}", index_config.max_connections);
        println!("  Current nodes:  {}", index.len());
        println!("\nNote: The hyperbolic index is populated at runtime via the");
        println!("embedding pipeline. Use 'nagual learn embed' to generate");
        println!("embeddings, then the index is built on demand.");

        let stats = index.stats();
        println!("\nIndex Stats:");
        println!("  Total nodes:    {}", stats.total_nodes);
        println!("  Max layer:      {}", stats.max_layer);
        if stats.total_nodes > 0 {
            println!("  Avg connections: {:.2}", stats.avg_connections);
            if !stats.depth_distribution.is_empty() {
                println!("  Depth distribution:");
                for (bucket, count) in &stats.depth_distribution {
                    println!("    {:.2}: {}", bucket, count);
                }
            }
        }

        Ok(())
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    /// Wrapper for parsing KOS commands in tests.
    #[derive(Parser, Debug)]
    #[command(name = "nagual")]
    struct TestCli {
        #[command(subcommand)]
        command: TestCommands,
    }

    #[derive(clap::Subcommand, Debug)]
    enum TestCommands {
        Kos(KosCommand),
    }

    #[test]
    fn test_kos_status() {
        let args = vec!["nagual", "kos", "status"];
        let cli = TestCli::try_parse_from(args);
        assert!(cli.is_ok(), "Failed to parse kos status: {:?}", cli.err());
    }

    #[test]
    fn test_kos_db_path() {
        let args = vec!["nagual", "kos", "--db-path", "/tmp/test.db", "status"];
        let cli = TestCli::try_parse_from(args);
        assert!(cli.is_ok(), "Failed to parse kos with db-path: {:?}", cli.err());
    }

    #[test]
    fn test_kos_lineage_get() {
        let args = vec!["nagual", "kos", "lineage", "get", "abc123"];
        let cli = TestCli::try_parse_from(args);
        assert!(cli.is_ok(), "Failed to parse lineage get: {:?}", cli.err());
    }

    #[test]
    fn test_kos_lineage_distribution() {
        let args = vec!["nagual", "kos", "lineage", "distribution"];
        let cli = TestCli::try_parse_from(args);
        assert!(cli.is_ok());
    }

    #[test]
    fn test_kos_witness_verify() {
        let args = vec!["nagual", "kos", "witness", "verify"];
        let cli = TestCli::try_parse_from(args);
        assert!(cli.is_ok());
    }

    #[test]
    fn test_kos_witness_audit() {
        let args = vec!["nagual", "kos", "witness", "audit", "pat-123"];
        let cli = TestCli::try_parse_from(args);
        assert!(cli.is_ok());
    }

    #[test]
    fn test_kos_delta_history() {
        let args = vec!["nagual", "kos", "delta", "history", "pat-123"];
        let cli = TestCli::try_parse_from(args);
        assert!(cli.is_ok());
    }

    #[test]
    fn test_kos_delta_summary() {
        let args = vec!["nagual", "kos", "delta", "summary", "pat-123"];
        let cli = TestCli::try_parse_from(args);
        assert!(cli.is_ok());
    }

    #[test]
    fn test_kos_scoring_scan() {
        let args = vec!["nagual", "kos", "scoring", "scan", "rust"];
        let cli = TestCli::try_parse_from(args);
        assert!(cli.is_ok());
    }

    #[test]
    fn test_kos_scoring_health() {
        let args = vec!["nagual", "kos", "scoring", "health"];
        let cli = TestCli::try_parse_from(args);
        assert!(cli.is_ok());
    }

    #[test]
    fn test_kos_transfer_candidates() {
        let args = vec!["nagual", "kos", "transfer", "candidates", "rust", "python", "--limit", "5"];
        let cli = TestCli::try_parse_from(args);
        assert!(cli.is_ok());
    }

    #[test]
    fn test_kos_transfer_stats() {
        let args = vec!["nagual", "kos", "transfer", "stats"];
        let cli = TestCli::try_parse_from(args);
        assert!(cli.is_ok());
    }

    #[test]
    fn test_kos_epoch_create() {
        let args = vec!["nagual", "kos", "epoch", "create", "v1.0", "--description", "First release"];
        let cli = TestCli::try_parse_from(args);
        assert!(cli.is_ok());
    }

    #[test]
    fn test_kos_epoch_list() {
        let args = vec!["nagual", "kos", "epoch", "list"];
        let cli = TestCli::try_parse_from(args);
        assert!(cli.is_ok());
    }

    #[test]
    fn test_kos_epoch_diff() {
        let args = vec!["nagual", "kos", "epoch", "diff", "v1.0", "v2.0"];
        let cli = TestCli::try_parse_from(args);
        assert!(cli.is_ok());
    }

    #[test]
    fn test_kos_tiering_stats() {
        let args = vec!["nagual", "kos", "tiering", "stats"];
        let cli = TestCli::try_parse_from(args);
        assert!(cli.is_ok());
    }

    #[test]
    fn test_kos_tiering_hot() {
        let args = vec!["nagual", "kos", "tiering", "hot", "--limit", "5"];
        let cli = TestCli::try_parse_from(args);
        assert!(cli.is_ok());
    }

    #[test]
    fn test_kos_views_list() {
        let args = vec!["nagual", "kos", "views", "list"];
        let cli = TestCli::try_parse_from(args);
        assert!(cli.is_ok());
    }

    #[test]
    fn test_kos_views_register() {
        let args = vec!["nagual", "kos", "views", "register", "agent-1", "--mode", "include"];
        let cli = TestCli::try_parse_from(args);
        assert!(cli.is_ok());
    }

    #[test]
    fn test_kos_ewc_stats() {
        let args = vec!["nagual", "kos", "ewc", "stats"];
        let cli = TestCli::try_parse_from(args);
        assert!(cli.is_ok());
    }

    #[test]
    fn test_kos_ewc_fisher() {
        let args = vec!["nagual", "kos", "ewc", "fisher", "rust"];
        let cli = TestCli::try_parse_from(args);
        assert!(cli.is_ok());
    }

    #[test]
    fn test_kos_ewc_protection() {
        let args = vec!["nagual", "kos", "ewc", "protection", "pat-1", "rust"];
        let cli = TestCli::try_parse_from(args);
        assert!(cli.is_ok());
    }

    #[test]
    fn test_kos_ladder_stats() {
        let args = vec!["nagual", "kos", "ladder", "stats"];
        let cli = TestCli::try_parse_from(args);
        assert!(cli.is_ok());
    }

    #[test]
    fn test_kos_ladder_route() {
        let args = vec!["nagual", "kos", "ladder", "route", "how to fix timeout", "--complexity", "0.7"];
        let cli = TestCli::try_parse_from(args);
        assert!(cli.is_ok());
    }

    #[test]
    fn test_kos_ladder_reflex() {
        let args = vec!["nagual", "kos", "ladder", "reflex", "what is Rust"];
        let cli = TestCli::try_parse_from(args);
        assert!(cli.is_ok());
    }

    #[test]
    fn test_kos_hyperbolic_stats() {
        let args = vec!["nagual", "kos", "hyperbolic", "stats"];
        let cli = TestCli::try_parse_from(args);
        assert!(cli.is_ok());
    }
}
