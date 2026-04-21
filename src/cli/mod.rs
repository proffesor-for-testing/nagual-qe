//! CLI module for Nagual
//!
//! Provides the command-line interface using clap derive macros.
//! Supports subcommands for migrations, health checks, conflict resolution,
//! graph operations, knowledge management, sync, and pattern management.

mod activity;
mod apikey;
mod cloud;
mod coherence;
pub(crate) mod common;
mod conflicts;
mod constitution;
mod dream;
mod graph;
mod health;
mod introspect;
mod knowledge;
mod learn;
mod migrate;
mod patterns;
mod plan;
mod predict;
pub mod research;
mod pulse;
mod session;
mod status;
mod sync;
mod transfuse;
mod user;

#[cfg(feature = "serve")]
mod serve;

mod kos;

pub use activity::ActivityCommand;
pub use apikey::ApiKeyCommand;
pub use cloud::CloudCommand;
pub use coherence::CoherenceCommand;
pub use conflicts::ConflictsCommand;
pub use constitution::ConstitutionCommand;
pub use dream::DreamCommand;
pub use graph::GraphCommand;
pub use health::HealthCommand;
pub use introspect::IntrospectCommand;
pub use knowledge::KnowledgeCommand;
pub use learn::LearnCommand;
pub use migrate::MigrateCommand;
pub use patterns::PatternsCommand;
pub use plan::PlanCommand;
pub use predict::PredictCommand;
pub use research::ResearchCommand;
pub use pulse::PulseCommand;
pub use session::SessionCommand;
pub use status::StatusCommand;
pub use sync::SyncCommand;
pub use transfuse::TransfuseCommand;
pub use user::UserCommand;

#[cfg(feature = "serve")]
pub use serve::ServeCommand;

pub use kos::KosCommand;

use clap::{Parser, Subcommand, ValueEnum};
use std::path::PathBuf;

use crate::constitution::EnforcementMode;

/// Nagual - Self-Learning Agentic System
///
/// A dual-write persistence system with SQLite (local) and PostgreSQL (cloud),
/// featuring ReasoningBank pattern storage, SONA learning, and Brier-calibrated predictions.
#[derive(Parser, Debug)]
#[command(name = "nagual")]
#[command(author = "Nagual Team")]
#[command(version = env!("CARGO_PKG_VERSION"))]
#[command(about = "Self-learning agentic system with dual-write persistence")]
#[command(propagate_version = true)]
pub struct Cli {
    /// Enable verbose output
    #[arg(short, long, global = true)]
    pub verbose: bool,

    /// Path to configuration file
    #[arg(short, long, global = true, value_name = "FILE")]
    pub config: Option<PathBuf>,

    /// Output results as JSON
    #[arg(long, global = true)]
    pub json: bool,

    /// Constitution enforcement mode (audit, warn, block)
    ///
    /// Can also be set via NAGUAL_CONSTITUTION_MODE env var or
    /// constitution_mode in ~/.nagual/config.toml
    #[arg(long, global = true, value_name = "MODE")]
    pub constitution_mode: Option<ConstitutionModeArg>,

    #[command(subcommand)]
    pub command: Commands,
}

/// Constitution enforcement mode argument for CLI.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum ConstitutionModeArg {
    /// Log violations, allow all operations (observation mode)
    Audit,
    /// Log violations, warn user, allow operations (default)
    Warn,
    /// Block operations that violate rules (strict mode)
    Block,
}

impl From<ConstitutionModeArg> for EnforcementMode {
    fn from(arg: ConstitutionModeArg) -> Self {
        match arg {
            ConstitutionModeArg::Audit => EnforcementMode::Audit,
            ConstitutionModeArg::Warn => EnforcementMode::Warn,
            ConstitutionModeArg::Block => EnforcementMode::Block,
        }
    }
}

/// Available subcommands for Nagual CLI
#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Activity tracking via Screenpipe integration
    ///
    /// Ingest, search, and summarize screen activity data from Screenpipe.
    /// Stores activity as patterns in the Nagual knowledge base.
    Activity(ActivityCommand),

    /// API key management for per-agent authentication
    ///
    /// Create, list, and revoke API keys with scoped access (read/write/admin)
    /// for authenticating agents against `nagual serve`.
    Apikey(ApiKeyCommand),

    /// Cloud sync - push/pull patterns to/from a remote nagual server
    ///
    /// Bidirectional incremental sync between local SQLite and a remote
    /// nagual serve instance. Configure the URL via `--remote`,
    /// `NAGUAL_CLOUD_URL` env var, or `~/.nagual/config.toml`.
    Cloud(CloudCommand),

    /// Coherence Gate for belief consistency verification
    ///
    /// Verifies belief consistency before allowing pattern storage.
    /// Detects contradictions, calculates coherence energy, and provides
    /// conflict resolution recommendations.
    Coherence(CoherenceCommand),

    /// Nagual Constitution - principles and rules
    ///
    /// Displays the 8 philosophical principles (rooted in Castaneda's
    /// Tonal/Nagual teachings) and the 5 operational rules that are
    /// runtime-enforced to protect knowledge integrity.
    Constitution(ConstitutionCommand),

    /// Dream Cycle - background maintenance and consolidation
    ///
    /// Runs background maintenance during idle periods: consolidates
    /// similar patterns, refreshes stale knowledge, calibrates predictions,
    /// and strengthens pattern connections via spreading activation.
    Dream(DreamCommand),

    /// Run database migrations
    ///
    /// Applies pending migrations to SQLite and/or PostgreSQL databases.
    /// Supports bidirectional sync and conflict detection.
    Migrate(MigrateCommand),

    /// Check system health
    ///
    /// Performs health checks on database connections, embedding models,
    /// and cloud sync status.
    Health(HealthCommand),

    /// Manage sync conflicts
    ///
    /// Lists, inspects, and resolves conflicts between local SQLite
    /// and cloud PostgreSQL databases.
    Conflicts(ConflictsCommand),

    /// Graph operations
    ///
    /// Provides GNN-style graph analysis including pressure propagation,
    /// edge creation, and neighbor queries for the knowledge graph.
    Graph(GraphCommand),

    /// Learning and self-improvement operations
    ///
    /// Run improvement cycles, view domain insights, trigger consolidation,
    /// record outcomes, and manage pattern recommendations.
    Learn(LearnCommand),

    /// Prediction management
    ///
    /// Create, resolve, and list predictions with Brier score calibration
    /// for tracking prediction accuracy over time.
    Predict(PredictCommand),

    /// Knowledge management
    ///
    /// Store, search, retrieve, and delete knowledge items in the
    /// ReasoningBank for self-learning capabilities.
    Knowledge(KnowledgeCommand),

    /// Sync and backup operations
    ///
    /// Manage database backups, restore from backups, check sync status,
    /// and view sync history between SQLite and PostgreSQL.
    Sync(SyncCommand),

    /// Pattern management
    ///
    /// Store, search, analyze, and consolidate patterns in the ReasoningBank
    /// for improved learning and pattern recognition.
    Patterns(PatternsCommand),

    /// Goal-Oriented Action Planning (GOAP)
    ///
    /// Create and execute plans to achieve goals using A* search through
    /// action space. Transforms natural language goals into executable
    /// step-by-step action sequences.
    Plan(PlanCommand),

    /// Research Swarm for autonomous knowledge acquisition
    ///
    /// Spawns ephemeral research agents to gather knowledge from multiple
    /// sources. Uses MaTTS (Memory-aware Test-Time Scaling) for quality
    /// control and automatically converts findings into stored patterns.
    Research(ResearchCommand),

    /// Strange Loop Introspection
    ///
    /// Self-referential system analysis that examines pattern health,
    /// domain coverage, temporal trends, and generates self-improvement
    /// recommendations. Inspired by Hofstadter's "strange loop" concept.
    Introspect(IntrospectCommand),

    /// Heartbeat visualization of pattern creation activity
    ///
    /// Renders a terminal heatmap showing daily pattern creation frequency
    /// over the past N weeks using Unicode block characters.
    Pulse(PulseCommand),

    /// Session management and analytics
    ///
    /// Manage development sessions and view token efficiency metrics.
    /// Track tokens used, patterns learned, and patterns retrieved per session.
    Session(SessionCommand),

    /// System status dashboard
    ///
    /// Shows overall system status including health, learning stats,
    /// sync status, and system metrics. Optionally displays a TUI dashboard.
    Status(StatusCommand),

    /// Gene Transfusion - extract patterns from codebases
    ///
    /// Scans source code files to detect common patterns (error handling,
    /// async patterns, API design, testing, database patterns) and stores
    /// them in the ReasoningBank for self-learning.
    Transfuse(TransfuseCommand),

    /// Browser-based dashboard server
    ///
    /// Starts an HTTP server with a web dashboard for browsing patterns,
    /// domains, pulse activity, and live event streaming via WebSocket.
    #[cfg(feature = "serve")]
    Serve(ServeCommand),

    /// Dashboard user management
    ///
    /// Create, list, and delete users for the browser-based dashboard login.
    /// Users authenticate via username/password to access the web dashboard.
    /// API keys (ngk_*) are separate — use `nagual apikey`.
    User(UserCommand),

    /// Knowledge Operating System (KOS) unified interface
    ///
    /// Provides access to all KOS subsystems: lineage tracking,
    /// witness chains, delta events, coherence scoring, domain transfer,
    /// epochs, tiering, agent views, EWC, routing ladder, and hyperbolic index.
    Kos(KosCommand),
}

/// Global options that can be passed to all commands
#[derive(Debug, Clone)]
pub struct GlobalOptions {
    pub verbose: bool,
    pub config: Option<PathBuf>,
    pub json: bool,
    pub constitution_mode: Option<EnforcementMode>,
}

impl From<&Cli> for GlobalOptions {
    fn from(cli: &Cli) -> Self {
        Self {
            verbose: cli.verbose,
            config: cli.config.clone(),
            json: cli.json,
            constitution_mode: cli.constitution_mode.map(EnforcementMode::from),
        }
    }
}

/// Resolve the constitution enforcement mode from multiple sources.
///
/// Priority (highest to lowest):
/// 1. CLI flag (`--constitution-mode`)
/// 2. Environment variable (`NAGUAL_CONSTITUTION_MODE`)
/// 3. Config file (`~/.nagual/config.toml` -> `constitution_mode`)
/// 4. Default (`warn`)
pub fn resolve_constitution_mode(cli_mode: Option<ConstitutionModeArg>) -> EnforcementMode {
    // 1. CLI flag takes precedence
    if let Some(mode) = cli_mode {
        return mode.into();
    }

    // 2. Environment variable
    if let Ok(env_mode) = std::env::var("NAGUAL_CONSTITUTION_MODE") {
        match env_mode.to_lowercase().as_str() {
            "audit" => return EnforcementMode::Audit,
            "warn" => return EnforcementMode::Warn,
            "block" => return EnforcementMode::Block,
            _ => {} // Fall through to config file
        }
    }

    // 3. Config file (~/.nagual/config.toml)
    if let Ok(home) = std::env::var("HOME") {
        let config_path = PathBuf::from(home).join(".nagual").join("config.toml");
        if let Ok(content) = std::fs::read_to_string(&config_path) {
            for line in content.lines() {
                let trimmed = line.trim();
                if trimmed.starts_with("constitution_mode") {
                    if let Some(value) = trimmed.split('=').nth(1) {
                        let mode_str = value.trim().trim_matches('"').trim_matches('\'');
                        match mode_str.to_lowercase().as_str() {
                            "audit" => return EnforcementMode::Audit,
                            "warn" => return EnforcementMode::Warn,
                            "block" => return EnforcementMode::Block,
                            _ => {} // Fall through to default
                        }
                    }
                }
            }
        }
    }

    // 4. Default
    EnforcementMode::default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cli_parse_migrate() {
        let args = vec!["nagual", "migrate", "--up"];
        let cli = Cli::try_parse_from(args);
        assert!(cli.is_ok());
    }

    #[test]
    fn test_cli_parse_health() {
        let args = vec!["nagual", "health"];
        let cli = Cli::try_parse_from(args);
        assert!(cli.is_ok());
    }

    #[test]
    fn test_cli_parse_conflicts() {
        let args = vec!["nagual", "conflicts", "list"];
        let cli = Cli::try_parse_from(args);
        assert!(cli.is_ok());
    }

    #[test]
    fn test_cli_global_options() {
        let args = vec!["nagual", "--verbose", "--json", "health"];
        let cli = Cli::try_parse_from(args).unwrap();
        assert!(cli.verbose);
        assert!(cli.json);
    }

    #[test]
    fn test_cli_parse_graph_pressure() {
        let args = vec!["nagual", "graph", "pressure", "rust", "--demo"];
        let cli = Cli::try_parse_from(args);
        assert!(cli.is_ok());
    }

    #[test]
    fn test_cli_parse_graph_pressure_with_options() {
        let args = vec![
            "nagual",
            "graph",
            "pressure",
            "node-id",
            "--depth",
            "5",
            "--damping",
            "0.9",
            "--top",
            "10",
            "--normalize",
            "--json",
            "--stats",
        ];
        let cli = Cli::try_parse_from(args);
        assert!(cli.is_ok());
    }

    #[test]
    fn test_cli_parse_graph_stats() {
        let args = vec!["nagual", "graph", "stats"];
        let cli = Cli::try_parse_from(args);
        assert!(cli.is_ok());
    }

    #[test]
    fn test_cli_parse_learn_improve() {
        let args = vec!["nagual", "learn", "improve", "rust.async", "--demo"];
        let cli = Cli::try_parse_from(args);
        assert!(cli.is_ok());
    }

    #[test]
    fn test_cli_parse_learn_insights() {
        let args = vec!["nagual", "learn", "insights", "rust", "--windows", "7d,30d"];
        let cli = Cli::try_parse_from(args);
        assert!(cli.is_ok());
    }

    #[test]
    fn test_cli_parse_learn_consolidate() {
        let args = vec!["nagual", "learn", "consolidate", "--trigger", "manual"];
        let cli = Cli::try_parse_from(args);
        assert!(cli.is_ok());
    }

    #[test]
    fn test_cli_parse_learn_recommendations() {
        let args = vec!["nagual", "learn", "recommendations", "--domain", "rust", "--demo"];
        let cli = Cli::try_parse_from(args);
        assert!(cli.is_ok());
    }

    #[test]
    fn test_cli_parse_predict_create() {
        let args = vec![
            "nagual",
            "predict",
            "create",
            "Deployment will succeed",
            "-p",
            "0.85",
        ];
        let cli = Cli::try_parse_from(args);
        assert!(cli.is_ok());
    }

    #[test]
    fn test_cli_parse_predict_resolve() {
        let args = vec!["nagual", "predict", "resolve", "pred-123", "true"];
        let cli = Cli::try_parse_from(args);
        assert!(cli.is_ok());
    }

    #[test]
    fn test_cli_parse_predict_list() {
        let args = vec![
            "nagual",
            "predict",
            "list",
            "--status",
            "pending",
            "--limit",
            "10",
        ];
        let cli = Cli::try_parse_from(args);
        assert!(cli.is_ok());
    }

    #[test]
    fn test_cli_parse_predict_calibration() {
        let args = vec!["nagual", "predict", "calibration", "--detailed", "--json"];
        let cli = Cli::try_parse_from(args);
        assert!(cli.is_ok());
    }

    #[test]
    fn test_cli_parse_predict_stats() {
        let args = vec!["nagual", "predict", "stats"];
        let cli = Cli::try_parse_from(args);
        assert!(cli.is_ok());
    }

    // Knowledge command tests
    #[test]
    fn test_cli_parse_knowledge_store() {
        let args = vec![
            "nagual",
            "knowledge",
            "store",
            "Test content",
            "--domain",
            "rust.async",
        ];
        let cli = Cli::try_parse_from(args);
        assert!(cli.is_ok());
    }

    #[test]
    fn test_cli_parse_knowledge_search() {
        let args = vec![
            "nagual",
            "knowledge",
            "search",
            "async error",
            "--limit",
            "10",
        ];
        let cli = Cli::try_parse_from(args);
        assert!(cli.is_ok());
    }

    #[test]
    fn test_cli_parse_knowledge_get() {
        let args = vec!["nagual", "knowledge", "get", "abc123"];
        let cli = Cli::try_parse_from(args);
        assert!(cli.is_ok());
    }

    #[test]
    fn test_cli_parse_knowledge_delete() {
        let args = vec!["nagual", "knowledge", "delete", "abc123", "--force"];
        let cli = Cli::try_parse_from(args);
        assert!(cli.is_ok());
    }

    // Sync command tests
    #[test]
    fn test_cli_parse_sync_backup() {
        // Use --compression with a value instead of --compress
        let args = vec!["nagual", "sync", "backup", "--compression", "9"];
        let cli = Cli::try_parse_from(args);
        assert!(cli.is_ok());
    }

    #[test]
    fn test_cli_parse_sync_restore() {
        let args = vec!["nagual", "sync", "restore", "backup.db.gz", "--force"];
        let cli = Cli::try_parse_from(args);
        assert!(cli.is_ok());
    }

    #[test]
    fn test_cli_parse_sync_status() {
        let args = vec!["nagual", "sync", "status", "--json"];
        let cli = Cli::try_parse_from(args);
        assert!(cli.is_ok());
    }

    #[test]
    fn test_cli_parse_sync_history() {
        // History is an option on status, not a separate subcommand
        let args = vec!["nagual", "sync", "status", "--history", "--limit", "10"];
        let cli = Cli::try_parse_from(args);
        assert!(cli.is_ok());
    }

    // Patterns command tests
    #[test]
    fn test_cli_parse_patterns_store() {
        let args = vec![
            "nagual",
            "patterns",
            "store",
            "--problem",
            "Test problem",
            "--solution",
            "Test solution",
        ];
        let cli = Cli::try_parse_from(args);
        assert!(cli.is_ok());
    }

    #[test]
    fn test_cli_parse_patterns_search() {
        let args = vec![
            "nagual",
            "patterns",
            "search",
            "retry backoff",
            "--limit",
            "5",
        ];
        let cli = Cli::try_parse_from(args);
        assert!(cli.is_ok());
    }

    #[test]
    fn test_cli_parse_patterns_stats() {
        let args = vec!["nagual", "patterns", "stats", "--detailed"];
        let cli = Cli::try_parse_from(args);
        assert!(cli.is_ok());
    }

    #[test]
    fn test_cli_parse_patterns_consolidate() {
        let args = vec![
            "nagual",
            "patterns",
            "consolidate",
            "--similarity",
            "0.9",
            "--dry-run",
        ];
        let cli = Cli::try_parse_from(args);
        assert!(cli.is_ok());
    }

    // Graph link and query tests
    #[test]
    fn test_cli_parse_graph_link() {
        let args = vec![
            "nagual",
            "graph",
            "link",
            "source-node",
            "target-node",
            "--weight",
            "0.85",
        ];
        let cli = Cli::try_parse_from(args);
        assert!(cli.is_ok());
    }

    #[test]
    fn test_cli_parse_graph_query() {
        let args = vec![
            "nagual",
            "graph",
            "query",
            "node-id",
            "--direction",
            "both",
            "--depth",
            "2",
        ];
        let cli = Cli::try_parse_from(args);
        assert!(cli.is_ok());
    }

    // Learn record test
    #[test]
    fn test_cli_parse_learn_record() {
        let args = vec![
            "nagual",
            "learn",
            "record",
            "pattern-123",
            "success",
            "--feedback",
            "Worked great",
        ];
        let cli = Cli::try_parse_from(args);
        assert!(cli.is_ok());
    }

    // Status command tests
    // Constitution command tests
    #[test]
    fn test_cli_parse_constitution() {
        let args = vec!["nagual", "constitution"];
        let cli = Cli::try_parse_from(args);
        assert!(cli.is_ok());
    }

    #[test]
    fn test_cli_parse_constitution_principle() {
        let args = vec!["nagual", "constitution", "--principle", "0"];
        let cli = Cli::try_parse_from(args);
        assert!(cli.is_ok());
    }

    #[test]
    fn test_cli_parse_constitution_rules() {
        let args = vec!["nagual", "constitution", "--rules"];
        let cli = Cli::try_parse_from(args);
        assert!(cli.is_ok());
    }

    #[test]
    fn test_cli_parse_constitution_json() {
        let args = vec!["nagual", "constitution", "--json", "--short"];
        let cli = Cli::try_parse_from(args);
        assert!(cli.is_ok());
    }

    #[test]
    fn test_cli_parse_constitution_random() {
        let args = vec!["nagual", "constitution", "--random"];
        let cli = Cli::try_parse_from(args);
        assert!(cli.is_ok());
    }

    #[test]
    fn test_cli_parse_status() {
        let args = vec!["nagual", "status"];
        let cli = Cli::try_parse_from(args);
        assert!(cli.is_ok());
    }

    #[test]
    fn test_cli_parse_status_json() {
        let args = vec!["nagual", "status", "--json"];
        let cli = Cli::try_parse_from(args);
        assert!(cli.is_ok());
    }

    #[test]
    fn test_cli_parse_status_detailed() {
        let args = vec!["nagual", "status", "--detailed"];
        let cli = Cli::try_parse_from(args);
        assert!(cli.is_ok());
    }

    #[test]
    fn test_cli_parse_status_section() {
        let args = vec!["nagual", "status", "--section", "health"];
        let cli = Cli::try_parse_from(args);
        assert!(cli.is_ok());
    }

    #[test]
    fn test_cli_parse_status_dashboard() {
        let args = vec!["nagual", "status", "--dashboard", "--refresh", "10"];
        let cli = Cli::try_parse_from(args);
        assert!(cli.is_ok());
    }

    #[test]
    fn test_cli_parse_status_greeting() {
        let args = vec!["nagual", "status", "--greeting"];
        let cli = Cli::try_parse_from(args);
        assert!(cli.is_ok());
    }

    #[test]
    fn test_cli_parse_status_constitution_section() {
        let args = vec!["nagual", "status", "--section", "constitution"];
        let cli = Cli::try_parse_from(args);
        assert!(cli.is_ok());
    }

    // Constitution mode global flag tests
    #[test]
    fn test_cli_parse_constitution_mode_audit() {
        let args = vec!["nagual", "--constitution-mode", "audit", "status"];
        let cli = Cli::try_parse_from(args).unwrap();
        assert_eq!(cli.constitution_mode, Some(ConstitutionModeArg::Audit));
    }

    #[test]
    fn test_cli_parse_constitution_mode_warn() {
        let args = vec!["nagual", "--constitution-mode", "warn", "health"];
        let cli = Cli::try_parse_from(args).unwrap();
        assert_eq!(cli.constitution_mode, Some(ConstitutionModeArg::Warn));
    }

    #[test]
    fn test_cli_parse_constitution_mode_block() {
        let args = vec!["nagual", "--constitution-mode", "block", "constitution"];
        let cli = Cli::try_parse_from(args).unwrap();
        assert_eq!(cli.constitution_mode, Some(ConstitutionModeArg::Block));
    }

    #[test]
    fn test_constitution_mode_arg_to_enforcement_mode() {
        use crate::constitution::EnforcementMode;
        assert_eq!(EnforcementMode::from(ConstitutionModeArg::Audit), EnforcementMode::Audit);
        assert_eq!(EnforcementMode::from(ConstitutionModeArg::Warn), EnforcementMode::Warn);
        assert_eq!(EnforcementMode::from(ConstitutionModeArg::Block), EnforcementMode::Block);
    }

    #[test]
    fn test_resolve_constitution_mode_default() {
        // With no CLI arg, env var, or config file, should return default (Warn)
        let mode = resolve_constitution_mode(None);
        assert_eq!(mode, crate::constitution::EnforcementMode::Warn);
    }

    #[test]
    fn test_resolve_constitution_mode_cli_override() {
        // CLI flag should take precedence
        let mode = resolve_constitution_mode(Some(ConstitutionModeArg::Block));
        assert_eq!(mode, crate::constitution::EnforcementMode::Block);
    }

    // Transfuse command tests
    #[test]
    fn test_cli_parse_transfuse_basic() {
        let args = vec!["nagual", "transfuse", "./src"];
        let cli = Cli::try_parse_from(args);
        assert!(cli.is_ok());
    }

    #[test]
    fn test_cli_parse_transfuse_dry_run() {
        let args = vec!["nagual", "transfuse", ".", "--dry-run"];
        let cli = Cli::try_parse_from(args);
        assert!(cli.is_ok());
    }

    #[test]
    fn test_cli_parse_transfuse_with_options() {
        let args = vec![
            "nagual",
            "transfuse",
            "/path/to/project",
            "--min-confidence",
            "0.8",
            "--extensions",
            "rs,go",
            "--max-files",
            "100",
            "--verbose",
            "--json",
        ];
        let cli = Cli::try_parse_from(args);
        assert!(cli.is_ok());
    }

    // Brain sync command tests (behind feature flag)
    #[cfg(feature = "brain-sync")]
    #[test]
    fn test_cli_parse_sync_brain_share() {
        let args = vec![
            "nagual",
            "sync",
            "brain",
            "share",
            "pattern-123",
            "--db-path",
            "nagual.db",
        ];
        let cli = Cli::try_parse_from(args);
        assert!(cli.is_ok());
    }

    #[cfg(feature = "brain-sync")]
    #[test]
    fn test_cli_parse_sync_brain_search() {
        let args = vec![
            "nagual",
            "sync",
            "brain",
            "search",
            "error handling",
            "--category",
            "rust",
            "--limit",
            "5",
        ];
        let cli = Cli::try_parse_from(args);
        assert!(cli.is_ok());
    }

    #[cfg(feature = "brain-sync")]
    #[test]
    fn test_cli_parse_sync_brain_status() {
        let args = vec!["nagual", "sync", "brain", "status"];
        let cli = Cli::try_parse_from(args);
        assert!(cli.is_ok());
    }

    // Migrate rewards/embeddings command tests
    #[test]
    fn test_cli_parse_migrate_rewards() {
        let args = vec!["nagual", "migrate", "rewards", "--dry-run"];
        let cli = Cli::try_parse_from(args);
        assert!(cli.is_ok());
    }

    #[test]
    fn test_cli_parse_migrate_rewards_with_db_path() {
        let args = vec![
            "nagual",
            "migrate",
            "rewards",
            "--db-path",
            "custom.db",
        ];
        let cli = Cli::try_parse_from(args);
        assert!(cli.is_ok());
    }

    #[test]
    fn test_cli_parse_migrate_embeddings() {
        let args = vec![
            "nagual",
            "migrate",
            "embeddings",
            "--missing-only",
            "--dry-run",
        ];
        let cli = Cli::try_parse_from(args);
        assert!(cli.is_ok());
    }

    #[test]
    fn test_cli_parse_migrate_embeddings_all() {
        let args = vec!["nagual", "migrate", "embeddings"];
        let cli = Cli::try_parse_from(args);
        assert!(cli.is_ok());
    }
}
