//! Dream Cycle CLI Command
//!
//! Background maintenance for pattern consolidation, refresh,
//! prediction calibration, and spreading activation.

use std::sync::Arc;
use clap::{Args, Subcommand};
use crate::db::SqliteDb;
use crate::dream::{DreamCycle, DreamConfig, DreamState};
use crate::error::Result;

/// Dream cycle management
#[derive(Debug, Args)]
pub struct DreamCommand {
    #[command(subcommand)]
    pub command: Option<DreamSubcommand>,

    /// Output as JSON
    #[arg(long, global = true)]
    pub json: bool,

    /// Path to the SQLite database
    #[arg(long, default_value = "nagual.db")]
    pub db_path: String,
}

/// Dream cycle subcommands
#[derive(Debug, Subcommand)]
pub enum DreamSubcommand {
    /// Run a dream cycle manually
    Run {
        /// Skip consolidation phase
        #[arg(long)]
        skip_consolidate: bool,
        /// Skip refresh phase
        #[arg(long)]
        skip_refresh: bool,
        /// Skip calibration phase
        #[arg(long)]
        skip_calibrate: bool,
        /// Skip activation phase
        #[arg(long)]
        skip_activate: bool,
        /// Maximum duration in seconds
        #[arg(long, default_value = "30")]
        max_duration: u64,
    },

    /// Show dream cycle status
    Status,

    /// Configure dream cycle settings
    Config {
        /// Show current configuration
        #[arg(long)]
        show: bool,
        /// Enable dream cycle
        #[arg(long)]
        enable: bool,
        /// Disable dream cycle
        #[arg(long)]
        disable: bool,
        /// Idle threshold in seconds
        #[arg(long)]
        idle_threshold: Option<u64>,
        /// Maximum duration in seconds
        #[arg(long)]
        max_duration: Option<u64>,
        /// Maximum patterns to consolidate
        #[arg(long)]
        max_consolidated: Option<usize>,
        /// Maximum patterns to refresh
        #[arg(long)]
        max_refreshed: Option<usize>,
    },

    /// Toggle individual phases
    Phases {
        /// Enable/disable consolidation
        #[arg(long)]
        consolidate: Option<bool>,
        /// Enable/disable refresh
        #[arg(long)]
        refresh: Option<bool>,
        /// Enable/disable calibration
        #[arg(long)]
        calibrate: Option<bool>,
        /// Enable/disable activation
        #[arg(long)]
        activate: Option<bool>,
    },

    /// View dream cycle history
    History {
        /// Number of cycles to show
        #[arg(long, default_value = "10")]
        limit: usize,
    },
}

impl DreamCommand {
    pub async fn run(&self) -> Result<()> {
        let db = Arc::new(SqliteDb::open(&self.db_path)?);

        // Ensure dream_cycles table exists
        db.execute(
            r#"CREATE TABLE IF NOT EXISTS dream_cycles (
                id TEXT PRIMARY KEY,
                started_at TEXT,
                completed_at TEXT,
                phases_json TEXT,
                total_duration_ms INTEGER,
                tokens_used INTEGER,
                items_processed INTEGER
            )"#,
            &[],
        ).await?;

        // Load config from database
        let config = self.load_config(&db).await;

        match &self.command {
            None => {
                // Default: run a dream cycle
                self.run_cycle(&db, config).await
            }
            Some(DreamSubcommand::Run {
                skip_consolidate,
                skip_refresh,
                skip_calibrate,
                skip_activate,
                max_duration,
            }) => {
                let mut config = config;
                config.phases.consolidate = !skip_consolidate;
                config.phases.refresh = !skip_refresh;
                config.phases.calibrate = !skip_calibrate;
                config.phases.activate = !skip_activate;
                config.max_duration_seconds = *max_duration;
                self.run_cycle(&db, config).await
            }
            Some(DreamSubcommand::Status) => self.show_status(&db, config).await,
            Some(DreamSubcommand::Config {
                show,
                enable,
                disable,
                idle_threshold,
                max_duration,
                max_consolidated,
                max_refreshed,
            }) => {
                if *show {
                    self.show_config(config);
                } else {
                    self.update_config(
                        &db,
                        config,
                        *enable,
                        *disable,
                        *idle_threshold,
                        *max_duration,
                        *max_consolidated,
                        *max_refreshed,
                    ).await?;
                }
                Ok(())
            }
            Some(DreamSubcommand::Phases {
                consolidate,
                refresh,
                calibrate,
                activate,
            }) => {
                self.update_phases(&db, config, *consolidate, *refresh, *calibrate, *activate).await
            }
            Some(DreamSubcommand::History { limit }) => self.show_history(&db, *limit).await,
        }
    }

    async fn run_cycle(&self, db: &Arc<SqliteDb>, config: DreamConfig) -> Result<()> {
        println!("🌙 Starting dream cycle...\n");

        // Create cycle with optional integrations
        let mut cycle = DreamCycle::new(db.clone(), config);

        // Try to set up ResearchCoordinator integration
        let research = std::sync::Arc::new(
            crate::research::ResearchCoordinator::with_defaults(db.clone())
        );
        cycle.set_research(research);

        // Try to set up GraphStorage integration
        if let Ok(graph) = crate::graph::GraphStorage::open(&self.db_path) {
            cycle.set_graph(std::sync::Arc::new(graph));
            println!("  Integrations: Research ✓, Graph ✓");
        } else {
            println!("  Integrations: Research ✓, Graph (fallback)");
        }
        println!();

        let result = cycle.run_cycle().await?;

        if self.json {
            println!("{}", serde_json::to_string_pretty(&result)?);
            return Ok(());
        }

        // Display results
        println!("Dream Cycle Results");
        println!("═══════════════════════════════════════════════════════════\n");

        println!("Cycle ID:    {}", result.cycle_id);
        println!("Duration:    {}ms", result.total_duration_ms);
        println!("Tokens Used: {}", result.tokens_used);
        println!();

        // Phase results
        for phase_result in &result.phases_completed {
            let status = if phase_result.success { "✅" } else { "❌" };
            println!("{} {} Phase", status, phase_result.phase);
            println!("   Items processed: {}", phase_result.items_processed);
            println!("   Duration: {}ms", phase_result.duration_ms);

            match &phase_result.details {
                crate::dream::PhaseDetails::Consolidate { patterns_merged, patterns_archived, duplicates_removed } => {
                    println!("   Merged: {}, Archived: {}, Deduped: {}",
                        patterns_merged, patterns_archived, duplicates_removed);
                }
                crate::dream::PhaseDetails::Refresh { patterns_refreshed, research_triggered, patterns_updated } => {
                    println!("   Refreshed: {}, Research triggered: {}, Updated: {}",
                        patterns_refreshed, research_triggered, patterns_updated);
                }
                crate::dream::PhaseDetails::Calibrate { predictions_reviewed, brier_score_before, brier_score_after } => {
                    println!("   Predictions: {}, Brier: {:.3} → {:.3}",
                        predictions_reviewed, brier_score_before, brier_score_after);
                }
                crate::dream::PhaseDetails::Activate { connections_strengthened, new_connections, activation_spread } => {
                    println!("   Strengthened: {}, New: {}, Spread: {:.2}",
                        connections_strengthened, new_connections, activation_spread);
                }
            }
            println!();
        }

        // Store result in history
        self.store_cycle_result(db, &result).await?;

        println!("───────────────────────────────────────────────────────────");
        println!("Total items processed: {}", result.total_items_processed());

        Ok(())
    }

    async fn show_status(&self, db: &Arc<SqliteDb>, config: DreamConfig) -> Result<()> {
        let cycle = DreamCycle::new(db.clone(), config);
        let status = cycle.status();

        if self.json {
            println!("{}", serde_json::to_string_pretty(&status)?);
            return Ok(());
        }

        println!("Dream Cycle Status");
        println!("═══════════════════════════════════════════════════════════\n");

        let state_emoji = match status.state {
            DreamState::Idle => "😴",
            DreamState::Running => "🌙",
            DreamState::Disabled => "🔇",
        };

        println!("State:          {} {}", state_emoji, status.state);
        println!("Enabled:        {}", if status.enabled { "Yes" } else { "No" });
        println!("Total Cycles:   {}", status.total_cycles);
        println!("Items Processed: {}", status.total_items_processed);

        if let Some(secs) = status.next_cycle_in_seconds {
            if secs == 0 {
                println!("Next Cycle:     Ready to run");
            } else {
                println!("Next Cycle:     In {}s", secs);
            }
        }

        if let Some(last) = &status.last_cycle {
            println!();
            println!("Last Cycle");
            println!("───────────────────────────────────────────────────────────");
            println!("  ID:       {}", last.cycle_id);
            println!("  Time:     {}", last.completed_at.format("%Y-%m-%d %H:%M:%S"));
            println!("  Duration: {}ms", last.total_duration_ms);
            println!("  Phases:   {}", last.phases_completed.len());
            println!("  Items:    {}", last.total_items_processed());
        }

        // Get history count
        let count: i64 = db.query_one(
            "SELECT COUNT(*) as count FROM dream_cycles",
            &[],
            |row| row.get(0),
        ).await?.unwrap_or(0);

        println!();
        println!("Historical Cycles: {}", count);

        Ok(())
    }

    fn show_config(&self, config: DreamConfig) {
        if self.json {
            println!("{}", serde_json::to_string_pretty(&config).unwrap());
            return;
        }

        println!("Dream Cycle Configuration");
        println!("═══════════════════════════════════════════════════════════\n");

        println!("General");
        println!("  Enabled:           {}", if config.enabled { "Yes" } else { "No" });
        println!("  Idle Threshold:    {}s", config.idle_threshold_seconds);
        println!("  Max Duration:      {}s", config.max_duration_seconds);
        println!();

        println!("Phases");
        println!("  Consolidate:       {}", if config.phases.consolidate { "✅" } else { "❌" });
        println!("  Refresh:           {}", if config.phases.refresh { "✅" } else { "❌" });
        println!("  Calibrate:         {}", if config.phases.calibrate { "✅" } else { "❌" });
        println!("  Activate:          {}", if config.phases.activate { "✅" } else { "❌" });
        println!();

        println!("Budget");
        println!("  Max Consolidated:  {}", config.budget.max_patterns_consolidated);
        println!("  Max Refreshed:     {}", config.budget.max_patterns_refreshed);
        println!("  Max Calibrated:    {}", config.budget.max_predictions_calibrated);
        println!("  Max Tokens:        {}", config.budget.max_tokens_per_cycle);
    }

    async fn update_config(
        &self,
        db: &Arc<SqliteDb>,
        mut config: DreamConfig,
        enable: bool,
        disable: bool,
        idle_threshold: Option<u64>,
        max_duration: Option<u64>,
        max_consolidated: Option<usize>,
        max_refreshed: Option<usize>,
    ) -> Result<()> {
        if enable {
            config.enabled = true;
        }
        if disable {
            config.enabled = false;
        }
        if let Some(t) = idle_threshold {
            config.idle_threshold_seconds = t;
        }
        if let Some(d) = max_duration {
            config.max_duration_seconds = d;
        }
        if let Some(c) = max_consolidated {
            config.budget.max_patterns_consolidated = c;
        }
        if let Some(r) = max_refreshed {
            config.budget.max_patterns_refreshed = r;
        }

        self.save_config(db, &config).await?;

        println!("✅ Configuration updated");
        self.show_config(config);

        Ok(())
    }

    async fn update_phases(
        &self,
        db: &Arc<SqliteDb>,
        mut config: DreamConfig,
        consolidate: Option<bool>,
        refresh: Option<bool>,
        calibrate: Option<bool>,
        activate: Option<bool>,
    ) -> Result<()> {
        if let Some(v) = consolidate {
            config.phases.consolidate = v;
        }
        if let Some(v) = refresh {
            config.phases.refresh = v;
        }
        if let Some(v) = calibrate {
            config.phases.calibrate = v;
        }
        if let Some(v) = activate {
            config.phases.activate = v;
        }

        self.save_config(db, &config).await?;

        println!("✅ Phase configuration updated");
        println!();
        println!("Phases");
        println!("  Consolidate: {}", if config.phases.consolidate { "✅" } else { "❌" });
        println!("  Refresh:     {}", if config.phases.refresh { "✅" } else { "❌" });
        println!("  Calibrate:   {}", if config.phases.calibrate { "✅" } else { "❌" });
        println!("  Activate:    {}", if config.phases.activate { "✅" } else { "❌" });

        Ok(())
    }

    async fn show_history(&self, db: &Arc<SqliteDb>, limit: usize) -> Result<()> {
        let limit_str = limit.to_string();
        let rows: Vec<(String, String, String, i64, i64, i64)> = db.query(
            r#"
            SELECT id, started_at, completed_at, total_duration_ms, tokens_used, items_processed
            FROM dream_cycles
            ORDER BY started_at DESC
            LIMIT ?
            "#,
            &[&limit_str],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?, row.get(4)?, row.get(5)?)),
        ).await?;

        if self.json {
            let history: Vec<serde_json::Value> = rows.iter().map(|(id, started, completed, duration, tokens, items)| {
                serde_json::json!({
                    "id": id,
                    "started_at": started,
                    "completed_at": completed,
                    "total_duration_ms": duration,
                    "tokens_used": tokens,
                    "items_processed": items,
                })
            }).collect();
            println!("{}", serde_json::to_string_pretty(&history)?);
            return Ok(());
        }

        println!("Dream Cycle History");
        println!("═══════════════════════════════════════════════════════════\n");

        if rows.is_empty() {
            println!("No dream cycles recorded yet.");
            println!("\nRun a cycle with: nagual dream");
            return Ok(());
        }

        println!("{:<36} {:>10} {:>8} {:>8}",
            "Cycle ID", "Duration", "Tokens", "Items");
        println!("───────────────────────────────────────────────────────────");

        for (id, _started, _completed, duration, tokens, items) in &rows {
            println!("{:<36} {:>8}ms {:>8} {:>8}",
                &id[..36.min(id.len())],
                duration,
                tokens,
                items);
        }

        println!();
        println!("Showing {} cycles", rows.len());

        Ok(())
    }

    async fn load_config(&self, db: &Arc<SqliteDb>) -> DreamConfig {
        // Ensure config table exists
        let _ = db.execute(
            r#"CREATE TABLE IF NOT EXISTS dream_config (
                key TEXT PRIMARY KEY,
                value TEXT
            )"#,
            &[],
        ).await;

        // Try to load from database
        let result: Option<String> = db.query_one(
            "SELECT value FROM dream_config WHERE key = 'config'",
            &[],
            |row| row.get(0),
        ).await.unwrap_or(None);

        if let Some(json) = result {
            if let Ok(config) = serde_json::from_str(&json) {
                return config;
            }
        }

        DreamConfig::default()
    }

    async fn save_config(&self, db: &Arc<SqliteDb>, config: &DreamConfig) -> Result<()> {
        let json = serde_json::to_string(config)?;

        db.execute(
            "INSERT OR REPLACE INTO dream_config (key, value) VALUES ('config', ?)",
            &[&json],
        ).await?;

        Ok(())
    }

    async fn store_cycle_result(&self, db: &Arc<SqliteDb>, result: &crate::dream::DreamResult) -> Result<()> {
        let phases_json = serde_json::to_string(&result.phases_completed)?;

        db.execute(
            r#"
            INSERT INTO dream_cycles (id, started_at, completed_at, phases_json, total_duration_ms, tokens_used, items_processed)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            "#,
            &[
                &result.cycle_id,
                &result.started_at.to_rfc3339(),
                &result.completed_at.to_rfc3339(),
                &phases_json,
                &(result.total_duration_ms as i64),
                &(result.tokens_used as i64),
                &(result.total_items_processed() as i64),
            ],
        ).await?;

        Ok(())
    }
}
