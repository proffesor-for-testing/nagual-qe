//! Status command with terminal dashboard.
//!
//! Provides the `nagual status` command with optional TUI dashboard
//! using ratatui, and fallback to simple text output.

use std::path::PathBuf;

use chrono::{DateTime, Utc};
use clap::Args;
use rusqlite::Connection;
use serde::Serialize;

use crate::constitution::{Constitution, Principle};
use crate::drift::DriftTrend;
use crate::error::Result;
use crate::learning::get_drift_reports;

/// Status command for system overview.
///
/// Shows health status, learning stats, sync status, and system metrics.
/// Optionally uses a TUI dashboard if the `tui` feature is enabled.
#[derive(Args, Debug)]
pub struct StatusCommand {
    /// Enable TUI dashboard mode (requires `tui` feature).
    #[arg(long)]
    pub dashboard: bool,

    /// Refresh interval in seconds for dashboard mode.
    #[arg(long, default_value = "5")]
    pub refresh: u64,

    /// Show detailed information.
    #[arg(short, long)]
    pub detailed: bool,

    /// Output as JSON.
    #[arg(long)]
    pub json: bool,

    /// Path to SQLite database.
    #[arg(long, default_value = "./nagual.db")]
    pub db_path: PathBuf,

    /// Show only specific section: health, learning, sync, metrics, constitution.
    #[arg(long)]
    pub section: Option<String>,

    /// Show constitution greeting (random principle) at start.
    #[arg(long)]
    pub greeting: bool,
}

impl StatusCommand {
    /// Execute the status command.
    pub async fn run(&self) -> Result<()> {
        if self.dashboard {
            #[cfg(feature = "tui")]
            {
                return self.run_tui_dashboard().await;
            }
            #[cfg(not(feature = "tui"))]
            {
                eprintln!("TUI dashboard requires the `tui` feature. Using text output.");
            }
        }

        self.run_text_status().await
    }

    /// Run simple text-based status output.
    async fn run_text_status(&self) -> Result<()> {
        let status = self.collect_status().await?;

        if self.json {
            println!("{}", serde_json::to_string_pretty(&status)?);
            return Ok(());
        }

        // Optional greeting (random principle)
        if self.greeting {
            println!("{}", Constitution::startup_greeting());
            println!();
        }

        // Header
        println!();
        println!("========================================");
        println!("        Nagual System Status");
        println!("========================================");
        println!();

        // Show sections based on filter
        let show_all = self.section.is_none();
        let section = self.section.as_deref().unwrap_or("");

        // Constitution Summary (always show at top unless filtered)
        if show_all || section == "constitution" {
            self.print_constitution_section(&status);
        }

        // Health Status
        if show_all || section == "health" {
            self.print_health_section(&status);
        }

        // Learning Stats
        if show_all || section == "learning" {
            self.print_learning_section(&status);
        }

        // Drift Monitoring (always available, shown after learning)
        if show_all || section == "learning" || section == "drift" {
            self.print_drift_section();
        }

        // Meta-Cognitive (Strange Loop) status
        if show_all || section == "learning" || section == "meta" {
            self.print_meta_section();
        }

        // Domain Expansion (ruvector-domain-expansion)
        if show_all || section == "learning" || section == "expansion" {
            self.print_domain_expansion_section();
        }

        // Sync Status
        if show_all || section == "sync" {
            self.print_sync_section(&status);
        }

        // Metrics
        if show_all || section == "metrics" {
            self.print_metrics_section(&status);
        }

        println!("========================================");
        println!("Last updated: {}", Utc::now().format("%Y-%m-%d %H:%M:%S UTC"));
        println!();

        Ok(())
    }

    /// Collect all status information.
    async fn collect_status(&self) -> Result<SystemStatus> {
        // Collect constitution status
        let constitution = self.collect_constitution_status().await;

        // Collect health status
        let health = self.collect_health_status().await;

        // Collect learning stats from the real database
        let learning = self.collect_learning_stats().await;

        // Collect sync status
        let sync = self.collect_sync_status().await;

        // Collect system metrics
        let metrics = self.collect_system_metrics().await;

        Ok(SystemStatus {
            timestamp: Utc::now(),
            constitution,
            health,
            learning,
            sync,
            metrics,
        })
    }

    /// Collect constitution status.
    async fn collect_constitution_status(&self) -> ConstitutionStatus {
        let constitution = Constitution::new();
        let random_principle = Principle::random();

        ConstitutionStatus {
            principle_count: Principle::ALL.len(),
            rule_count: constitution.rule_count(),
            enforcement_mode: constitution.mode().to_string(),
            violations_24h: 0, // Future: track from audit log
            rule_violations_24h: 0, // Future: track from audit log
            random_principle: RandomPrincipleDisplay {
                number: random_principle.number(),
                name: random_principle.name().to_string(),
                summary: random_principle.summary().to_string(),
            },
        }
    }

    /// Collect health status by probing SQLite and checking disk.
    async fn collect_health_status(&self) -> HealthStatusSummary {
        let mut components = Vec::new();
        let mut healthy = 0u32;
        let mut degraded = 0u32;
        let mut unhealthy = 0u32;

        // Check SQLite connectivity
        let start = std::time::Instant::now();
        match Connection::open(&self.db_path) {
            Ok(conn) => {
                match conn.execute_batch("SELECT 1") {
                    Ok(_) => {
                        let latency = start.elapsed().as_secs_f64() * 1000.0;
                        components.push(ComponentHealth {
                            name: "sqlite".to_string(),
                            status: HealthStatusDisplay::Healthy,
                            message: format!(
                                "Database operational ({})",
                                self.db_path.display()
                            ),
                            latency_ms: Some(latency),
                        });
                        healthy += 1;
                    }
                    Err(e) => {
                        components.push(ComponentHealth {
                            name: "sqlite".to_string(),
                            status: HealthStatusDisplay::Unhealthy,
                            message: format!("Query failed: {}", e),
                            latency_ms: None,
                        });
                        unhealthy += 1;
                    }
                }
            }
            Err(e) => {
                components.push(ComponentHealth {
                    name: "sqlite".to_string(),
                    status: HealthStatusDisplay::Unhealthy,
                    message: format!("Cannot open: {}", e),
                    latency_ms: None,
                });
                unhealthy += 1;
            }
        }

        // Check disk / DB file size
        if let Ok(metadata) = std::fs::metadata(&self.db_path) {
            let db_size_mb = metadata.len() as f64 / (1024.0 * 1024.0);
            components.push(ComponentHealth {
                name: "disk".to_string(),
                status: HealthStatusDisplay::Healthy,
                message: format!("DB size: {:.1} MB", db_size_mb),
                latency_ms: Some(0.1),
            });
            healthy += 1;
        } else {
            components.push(ComponentHealth {
                name: "disk".to_string(),
                status: HealthStatusDisplay::Degraded,
                message: "DB file not found on disk".to_string(),
                latency_ms: None,
            });
            degraded += 1;
        }

        let overall = if unhealthy > 0 {
            HealthStatusDisplay::Unhealthy
        } else if degraded > 0 {
            HealthStatusDisplay::Degraded
        } else {
            HealthStatusDisplay::Healthy
        };

        HealthStatusSummary {
            overall,
            components,
            healthy_count: healthy as usize,
            degraded_count: degraded as usize,
            unhealthy_count: unhealthy as usize,
        }
    }

    /// Collect learning statistics from the SQLite database.
    async fn collect_learning_stats(&self) -> LearningStats {
        let fallback = LearningStats {
            total_patterns: 0,
            patterns_last_24h: 0,
            success_rate: 0.0,
            avg_reward: 0.0,
            consolidation_pending: 0,
            last_consolidation: None,
            top_domains: vec![],
        };

        let conn = match Connection::open(&self.db_path) {
            Ok(c) => c,
            Err(_) => return fallback,
        };

        // Total patterns
        let total_patterns: usize = conn
            .query_row(
                "SELECT COUNT(*) FROM reasoning_patterns",
                [],
                |row| row.get::<_, i64>(0),
            )
            .map(|v| v as usize)
            .unwrap_or(0);

        // Patterns in last 24 hours
        let patterns_last_24h: usize = conn
            .query_row(
                "SELECT COUNT(*) FROM reasoning_patterns WHERE created_at > datetime('now', '-1 day')",
                [],
                |row| row.get::<_, i64>(0),
            )
            .map(|v| v as usize)
            .unwrap_or(0);

        // Average reward
        let avg_reward: f64 = conn
            .query_row(
                "SELECT COALESCE(AVG(reward), 0.0) FROM reasoning_patterns",
                [],
                |row| row.get::<_, f64>(0),
            )
            .unwrap_or(0.0);

        // Success rate: average of (success_count / usage_count) where usage_count > 0
        let success_rate: f64 = conn
            .query_row(
                "SELECT COALESCE(AVG(CASE WHEN usage_count > 0 THEN CAST(success_count AS REAL) / usage_count ELSE 0 END), 0.0) FROM reasoning_patterns",
                [],
                |row| row.get::<_, f64>(0),
            )
            .unwrap_or(0.0);

        // Consolidation pending: patterns with low usage that could be merged
        let consolidation_pending: usize = conn
            .query_row(
                "SELECT COUNT(*) FROM reasoning_patterns WHERE reward < 0.3 AND usage_count < 2",
                [],
                |row| row.get::<_, i64>(0),
            )
            .map(|v| v as usize)
            .unwrap_or(0);

        // Top domains by pattern count
        let top_domains = conn
            .prepare(
                "SELECT domain, COUNT(*) as cnt, COALESCE(AVG(reward), 0.0) as avg_r \
                 FROM reasoning_patterns \
                 WHERE domain IS NOT NULL AND domain != '' \
                 GROUP BY domain ORDER BY cnt DESC LIMIT 5",
            )
            .and_then(|mut stmt| {
                let rows = stmt.query_map([], |row| {
                    Ok(DomainStats {
                        domain: row.get::<_, String>(0)?,
                        pattern_count: row.get::<_, i64>(1)? as usize,
                        avg_reward: row.get::<_, f64>(2)?,
                    })
                })?;
                rows.collect::<std::result::Result<Vec<_>, _>>()
            })
            .unwrap_or_default();

        LearningStats {
            total_patterns,
            patterns_last_24h,
            success_rate,
            avg_reward,
            consolidation_pending,
            last_consolidation: None,
            top_domains,
        }
    }

    /// Collect sync status by checking SQLite and PostgreSQL configuration.
    async fn collect_sync_status(&self) -> SyncStatus {
        // Check SQLite connectivity
        let sqlite_connected = Connection::open(&self.db_path)
            .and_then(|conn| conn.execute_batch("SELECT 1"))
            .is_ok();

        // Check if PostgreSQL is configured in ~/.nagual/config.toml or DATABASE_URL
        let pg_configured = resolve_postgres_url_for_status();

        let (postgres_connected, sync_mode) = if pg_configured {
            // PG is configured; we report it as configured but don't attempt
            // a full connection here (avoid blocking on network timeouts).
            (false, "dual-write".to_string())
        } else {
            (false, "sqlite-only".to_string())
        };

        SyncStatus {
            sqlite_connected,
            postgres_connected,
            last_sync: None,
            pending_sync_items: 0,
            conflicts_count: 0,
            sync_mode,
        }
    }

    /// Collect real system metrics (memory, DB size, uptime).
    async fn collect_system_metrics(&self) -> SystemMetrics {
        let (memory_used_mb, memory_total_mb) = get_memory_info();

        // DB file size
        let db_size_mb = std::fs::metadata(&self.db_path)
            .map(|m| m.len() as f64 / (1024.0 * 1024.0))
            .unwrap_or(0.0);

        let uptime_secs = get_process_uptime_secs();

        SystemMetrics {
            uptime_secs,
            memory_used_mb,
            memory_total_mb,
            cpu_usage_percent: 0.0, // Requires sysinfo crate; omitted
            db_size_mb,
            requests_total: 0,
            requests_per_second: 0.0,
            errors_total: 0,
            error_rate: 0.0,
        }
    }

    /// Print health section.
    fn print_health_section(&self, status: &SystemStatus) {
        println!("HEALTH STATUS");
        println!("----------------------------------------");

        let health = &status.health;
        let overall_icon = health.overall.icon();
        let overall_color = health.overall.color();
        let reset = "\x1b[0m";

        println!(
            "  Overall: {}{} {}{}",
            overall_color,
            overall_icon,
            health.overall,
            reset
        );
        println!(
            "  Components: {} healthy, {} degraded, {} unhealthy",
            health.healthy_count, health.degraded_count, health.unhealthy_count
        );

        if self.detailed {
            println!();
            for component in &health.components {
                let icon = component.status.icon();
                let color = component.status.color();
                let latency = component
                    .latency_ms
                    .map(|l| format!(" ({:.1}ms)", l))
                    .unwrap_or_default();

                println!(
                    "    {}{}{} {}: {}{}",
                    color, icon, reset, component.name, component.message, latency
                );
            }
        }

        println!();
    }

    /// Print learning section.
    fn print_learning_section(&self, status: &SystemStatus) {
        println!("LEARNING STATS");
        println!("----------------------------------------");

        let learning = &status.learning;

        println!("  Total Patterns: {}", learning.total_patterns);
        println!("  New (24h): {}", learning.patterns_last_24h);
        println!("  Success Rate: {:.1}%", learning.success_rate * 100.0);
        println!("  Avg Reward: {:.2}", learning.avg_reward);
        println!("  Pending Consolidation: {}", learning.consolidation_pending);

        if let Some(last) = learning.last_consolidation {
            let ago = Utc::now().signed_duration_since(last);
            println!("  Last Consolidation: {} ago", format_duration(ago));
        }

        if self.detailed && !learning.top_domains.is_empty() {
            println!();
            println!("  Top Domains:");
            for domain in &learning.top_domains {
                println!(
                    "    - {}: {} patterns, {:.2} avg reward",
                    domain.domain, domain.pattern_count, domain.avg_reward
                );
            }
        }

        println!();
    }

    /// Print drift monitoring section.
    ///
    /// Shows per-domain embedding drift reports from the global drift monitor.
    /// One line per domain with coefficient of variation and trend indicator.
    fn print_drift_section(&self) {
        println!("DRIFT MONITORING");
        println!("----------------------------------------");

        let reports = get_drift_reports();
        if reports.is_empty() {
            println!("  No drift data available.");
        } else {
            for report in &reports {
                let trend_label = match &report.trend {
                    DriftTrend::Stable => "stable",
                    DriftTrend::Increasing => "increasing",
                    DriftTrend::Decreasing => "decreasing",
                    DriftTrend::Insufficient => "insufficient data",
                };
                let warning = if report.is_drifting { " \x1b[33m!!\x1b[0m" } else { "" };
                println!(
                    "  {:<20} | CV: {:.3} | {}{}",
                    report.domain, report.coefficient_of_variation, trend_label, warning
                );
            }
        }

        println!();
    }

    /// Print meta-cognitive (strange loop) section.
    fn print_meta_section(&self) {
        println!("META-COGNITIVE (Strange Loop)");
        println!("----------------------------------------");

        let (avg_quality, health_rate, count) =
            crate::learning::get_meta_cognitive_stats();

        if count == 0 {
            println!("  No evaluations yet.");
        } else {
            println!(
                "  Evaluations: {} | Avg Quality: {:.3} | Health: {:.0}%",
                count,
                avg_quality,
                health_rate * 100.0
            );

            if let Some(latest) = crate::learning::get_meta_cognitive_status() {
                println!(
                    "  Latest: {} (score: {:.3})",
                    if latest.is_healthy {
                        "healthy"
                    } else {
                        "degraded"
                    },
                    latest.quality_score
                );
            }
        }

        println!();
    }

    /// Print domain expansion (Meta Thompson Sampling) section.
    fn print_domain_expansion_section(&self) {
        println!("DOMAIN EXPANSION");
        println!("----------------------------------------");

        let domains = crate::learning::get_expansion_domains();
        if domains.is_empty() {
            println!("  No domains registered.");
        } else {
            println!("  Domains: {}", domains.len());

            #[cfg(feature = "domain-expansion")]
            {
                if let Some(health) = crate::learning::get_expansion_health() {
                    println!(
                        "  Status: {} | Pareto: {} | Plateaus: {}",
                        if health.is_learning {
                            "learning"
                        } else {
                            "stalled"
                        },
                        health.pareto_size,
                        health.total_plateaus
                    );
                }
            }

            if self.detailed {
                for d in &domains {
                    println!("    - {}", d);
                }
            }
        }

        println!();
    }

    /// Print sync section.
    fn print_sync_section(&self, status: &SystemStatus) {
        println!("SYNC STATUS");
        println!("----------------------------------------");

        let sync = &status.sync;

        let sqlite_icon = if sync.sqlite_connected {
            "\x1b[32m[OK]\x1b[0m"
        } else {
            "\x1b[31m[X]\x1b[0m"
        };
        let postgres_icon = if sync.postgres_connected {
            "\x1b[32m[OK]\x1b[0m"
        } else {
            "\x1b[33m[--]\x1b[0m"
        };

        println!("  SQLite: {} Connected", sqlite_icon);
        println!("  PostgreSQL: {} {}", postgres_icon,
            if sync.postgres_connected { "Connected" } else { "Not configured" });
        println!("  Mode: {}", sync.sync_mode);
        println!("  Pending Items: {}", sync.pending_sync_items);
        println!("  Conflicts: {}", sync.conflicts_count);

        if let Some(last) = sync.last_sync {
            let ago = Utc::now().signed_duration_since(last);
            println!("  Last Sync: {} ago", format_duration(ago));
        }

        println!();
    }

    /// Print constitution section.
    fn print_constitution_section(&self, status: &SystemStatus) {
        println!("CONSTITUTION");
        println!("----------------------------------------");

        let c = &status.constitution;

        println!(
            "  Principles: {} | Rules: {} | Enforcement: {}",
            c.principle_count, c.rule_count, c.enforcement_mode
        );

        if c.violations_24h > 0 || c.rule_violations_24h > 0 {
            println!(
                "  Violations (24h): {} principle, {} rule",
                c.violations_24h, c.rule_violations_24h
            );
        }

        if self.detailed {
            println!();
            println!(
                "  Principle {}: {}",
                c.random_principle.number, c.random_principle.name
            );
            println!("    \"{}\"", c.random_principle.summary);
        }

        println!();
    }

    /// Print metrics section.
    fn print_metrics_section(&self, status: &SystemStatus) {
        println!("SYSTEM METRICS");
        println!("----------------------------------------");

        let metrics = &status.metrics;

        println!("  Uptime: {}", format_uptime(metrics.uptime_secs));
        if metrics.memory_total_mb > 0 {
            println!(
                "  Memory: {} MB / {} MB ({:.1}%)",
                metrics.memory_used_mb,
                metrics.memory_total_mb,
                (metrics.memory_used_mb as f64 / metrics.memory_total_mb as f64) * 100.0
            );
        } else {
            println!(
                "  Memory: {} MB used (total unavailable)",
                metrics.memory_used_mb
            );
        }
        println!("  CPU: {:.1}%", metrics.cpu_usage_percent);
        println!("  DB Size: {:.1} MB", metrics.db_size_mb);
        println!("  Requests: {} total ({:.1}/s)", metrics.requests_total, metrics.requests_per_second);
        println!(
            "  Errors: {} ({:.3}% error rate)",
            metrics.errors_total,
            metrics.error_rate * 100.0
        );

        println!();
    }

    /// Run TUI dashboard (when feature is enabled).
    #[cfg(feature = "tui")]
    async fn run_tui_dashboard(&self) -> Result<()> {
        use std::io;
        use std::time::Duration;

        use crossterm::{
            event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode},
            execute,
            terminal::{
                disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
            },
        };
        use ratatui::{backend::CrosstermBackend, Terminal};

        // Setup terminal
        enable_raw_mode().map_err(|e| crate::error::NagualError::internal(e.to_string()))?;
        let mut stdout = io::stdout();
        execute!(stdout, EnterAlternateScreen, EnableMouseCapture)
            .map_err(|e| crate::error::NagualError::internal(e.to_string()))?;
        let backend = CrosstermBackend::new(stdout);
        let mut terminal = Terminal::new(backend)
            .map_err(|e| crate::error::NagualError::internal(e.to_string()))?;

        let refresh_duration = Duration::from_secs(self.refresh);

        loop {
            // Collect status
            let status = self.collect_status().await?;
            let app = DashboardApp::from_status(status);

            // Draw UI
            terminal
                .draw(|f| {
                    app.render(f);
                })
                .map_err(|e| crate::error::NagualError::internal(e.to_string()))?;

            // Handle input
            if event::poll(refresh_duration)
                .map_err(|e| crate::error::NagualError::internal(e.to_string()))?
            {
                if let Event::Key(key) =
                    event::read().map_err(|e| crate::error::NagualError::internal(e.to_string()))?
                {
                    match key.code {
                        KeyCode::Char('q') => break,
                        KeyCode::Char('r') => continue, // Force refresh
                        _ => {}
                    }
                }
            }
        }

        // Restore terminal
        disable_raw_mode().map_err(|e| crate::error::NagualError::internal(e.to_string()))?;
        execute!(
            terminal.backend_mut(),
            LeaveAlternateScreen,
            DisableMouseCapture
        )
        .map_err(|e| crate::error::NagualError::internal(e.to_string()))?;
        terminal
            .show_cursor()
            .map_err(|e| crate::error::NagualError::internal(e.to_string()))?;

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// TUI Dashboard (behind `tui` feature gate)
// ---------------------------------------------------------------------------

/// Dashboard application holding a snapshot of system status for rendering.
#[cfg(feature = "tui")]
pub struct DashboardApp {
    /// Collected system status snapshot.
    pub status: SystemStatus,
}

#[cfg(feature = "tui")]
impl DashboardApp {
    /// Create a new DashboardApp from a SystemStatus snapshot.
    pub fn from_status(status: SystemStatus) -> Self {
        Self { status }
    }

    /// Render the full 4-panel dashboard into the given frame.
    pub fn render(&self, f: &mut ratatui::Frame) {
        use ratatui::{
            layout::{Constraint, Direction, Layout},
            style::{Color, Modifier, Style},
            widgets::{Block, Borders, Paragraph},
        };

        let outer = Layout::default()
            .direction(Direction::Vertical)
            .margin(1)
            .constraints([
                Constraint::Length(3), // Title bar
                Constraint::Min(10),   // Main content (4 panels)
                Constraint::Length(1), // Footer
            ])
            .split(f.size());

        // -- Title --
        let title = Paragraph::new("Nagual System Dashboard")
            .style(
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            )
            .block(Block::default().borders(Borders::ALL));
        f.render_widget(title, outer[0]);

        // -- 4-panel grid: 2x2 --
        let rows = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
            .split(outer[1]);

        let top_cols = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
            .split(rows[0]);

        let bottom_cols = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
            .split(rows[1]);

        // Top-left: Health
        render_health_panel(f, top_cols[0], &self.status.health);

        // Top-right: Learning Stats
        render_learning_panel(f, top_cols[1], &self.status.learning);

        // Bottom-left: Domain Breakdown
        render_domain_panel(f, bottom_cols[0], &self.status.learning);

        // Bottom-right: System Metrics
        render_metrics_panel(f, bottom_cols[1], &self.status.metrics);

        // -- Footer --
        let footer = Paragraph::new("Press 'q' to quit | 'r' to refresh")
            .style(Style::default().fg(Color::DarkGray));
        f.render_widget(footer, outer[2]);
    }
}

/// Map a HealthStatusDisplay to a ratatui Color.
#[cfg(feature = "tui")]
fn health_status_color(status: &HealthStatusDisplay) -> ratatui::style::Color {
    use ratatui::style::Color;
    match status {
        HealthStatusDisplay::Healthy => Color::Green,
        HealthStatusDisplay::Degraded => Color::Yellow,
        HealthStatusDisplay::Unhealthy => Color::Red,
        HealthStatusDisplay::Unknown => Color::Gray,
    }
}

/// Render the Health panel (top-left).
#[cfg(feature = "tui")]
fn render_health_panel(
    f: &mut ratatui::Frame,
    area: ratatui::layout::Rect,
    health: &HealthStatusSummary,
) {
    use ratatui::{
        style::Style,
        text::{Line, Span},
        widgets::{Block, Borders, Paragraph},
    };

    let status_color = health_status_color(&health.overall);

    let mut lines = vec![
        Line::from(vec![
            Span::raw("Overall: "),
            Span::styled(
                format!("{} {}", health.overall.icon(), health.overall),
                Style::default().fg(status_color),
            ),
        ]),
        Line::from(format!(
            "Components: {} healthy, {} degraded, {} unhealthy",
            health.healthy_count, health.degraded_count, health.unhealthy_count
        )),
        Line::from(""),
    ];

    for component in &health.components {
        let color = health_status_color(&component.status);
        let latency = component
            .latency_ms
            .map(|l| format!(" ({:.1}ms)", l))
            .unwrap_or_default();
        lines.push(Line::from(vec![
            Span::styled(
                format!("{} ", component.status.icon()),
                Style::default().fg(color),
            ),
            Span::raw(format!("{}: {}{}", component.name, component.message, latency)),
        ]));
    }

    let block = Block::default()
        .title(" Health Status ")
        .borders(Borders::ALL);
    let paragraph = Paragraph::new(lines).block(block);
    f.render_widget(paragraph, area);
}

/// Render the Learning Stats panel (top-right).
#[cfg(feature = "tui")]
fn render_learning_panel(
    f: &mut ratatui::Frame,
    area: ratatui::layout::Rect,
    learning: &LearningStats,
) {
    use ratatui::{
        layout::{Constraint, Direction, Layout},
        style::{Color, Style},
        text::Line,
        widgets::{Block, Borders, Gauge, Paragraph},
    };

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(3), Constraint::Min(3)])
        .split(area);

    // Success rate gauge (clamped to 0..=100)
    let pct = ((learning.success_rate * 100.0) as u16).min(100);
    let gauge_color = if pct >= 70 {
        Color::Green
    } else if pct >= 40 {
        Color::Yellow
    } else {
        Color::Red
    };
    let gauge = Gauge::default()
        .block(
            Block::default()
                .title(" Success Rate ")
                .borders(Borders::ALL),
        )
        .gauge_style(Style::default().fg(gauge_color))
        .percent(pct);
    f.render_widget(gauge, chunks[0]);

    // Text stats
    let text = vec![
        Line::from(format!(
            "Total Patterns: {} | New (24h): {}",
            learning.total_patterns, learning.patterns_last_24h
        )),
        Line::from(format!(
            "Avg Reward: {:.2} | Pending Consolidation: {}",
            learning.avg_reward, learning.consolidation_pending
        )),
    ];

    let block = Block::default()
        .title(" Learning Stats ")
        .borders(Borders::ALL);
    let paragraph = Paragraph::new(text).block(block);
    f.render_widget(paragraph, chunks[1]);
}

/// Render the Domain Breakdown panel (bottom-left).
#[cfg(feature = "tui")]
fn render_domain_panel(
    f: &mut ratatui::Frame,
    area: ratatui::layout::Rect,
    learning: &LearningStats,
) {
    use ratatui::{
        style::{Color, Modifier, Style},
        widgets::{Block, Borders, List, ListItem, Paragraph},
    };

    if learning.top_domains.is_empty() {
        let paragraph = Paragraph::new("No domain data available.")
            .block(
                Block::default()
                    .title(" Domain Breakdown ")
                    .borders(Borders::ALL),
            )
            .style(Style::default().fg(Color::DarkGray));
        f.render_widget(paragraph, area);
        return;
    }

    let items: Vec<ListItem> = learning
        .top_domains
        .iter()
        .map(|d| {
            let bar = build_bar(d.pattern_count, learning.total_patterns, 20);
            ListItem::new(format!(
                "{:<12} {:>4} patterns  {:.2} avg  {}",
                d.domain, d.pattern_count, d.avg_reward, bar,
            ))
        })
        .collect();

    let list = List::new(items)
        .block(
            Block::default()
                .title(" Domain Breakdown ")
                .borders(Borders::ALL),
        )
        .style(Style::default().fg(Color::White))
        .highlight_style(
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        );
    f.render_widget(list, area);
}

/// Render the System Metrics panel (bottom-right).
#[cfg(feature = "tui")]
fn render_metrics_panel(
    f: &mut ratatui::Frame,
    area: ratatui::layout::Rect,
    metrics: &SystemMetrics,
) {
    use ratatui::{
        layout::{Constraint, Direction, Layout},
        style::{Color, Style},
        text::Line,
        widgets::{Block, Borders, Gauge, Paragraph},
    };

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(3), Constraint::Min(3)])
        .split(area);

    // Memory gauge
    let mem_pct = if metrics.memory_total_mb > 0 {
        ((metrics.memory_used_mb as f64 / metrics.memory_total_mb as f64) * 100.0) as u16
    } else {
        0
    };
    let mem_pct = mem_pct.min(100);
    let mem_color = if mem_pct < 60 {
        Color::Green
    } else if mem_pct < 85 {
        Color::Yellow
    } else {
        Color::Red
    };
    let gauge = Gauge::default()
        .block(
            Block::default()
                .title(format!(
                    " Memory: {} MB / {} MB ",
                    metrics.memory_used_mb, metrics.memory_total_mb
                ))
                .borders(Borders::ALL),
        )
        .gauge_style(Style::default().fg(mem_color))
        .percent(mem_pct);
    f.render_widget(gauge, chunks[0]);

    // Text metrics
    let text = vec![
        Line::from(format!("Uptime: {}", format_uptime(metrics.uptime_secs))),
        Line::from(format!(
            "CPU: {:.1}% | DB Size: {:.1} MB",
            metrics.cpu_usage_percent, metrics.db_size_mb
        )),
        Line::from(format!(
            "Requests: {} ({:.1}/s) | Errors: {} ({:.3}%)",
            metrics.requests_total,
            metrics.requests_per_second,
            metrics.errors_total,
            metrics.error_rate * 100.0
        )),
    ];

    let block = Block::default()
        .title(" System Metrics ")
        .borders(Borders::ALL);
    let paragraph = Paragraph::new(text).block(block);
    f.render_widget(paragraph, chunks[1]);
}

/// Build a simple ASCII bar for proportional display.
///
/// Returns a string of filled and empty characters representing
/// the proportion `value / total` over `width` characters.
///
/// # Examples
/// ```ignore
/// assert_eq!(build_bar(50, 100, 10), "[=====     ]");
/// ```
fn build_bar(value: usize, total: usize, width: usize) -> String {
    if total == 0 || width == 0 {
        return format!("[{}]", " ".repeat(width));
    }
    let filled = ((value as f64 / total as f64) * width as f64).round() as usize;
    let filled = filled.min(width);
    let empty = width - filled;
    format!("[{}{}]", "=".repeat(filled), " ".repeat(empty))
}

/// Check if a PostgreSQL URL is configured (env var or config file).
/// Returns true if a non-empty URL is found; does not attempt a connection.
fn resolve_postgres_url_for_status() -> bool {
    // 1. Check DATABASE_URL environment variable
    if let Ok(url) = std::env::var("DATABASE_URL") {
        if !url.is_empty() {
            return true;
        }
    }

    // 2. Check ~/.nagual/config.toml
    if let Ok(home) = std::env::var("HOME") {
        let config_path = PathBuf::from(home).join(".nagual").join("config.toml");
        if let Ok(content) = std::fs::read_to_string(&config_path) {
            for line in content.lines() {
                let trimmed = line.trim();
                if trimmed.starts_with("postgres_url") {
                    if let Some(value) = trimmed.split('=').nth(1) {
                        let url = value.trim().trim_matches('"').trim_matches('\'');
                        if !url.is_empty() {
                            return true;
                        }
                    }
                }
            }
        }
    }

    false
}

/// Get total system memory and process RSS.
fn get_memory_info() -> (u64, u64) {
    let total_mb = get_total_memory_mb();
    let used_mb = get_process_rss_mb();
    (used_mb, total_mb)
}

/// Get total system memory in MB.
fn get_total_memory_mb() -> u64 {
    #[cfg(target_os = "macos")]
    {
        unsafe {
            let mut size: u64 = 0;
            let mut len = std::mem::size_of::<u64>();
            let mib = [libc::CTL_HW, libc::HW_MEMSIZE];
            let ret = libc::sysctl(
                mib.as_ptr() as *mut _,
                2,
                &mut size as *mut u64 as *mut libc::c_void,
                &mut len,
                std::ptr::null_mut(),
                0,
            );
            if ret == 0 {
                size / (1024 * 1024)
            } else {
                0
            }
        }
    }
    #[cfg(target_os = "linux")]
    {
        let pages = unsafe { libc::sysconf(libc::_SC_PHYS_PAGES) };
        let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        if pages > 0 && page_size > 0 {
            (pages as u64 * page_size as u64) / (1024 * 1024)
        } else {
            0
        }
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        0
    }
}

/// Get process RSS (resident set size) in MB.
fn get_process_rss_mb() -> u64 {
    #[cfg(target_os = "macos")]
    {
        // On macOS, ru_maxrss from getrusage is in bytes
        unsafe {
            let mut usage: libc::rusage = std::mem::zeroed();
            let ret = libc::getrusage(libc::RUSAGE_SELF, &mut usage);
            if ret == 0 {
                (usage.ru_maxrss as u64) / (1024 * 1024)
            } else {
                0
            }
        }
    }
    #[cfg(target_os = "linux")]
    {
        std::fs::read_to_string("/proc/self/status")
            .ok()
            .and_then(|s| {
                s.lines()
                    .find(|l| l.starts_with("VmRSS:"))
                    .and_then(|l| l.split_whitespace().nth(1))
                    .and_then(|v| v.parse::<u64>().ok())
                    .map(|kb| kb / 1024)
            })
            .unwrap_or(0)
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        0
    }
}

/// Get an approximate process uptime in seconds.
///
/// On Linux, reads `/proc/self/stat` for the actual process start time.
/// On other platforms, tracks elapsed time since first invocation.
fn get_process_uptime_secs() -> u64 {
    use std::sync::OnceLock;
    static START: OnceLock<std::time::Instant> = OnceLock::new();

    #[cfg(target_os = "linux")]
    {
        let clock_ticks = unsafe { libc::sysconf(libc::_SC_CLK_TCK) } as f64;
        let uptime: Option<f64> = std::fs::read_to_string("/proc/uptime")
            .ok()
            .and_then(|s| s.split_whitespace().next().and_then(|v| v.parse().ok()));
        let start_ticks: Option<f64> = std::fs::read_to_string("/proc/self/stat")
            .ok()
            .and_then(|s| {
                s.split_whitespace().nth(21).and_then(|v| v.parse().ok())
            });
        if let (Some(up), Some(st)) = (uptime, start_ticks) {
            let start_secs = st / clock_ticks;
            let process_up = up - start_secs;
            if process_up > 0.0 {
                return process_up as u64;
            }
        }
    }

    // Fallback (macOS and others): track elapsed time since first invocation
    let start = START.get_or_init(std::time::Instant::now);
    start.elapsed().as_secs()
}

/// Format a duration for display.
fn format_duration(duration: chrono::Duration) -> String {
    let secs = duration.num_seconds();
    if secs < 60 {
        format!("{}s", secs)
    } else if secs < 3600 {
        format!("{}m", secs / 60)
    } else if secs < 86400 {
        format!("{}h {}m", secs / 3600, (secs % 3600) / 60)
    } else {
        format!("{}d {}h", secs / 86400, (secs % 86400) / 3600)
    }
}

/// Format uptime for display.
fn format_uptime(secs: u64) -> String {
    if secs < 60 {
        format!("{}s", secs)
    } else if secs < 3600 {
        format!("{}m {}s", secs / 60, secs % 60)
    } else if secs < 86400 {
        format!("{}h {}m", secs / 3600, (secs % 3600) / 60)
    } else {
        format!("{}d {}h {}m", secs / 86400, (secs % 86400) / 3600, (secs % 3600) / 60)
    }
}

/// Overall system status.
#[derive(Debug, Clone, Serialize)]
pub struct SystemStatus {
    /// Timestamp of status collection.
    pub timestamp: DateTime<Utc>,
    /// Constitution status.
    pub constitution: ConstitutionStatus,
    /// Health status summary.
    pub health: HealthStatusSummary,
    /// Learning statistics.
    pub learning: LearningStats,
    /// Sync status.
    pub sync: SyncStatus,
    /// System metrics.
    pub metrics: SystemMetrics,
}

/// Constitution status.
#[derive(Debug, Clone, Serialize)]
pub struct ConstitutionStatus {
    /// Number of philosophical principles.
    pub principle_count: usize,
    /// Number of operational rules.
    pub rule_count: usize,
    /// Current enforcement mode.
    pub enforcement_mode: String,
    /// Principle violations in last 24h (placeholder for future tracking).
    pub violations_24h: usize,
    /// Rule violations in last 24h (placeholder for future tracking).
    pub rule_violations_24h: usize,
    /// Random principle for display.
    pub random_principle: RandomPrincipleDisplay,
}

/// Display info for a random principle.
#[derive(Debug, Clone, Serialize)]
pub struct RandomPrincipleDisplay {
    /// Principle number.
    pub number: u8,
    /// Principle name.
    pub name: String,
    /// Principle summary.
    pub summary: String,
}

/// Health status display enum (for serialization).
#[derive(Debug, Clone, Copy, Serialize)]
pub enum HealthStatusDisplay {
    Healthy,
    Degraded,
    Unhealthy,
    Unknown,
}

impl HealthStatusDisplay {
    /// Get the icon for display.
    pub fn icon(self) -> &'static str {
        match self {
            HealthStatusDisplay::Healthy => "[OK]",
            HealthStatusDisplay::Degraded => "[WARN]",
            HealthStatusDisplay::Unhealthy => "[FAIL]",
            HealthStatusDisplay::Unknown => "[?]",
        }
    }

    /// Get the color code for display.
    pub fn color(self) -> &'static str {
        match self {
            HealthStatusDisplay::Healthy => "\x1b[32m",   // Green
            HealthStatusDisplay::Degraded => "\x1b[33m",  // Yellow
            HealthStatusDisplay::Unhealthy => "\x1b[31m", // Red
            HealthStatusDisplay::Unknown => "\x1b[37m",   // White
        }
    }
}

impl std::fmt::Display for HealthStatusDisplay {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HealthStatusDisplay::Healthy => write!(f, "Healthy"),
            HealthStatusDisplay::Degraded => write!(f, "Degraded"),
            HealthStatusDisplay::Unhealthy => write!(f, "Unhealthy"),
            HealthStatusDisplay::Unknown => write!(f, "Unknown"),
        }
    }
}

/// Health status summary.
#[derive(Debug, Clone, Serialize)]
pub struct HealthStatusSummary {
    /// Overall health status.
    pub overall: HealthStatusDisplay,
    /// Individual component health.
    pub components: Vec<ComponentHealth>,
    /// Count of healthy components.
    pub healthy_count: usize,
    /// Count of degraded components.
    pub degraded_count: usize,
    /// Count of unhealthy components.
    pub unhealthy_count: usize,
}

/// Individual component health.
#[derive(Debug, Clone, Serialize)]
pub struct ComponentHealth {
    /// Component name.
    pub name: String,
    /// Health status.
    pub status: HealthStatusDisplay,
    /// Status message.
    pub message: String,
    /// Check latency in milliseconds.
    pub latency_ms: Option<f64>,
}

/// Learning statistics.
#[derive(Debug, Clone, Serialize)]
pub struct LearningStats {
    /// Total patterns stored.
    pub total_patterns: usize,
    /// Patterns added in last 24 hours.
    pub patterns_last_24h: usize,
    /// Success rate (0.0-1.0).
    pub success_rate: f64,
    /// Average reward across patterns.
    pub avg_reward: f64,
    /// Patterns pending consolidation.
    pub consolidation_pending: usize,
    /// Last consolidation timestamp.
    pub last_consolidation: Option<DateTime<Utc>>,
    /// Top domains by pattern count.
    pub top_domains: Vec<DomainStats>,
}

/// Domain statistics.
#[derive(Debug, Clone, Serialize)]
pub struct DomainStats {
    /// Domain name.
    pub domain: String,
    /// Pattern count.
    pub pattern_count: usize,
    /// Average reward.
    pub avg_reward: f64,
}

/// Sync status.
#[derive(Debug, Clone, Serialize)]
pub struct SyncStatus {
    /// SQLite connection status.
    pub sqlite_connected: bool,
    /// PostgreSQL connection status.
    pub postgres_connected: bool,
    /// Last sync timestamp.
    pub last_sync: Option<DateTime<Utc>>,
    /// Items pending sync.
    pub pending_sync_items: usize,
    /// Number of conflicts.
    pub conflicts_count: usize,
    /// Current sync mode.
    pub sync_mode: String,
}

/// System metrics.
#[derive(Debug, Clone, Serialize)]
pub struct SystemMetrics {
    /// Uptime in seconds.
    pub uptime_secs: u64,
    /// Memory used in MB.
    pub memory_used_mb: u64,
    /// Total memory in MB.
    pub memory_total_mb: u64,
    /// CPU usage percentage.
    pub cpu_usage_percent: f64,
    /// Database size in MB.
    pub db_size_mb: f64,
    /// Total requests processed.
    pub requests_total: u64,
    /// Requests per second.
    pub requests_per_second: f64,
    /// Total errors.
    pub errors_total: u64,
    /// Error rate (0.0-1.0).
    pub error_rate: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_duration() {
        assert_eq!(format_duration(chrono::Duration::seconds(30)), "30s");
        assert_eq!(format_duration(chrono::Duration::seconds(90)), "1m");
        assert_eq!(format_duration(chrono::Duration::seconds(3660)), "1h 1m");
        assert_eq!(format_duration(chrono::Duration::seconds(90000)), "1d 1h");
    }

    #[test]
    fn test_format_uptime() {
        assert_eq!(format_uptime(30), "30s");
        assert_eq!(format_uptime(90), "1m 30s");
        assert_eq!(format_uptime(3661), "1h 1m");
        assert_eq!(format_uptime(90061), "1d 1h 1m");
    }

    #[test]
    fn test_health_status_display() {
        assert_eq!(HealthStatusDisplay::Healthy.icon(), "[OK]");
        assert_eq!(HealthStatusDisplay::Degraded.icon(), "[WARN]");
        assert_eq!(HealthStatusDisplay::Unhealthy.icon(), "[FAIL]");
        assert_eq!(HealthStatusDisplay::Unknown.icon(), "[?]");
    }

    #[tokio::test]
    async fn test_collect_status_empty_db() {
        // With a fresh DB path, status should succeed but report zero patterns.
        let tmp_path = std::env::temp_dir().join("nagual_test_status_empty.db");
        let _ = std::fs::remove_file(&tmp_path);

        let cmd = StatusCommand {
            dashboard: false,
            refresh: 5,
            detailed: false,
            json: false,
            db_path: tmp_path.clone(),
            section: None,
            greeting: false,
        };

        let status = cmd.collect_status().await.unwrap();
        // SQLite creates the file on open, so health check passes
        assert!(status.health.healthy_count > 0);
        // No reasoning_patterns table exists, so total_patterns falls back to 0
        assert_eq!(status.learning.total_patterns, 0);
        // Constitution status should be populated
        assert_eq!(status.constitution.principle_count, 8);
        assert_eq!(status.constitution.rule_count, 5);
        let _ = std::fs::remove_file(&tmp_path);
    }

    #[tokio::test]
    async fn test_collect_status_with_patterns() {
        let tmp_path = std::env::temp_dir().join("nagual_test_status_with_patterns.db");
        let _ = std::fs::remove_file(&tmp_path);

        {
            let conn = Connection::open(&tmp_path).unwrap();
            conn.execute_batch(
                "CREATE TABLE reasoning_patterns (
                    id TEXT PRIMARY KEY,
                    problem TEXT,
                    solution TEXT,
                    domain TEXT,
                    tags TEXT,
                    reward REAL DEFAULT 0.5,
                    usage_count INTEGER DEFAULT 0,
                    success_count INTEGER DEFAULT 0,
                    created_at TEXT DEFAULT (datetime('now')),
                    updated_at TEXT DEFAULT (datetime('now'))
                );
                INSERT INTO reasoning_patterns (id, problem, solution, domain, reward, usage_count, success_count)
                VALUES ('p1', 'test problem', 'test solution', 'rust', 0.8, 5, 4);
                INSERT INTO reasoning_patterns (id, problem, solution, domain, reward, usage_count, success_count)
                VALUES ('p2', 'another problem', 'another solution', 'rust', 0.6, 3, 1);",
            )
            .unwrap();
        }

        let cmd = StatusCommand {
            dashboard: false,
            refresh: 5,
            detailed: false,
            json: false,
            db_path: tmp_path.clone(),
            section: None,
            greeting: false,
        };

        let status = cmd.collect_status().await.unwrap();
        assert!(status.health.healthy_count > 0);
        assert_eq!(status.learning.total_patterns, 2);
        assert!(status.learning.avg_reward > 0.0);
        assert_eq!(status.learning.top_domains.len(), 1);
        assert_eq!(status.learning.top_domains[0].domain, "rust");
        assert_eq!(status.learning.top_domains[0].pattern_count, 2);
        let _ = std::fs::remove_file(&tmp_path);
    }

    #[tokio::test]
    async fn test_system_metrics_real() {
        let tmp_path = std::env::temp_dir().join("nagual_test_metrics.db");
        let _ = std::fs::remove_file(&tmp_path);
        std::fs::write(&tmp_path, b"test data for size check").unwrap();

        let cmd = StatusCommand {
            dashboard: false,
            refresh: 5,
            detailed: false,
            json: false,
            db_path: tmp_path.clone(),
            section: None,
            greeting: false,
        };

        let metrics = cmd.collect_system_metrics().await;
        assert!(metrics.memory_total_mb > 0);
        assert!(metrics.db_size_mb > 0.0);
        let _ = std::fs::remove_file(&tmp_path);
    }

    #[test]
    fn test_resolve_postgres_url_for_status() {
        // Verify it does not panic; actual result depends on environment
        let _ = resolve_postgres_url_for_status();
    }

    #[test]
    fn test_build_bar_proportional() {
        // 50% filled over 10 chars
        assert_eq!(build_bar(50, 100, 10), "[=====     ]");
        // 100% filled
        assert_eq!(build_bar(100, 100, 10), "[==========]");
        // 0% filled
        assert_eq!(build_bar(0, 100, 10), "[          ]");
        // Edge: total is 0 (avoid division by zero)
        assert_eq!(build_bar(5, 0, 10), "[          ]");
        // Edge: width is 0
        assert_eq!(build_bar(5, 10, 0), "[]");
        // Edge: value exceeds total (should cap at width)
        assert_eq!(build_bar(200, 100, 10), "[==========]");
    }

    #[test]
    fn test_build_bar_rounding() {
        // 1 out of 3 over 10 chars -> 3.33 rounds to 3
        assert_eq!(build_bar(1, 3, 10), "[===       ]");
        // 2 out of 3 over 10 chars -> 6.67 rounds to 7
        assert_eq!(build_bar(2, 3, 10), "[=======   ]");
    }

    #[test]
    fn test_build_bar_small_width() {
        assert_eq!(build_bar(1, 2, 1), "[=]");
        assert_eq!(build_bar(0, 2, 1), "[ ]");
    }
}
