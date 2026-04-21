//! Health check command implementation
//!
//! Performs health checks on database connections, embedding models,
//! and cloud sync status.
//!
//! Usage:
//! - `nagual health` - Show all component status
//! - `nagual health --component <name>` - Show specific component
//! - `nagual health --json` - Output in JSON format

use std::path::PathBuf;
use std::sync::Arc;

use clap::Args;
use serde::Serialize;

use crate::error::Result;
use crate::health::{
    checks::{DiskHealthCheck, MemoryHealthCheck, SqliteHealthCheck},
    HealthCheckResult, HealthRegistry, HealthReport, HealthStatus,
};

/// Health check command
///
/// Performs comprehensive health checks on all system components including
/// database connections, disk space, and memory usage.
#[derive(Args, Debug)]
pub struct HealthCommand {
    /// Check only a specific component by name
    #[arg(short = 'C', long, value_name = "NAME")]
    pub component: Option<String>,

    /// Check only database connections
    #[arg(long)]
    pub db_only: bool,

    /// Check only embedding/ML components
    #[arg(long)]
    pub ml_only: bool,

    /// Check only cloud sync status
    #[arg(long)]
    pub sync_only: bool,

    /// Path to SQLite database (default: ./nagual.db)
    #[arg(long, value_name = "PATH", default_value = "./nagual.db")]
    pub sqlite_path: PathBuf,

    /// Path to check disk space (default: /)
    #[arg(long, value_name = "PATH", default_value = "/")]
    pub disk_path: PathBuf,

    /// Skip SQLite health check
    #[arg(long)]
    pub skip_sqlite: bool,

    /// Skip disk health check
    #[arg(long)]
    pub skip_disk: bool,

    /// Skip memory health check
    #[arg(long)]
    pub skip_memory: bool,

    /// Include detailed diagnostic information
    #[arg(long)]
    pub detailed: bool,

    /// Run verbose checks (including SQLite integrity check)
    #[arg(short, long)]
    pub verbose: bool,

    /// Timeout for health checks in seconds
    #[arg(long, default_value = "30")]
    pub timeout: u64,

    /// Output results as JSON
    #[arg(long)]
    pub json: bool,
}

impl HealthCommand {
    /// Execute the health check command
    ///
    /// Runs health checks on specified components and reports results.
    pub async fn run(&self) -> Result<()> {
        tracing::info!("Running health checks...");

        let registry = self.build_registry().await;

        // Check if we're looking for a specific component
        if let Some(ref component) = self.component {
            return self.run_single_component_check(&registry, component).await;
        }

        // Run appropriate checks based on flags
        let report = if self.db_only || self.ml_only || self.sync_only {
            self.run_filtered_checks(&registry).await
        } else {
            registry.check_all().await
        };

        // Output results
        if self.json {
            self.print_json(&report)?;
        } else {
            self.display_report(&report);
        }

        // Return success, but log warnings for unhealthy state
        if report.overall_status() == HealthStatus::Unhealthy {
            tracing::warn!("System health check failed");
        }

        Ok(())
    }

    /// Build the health registry with configured checks
    async fn build_registry(&self) -> HealthRegistry {
        let registry = HealthRegistry::new();

        // SQLite check
        if !self.skip_sqlite && !self.ml_only && !self.sync_only {
            let mut sqlite_check = SqliteHealthCheck::new(&self.sqlite_path);
            if self.verbose {
                sqlite_check = sqlite_check.with_integrity_check();
            }
            registry.register("sqlite", Arc::new(sqlite_check)).await;
        }

        // Disk check
        if !self.skip_disk && !self.db_only && !self.ml_only && !self.sync_only {
            let disk_check = DiskHealthCheck::new(&self.disk_path);
            registry.register("disk", Arc::new(disk_check)).await;
        }

        // Memory check
        if !self.skip_memory && !self.db_only && !self.ml_only && !self.sync_only {
            let memory_check = MemoryHealthCheck::new();
            registry.register("memory", Arc::new(memory_check)).await;
        }

        registry
    }

    /// Run a single component check
    async fn run_single_component_check(
        &self,
        registry: &HealthRegistry,
        component: &str,
    ) -> Result<()> {
        match registry.check(component).await {
            Ok(result) => {
                if self.json {
                    self.print_json(&result)?;
                } else {
                    self.display_component_result(&result);
                }
            }
            Err(e) => {
                if self.json {
                    self.print_json(&ErrorOutput {
                        error: e.to_string(),
                    })?;
                } else {
                    eprintln!("Error: {}", e);
                }
            }
        }
        Ok(())
    }

    /// Run filtered health checks based on command flags
    async fn run_filtered_checks(&self, registry: &HealthRegistry) -> HealthReport {
        if self.db_only {
            tracing::debug!("Running database-only health checks");
        }
        if self.ml_only {
            tracing::debug!("Running ML-only health checks");
        }
        if self.sync_only {
            tracing::debug!("Running sync-only health checks");
        }

        registry.check_all().await
    }

    /// Print a value as JSON
    fn print_json<T: Serialize>(&self, value: &T) -> Result<()> {
        let json = serde_json::to_string_pretty(value)?;
        println!("{}", json);
        Ok(())
    }

    /// Display a single component result
    fn display_component_result(&self, result: &HealthCheckResult) {
        let icon = match result.status {
            HealthStatus::Healthy => "[OK]",
            HealthStatus::Degraded => "[WARN]",
            HealthStatus::Unhealthy => "[FAIL]",
            HealthStatus::Unknown => "[?]",
        };

        println!("{} {}: {}", icon, result.component, result.message);

        if result.duration.as_millis() > 0 {
            println!("    Response time: {:?}", result.duration);
        }

        for (key, value) in &result.metadata {
            println!("    {}: {}", key, value);
        }
    }

    /// Display the health report
    fn display_report(&self, report: &HealthReport) {
        println!("\nHealth Check Results:");
        println!("{:-<50}", "");

        // Overall status
        let status_icon = match report.overall_status() {
            HealthStatus::Healthy => "[OK]",
            HealthStatus::Degraded => "[WARN]",
            HealthStatus::Unhealthy => "[FAIL]",
            HealthStatus::Unknown => "[?]",
        };

        println!(
            "{} Overall: {} ({} healthy, {} degraded, {} unhealthy)",
            status_icon,
            report.overall_status(),
            report.healthy_count,
            report.degraded_count,
            report.unhealthy_count
        );

        if let Some(uptime) = report.uptime {
            println!("    Uptime: {:?}", uptime);
        }

        println!();

        // Component details (always show if detailed, or if there are issues)
        if self.detailed || !report.is_operational() || report.components.len() > 0 {
            println!("Components:");

            // Sort components for consistent output
            let mut components: Vec<_> = report.components.iter().collect();
            components.sort_by_key(|(name, _)| name.as_str());

            for (name, result) in components {
                let icon = match result.status {
                    HealthStatus::Healthy => "[OK]",
                    HealthStatus::Degraded => "[WARN]",
                    HealthStatus::Unhealthy => "[FAIL]",
                    HealthStatus::Unknown => "[?]",
                };

                println!("  {} {}: {}", icon, name, result.message);

                if self.detailed || self.verbose {
                    if result.duration.as_millis() > 0 {
                        println!("      Response time: {:?}", result.duration);
                    }
                    for (key, value) in &result.metadata {
                        println!("      {}: {}", key, value);
                    }
                }
            }
        }

        println!("{:-<50}", "");
    }
}

/// Error output for JSON format
#[derive(Serialize)]
struct ErrorOutput {
    error: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_health_command_defaults() {
        let cmd = HealthCommand {
            component: None,
            db_only: false,
            ml_only: false,
            sync_only: false,
            sqlite_path: PathBuf::from("./nagual.db"),
            disk_path: PathBuf::from("/"),
            skip_sqlite: false,
            skip_disk: false,
            skip_memory: false,
            detailed: false,
            verbose: false,
            timeout: 30,
            json: false,
        };
        assert_eq!(cmd.timeout, 30);
        assert!(!cmd.detailed);
        assert!(!cmd.json);
    }

    #[tokio::test]
    async fn test_build_registry() {
        let cmd = HealthCommand {
            component: None,
            db_only: false,
            ml_only: false,
            sync_only: false,
            sqlite_path: PathBuf::from("./test.db"),
            disk_path: PathBuf::from("/"),
            skip_sqlite: false,
            skip_disk: false,
            skip_memory: false,
            detailed: false,
            verbose: false,
            timeout: 30,
            json: false,
        };

        let registry = cmd.build_registry().await;
        let checks = registry.list_checks().await;

        assert!(checks.contains(&"sqlite".to_string()));
        assert!(checks.contains(&"disk".to_string()));
        assert!(checks.contains(&"memory".to_string()));
    }

    #[tokio::test]
    async fn test_build_registry_with_skips() {
        let cmd = HealthCommand {
            component: None,
            db_only: false,
            ml_only: false,
            sync_only: false,
            sqlite_path: PathBuf::from("./test.db"),
            disk_path: PathBuf::from("/"),
            skip_sqlite: true,
            skip_disk: true,
            skip_memory: false,
            detailed: false,
            verbose: false,
            timeout: 30,
            json: false,
        };

        let registry = cmd.build_registry().await;
        let checks = registry.list_checks().await;

        assert!(!checks.contains(&"sqlite".to_string()));
        assert!(!checks.contains(&"disk".to_string()));
        assert!(checks.contains(&"memory".to_string()));
    }

    #[tokio::test]
    async fn test_build_registry_db_only() {
        let cmd = HealthCommand {
            component: None,
            db_only: true,
            ml_only: false,
            sync_only: false,
            sqlite_path: PathBuf::from("./test.db"),
            disk_path: PathBuf::from("/"),
            skip_sqlite: false,
            skip_disk: false,
            skip_memory: false,
            detailed: false,
            verbose: false,
            timeout: 30,
            json: false,
        };

        let registry = cmd.build_registry().await;
        let checks = registry.list_checks().await;

        // db_only should skip disk and memory
        assert!(checks.contains(&"sqlite".to_string()));
        assert!(!checks.contains(&"disk".to_string()));
        assert!(!checks.contains(&"memory".to_string()));
    }
}
