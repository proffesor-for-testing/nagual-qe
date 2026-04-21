//! Introspect command for Strange Loop self-analysis

use clap::Args;
use std::path::PathBuf;
use std::sync::Arc;
use tracing::info;

use crate::db::SqliteDb;
use crate::error::NagualError;
use crate::introspection::{
    HealthStatus, IntrospectionConfig, IntrospectionEngine, SelfModel, Severity,
};

/// Introspect command for self-analysis
#[derive(Args, Debug)]
pub struct IntrospectCommand {
    /// Path to the SQLite database
    #[arg(long, default_value = "nagual.db")]
    pub db_path: PathBuf,

    /// Focus on a specific domain
    #[arg(long, short = 'd')]
    pub domain: Option<String>,

    /// Show only vulnerabilities
    #[arg(long)]
    pub vulnerabilities: bool,

    /// Show only recommendations
    #[arg(long)]
    pub recommendations: bool,

    /// Quick health check (exit code reflects health)
    #[arg(long)]
    pub health_check: bool,

    /// Maximum number of recommendations to show
    #[arg(long, default_value = "5")]
    pub max_recommendations: usize,

    /// Stale pattern threshold in days
    #[arg(long, default_value = "30")]
    pub stale_days: u64,
}

impl IntrospectCommand {
    pub async fn execute(&self, json_output: bool) -> Result<(), NagualError> {
        info!("Running Strange Loop introspection");

        // Initialize database connection
        let db = Arc::new(SqliteDb::open(&self.db_path)?);

        // Create introspection engine
        let intro_config = IntrospectionConfig {
            stale_threshold_days: self.stale_days,
            ..Default::default()
        };
        let engine = IntrospectionEngine::new(db, intro_config);

        if self.health_check {
            return self.run_health_check(&engine, json_output).await;
        }

        // Full introspection
        let model = engine.introspect().await?;

        if json_output {
            println!("{}", serde_json::to_string_pretty(&model).unwrap_or_default());
            return Ok(());
        }

        if self.vulnerabilities {
            self.print_vulnerabilities(&model);
        } else if self.recommendations {
            self.print_recommendations(&model, self.max_recommendations);
        } else {
            self.print_full_report(&model);
        }

        Ok(())
    }

    async fn run_health_check(
        &self,
        engine: &IntrospectionEngine,
        json_output: bool,
    ) -> Result<(), NagualError> {
        let (status, health) = engine.health_check().await?;

        if json_output {
            let output = serde_json::json!({
                "status": format!("{}", status),
                "patterns": health.total_patterns,
                "avg_reward": health.average_reward,
                "stale_count": health.stale_count,
            });
            println!("{}", serde_json::to_string_pretty(&output).unwrap_or_default());
        } else {
            let icon = match status {
                HealthStatus::Healthy => "●",
                HealthStatus::Warning => "◐",
                HealthStatus::Degraded => "○",
                HealthStatus::Critical => "✗",
            };
            println!(
                "{} System Health: {} ({} patterns, avg reward: {:.2})",
                icon, status, health.total_patterns, health.average_reward
            );
        }

        // Exit with code reflecting health status
        let exit_code = match status {
            HealthStatus::Healthy => 0,
            HealthStatus::Warning => 1,
            HealthStatus::Degraded => 2,
            HealthStatus::Critical => 3,
        };

        if exit_code != 0 {
            std::process::exit(exit_code);
        }

        Ok(())
    }

    fn print_full_report(&self, model: &SelfModel) {
        println!("Strange Loop Introspection Report");
        println!("==================================");
        println!("Snapshot: {}", model.snapshot_at.format("%Y-%m-%d %H:%M:%S UTC"));
        println!();

        // Health status
        let status = model.health_status();
        let icon = match status {
            HealthStatus::Healthy => "●",
            HealthStatus::Warning => "◐",
            HealthStatus::Degraded => "○",
            HealthStatus::Critical => "✗",
        };
        println!("{} Overall Health: {}", icon, status);
        println!();

        // Pattern health
        println!("Pattern Health:");
        println!("  Total Patterns: {}", model.pattern_health.total_patterns);
        println!(
            "  Reward Distribution: {} high, {} medium, {} low",
            model.pattern_health.high_reward_count,
            model.pattern_health.medium_reward_count,
            model.pattern_health.low_reward_count
        );
        println!("  Average Reward: {:.3}", model.pattern_health.average_reward);
        println!("  Average Effectiveness: {:.3}", model.pattern_health.average_effectiveness);
        println!("  Average Age: {:.1} days", model.pattern_health.average_age_days);
        println!("  Stale Patterns: {}", model.pattern_health.stale_count);
        println!("  With Embeddings: {}", model.pattern_health.with_embeddings);
        println!("  Total Reuse Count: {}", model.pattern_health.total_reuse_count);
        println!();

        // Temporal trends
        println!("Temporal Trends:");
        println!(
            "  7-day Reward Trend: {} (avg: {:.3})",
            model.temporal_trends.reward_trend_7d,
            model.temporal_trends.avg_reward_7d
        );
        println!(
            "  30-day Reward Trend: {} (avg: {:.3})",
            model.temporal_trends.reward_trend_30d,
            model.temporal_trends.avg_reward_30d
        );
        println!(
            "  Pattern Growth Rate: {:.1}/week",
            model.temporal_trends.pattern_growth_rate
        );
        println!(
            "  Patterns Created: {} (7d), {} (30d)",
            model.temporal_trends.patterns_created_7d,
            model.temporal_trends.patterns_created_30d
        );
        println!();

        // Domain coverage
        if !model.domain_coverage.is_empty() {
            println!("Domain Coverage:");
            let mut domains: Vec<_> = model.domain_coverage.values().collect();
            domains.sort_by(|a, b| b.pattern_count.cmp(&a.pattern_count));

            for (i, metrics) in domains.iter().take(10).enumerate() {
                let coverage_bar = self.make_bar(metrics.coverage_score, 10);
                println!(
                    "  {}. {} [{}] ({} patterns, avg reward: {:.2})",
                    i + 1,
                    metrics.domain,
                    coverage_bar,
                    metrics.pattern_count,
                    metrics.avg_reward
                );
            }
            if domains.len() > 10 {
                println!("  ... and {} more domains", domains.len() - 10);
            }
            println!();
        }

        // Vulnerabilities
        self.print_vulnerabilities(model);

        // Recommendations
        self.print_recommendations(model, self.max_recommendations);
    }

    fn print_vulnerabilities(&self, model: &SelfModel) {
        if model.vulnerabilities.is_empty() {
            println!("Vulnerabilities: None detected");
            println!();
            return;
        }

        println!("Vulnerabilities ({}):", model.vulnerabilities.len());
        for vuln in &model.vulnerabilities {
            let severity_icon = match vuln.severity {
                Severity::Critical => "!!",
                Severity::High => "! ",
                Severity::Medium => "* ",
                Severity::Low => "- ",
            };
            println!(
                "  [{}] {} ({}): {}",
                severity_icon, vuln.category, vuln.severity, vuln.description
            );
            if !vuln.affected_domains.is_empty() {
                println!("       Affected: {}", vuln.affected_domains.join(", "));
            }
        }
        println!();
    }

    fn print_recommendations(&self, model: &SelfModel, max: usize) {
        if model.recommendations.is_empty() {
            println!("Recommendations: None");
            println!();
            return;
        }

        println!("Top Recommendations:");
        for (i, rec) in model.recommendations.iter().take(max).enumerate() {
            let priority_bar = self.make_bar(rec.priority as f64 / 10.0, 5);
            println!(
                "  {}. [{}] {} (benefit: {:.0}%)",
                i + 1,
                priority_bar,
                rec.action,
                rec.estimated_benefit * 100.0
            );
            if let Some(goal) = &rec.goap_goal {
                println!("       GOAP Goal: {}", goal);
            }
        }
        if model.recommendations.len() > max {
            println!(
                "  ... and {} more recommendations",
                model.recommendations.len() - max
            );
        }
        println!();
    }

    fn make_bar(&self, value: f64, width: usize) -> String {
        let filled = (value * width as f64).round() as usize;
        let empty = width.saturating_sub(filled);
        format!("{}{}", "█".repeat(filled), "░".repeat(empty))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_make_bar() {
        let cmd = IntrospectCommand {
            db_path: "test.db".into(),
            domain: None,
            vulnerabilities: false,
            recommendations: false,
            health_check: false,
            max_recommendations: 5,
            stale_days: 30,
        };

        assert_eq!(cmd.make_bar(0.5, 10), "█████░░░░░");
        assert_eq!(cmd.make_bar(1.0, 5), "█████");
        assert_eq!(cmd.make_bar(0.0, 5), "░░░░░");
    }
}
