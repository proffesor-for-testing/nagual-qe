//! Introspection engine for self-analysis

use std::collections::HashMap;
use std::sync::Arc;

use chrono::Utc;
use tracing::{debug, info, instrument};

use super::self_model::*;
use crate::db::SqliteDb;
use crate::error::NagualError;

/// Introspection engine for Strange Loop self-awareness
pub struct IntrospectionEngine {
    db: Arc<SqliteDb>,
    config: IntrospectionConfig,
}

impl IntrospectionEngine {
    /// Create a new introspection engine
    pub fn new(db: Arc<SqliteDb>, config: IntrospectionConfig) -> Self {
        Self { db, config }
    }

    /// Create with default configuration
    pub fn with_defaults(db: Arc<SqliteDb>) -> Self {
        Self::new(db, IntrospectionConfig::default())
    }

    /// Perform full introspection and return self-model
    #[instrument(skip(self))]
    pub async fn introspect(&self) -> Result<SelfModel, NagualError> {
        info!("Starting Strange Loop introspection");

        let pattern_health = self.analyze_pattern_health().await?;
        let domain_coverage = self.analyze_domain_coverage().await?;
        let temporal_trends = self.analyze_trends().await?;
        let vulnerabilities = self.detect_vulnerabilities(
            &pattern_health,
            &domain_coverage,
            &temporal_trends,
        );
        let recommendations = self.generate_recommendations(&vulnerabilities);

        let model = SelfModel {
            snapshot_at: Utc::now(),
            pattern_health,
            domain_coverage,
            temporal_trends,
            vulnerabilities,
            recommendations,
        };

        info!(
            "Introspection complete: {} patterns, {} vulnerabilities, {} recommendations",
            model.pattern_health.total_patterns,
            model.vulnerabilities.len(),
            model.recommendations.len()
        );

        Ok(model)
    }

    /// Quick health check (returns exit code: 0=healthy, 1=warning, 2=degraded, 3=critical)
    pub async fn health_check(&self) -> Result<(HealthStatus, PatternHealth), NagualError> {
        let pattern_health = self.analyze_pattern_health().await?;
        let domain_coverage = self.analyze_domain_coverage().await?;
        let temporal_trends = self.analyze_trends().await?;
        let vulnerabilities = self.detect_vulnerabilities(
            &pattern_health,
            &domain_coverage,
            &temporal_trends,
        );

        let status = if vulnerabilities.iter().any(|v| matches!(v.severity, Severity::Critical)) {
            HealthStatus::Critical
        } else if vulnerabilities.iter().filter(|v| matches!(v.severity, Severity::High)).count() > 1 {
            HealthStatus::Degraded
        } else if vulnerabilities.iter().any(|v| matches!(v.severity, Severity::High | Severity::Medium)) {
            HealthStatus::Warning
        } else {
            HealthStatus::Healthy
        };

        Ok((status, pattern_health))
    }

    /// Analyze overall pattern health
    #[instrument(skip(self))]
    async fn analyze_pattern_health(&self) -> Result<PatternHealth, NagualError> {
        debug!("Analyzing pattern health");

        let sql = r#"
            SELECT
                COUNT(*) as total,
                COALESCE(AVG(reward), 0.0) as avg_reward,
                COALESCE(AVG(effectiveness), 0.0) as avg_effectiveness,
                COALESCE(SUM(reuse_count), 0) as total_reuse,
                SUM(CASE WHEN reward >= ? THEN 1 ELSE 0 END) as high_reward,
                SUM(CASE WHEN reward < ? THEN 1 ELSE 0 END) as low_reward,
                SUM(CASE WHEN embedding IS NOT NULL AND embedding != '' THEN 1 ELSE 0 END) as with_embeddings,
                AVG(julianday('now') - julianday(timestamp)) as avg_age_days
            FROM reasoning_patterns
        "#;

        let high_threshold = self.config.high_reward_threshold;
        let low_threshold = self.config.low_reward_threshold;

        let result = self.db.query_one(
            sql,
            &[&high_threshold, &low_threshold],
            |row| {
                Ok(PatternHealth {
                    total_patterns: row.get::<_, i64>(0)? as usize,
                    average_reward: row.get(1)?,
                    average_effectiveness: row.get(2)?,
                    total_reuse_count: row.get::<_, i64>(3)? as usize,
                    high_reward_count: row.get::<_, i64>(4)? as usize,
                    low_reward_count: row.get::<_, i64>(5)? as usize,
                    with_embeddings: row.get::<_, i64>(6)? as usize,
                    average_age_days: row.get(7).unwrap_or(0.0),
                    medium_reward_count: 0, // calculated below
                    stale_count: 0,         // calculated below
                    orphan_count: 0,        // TODO: graph connectivity
                })
            },
        ).await?;

        let mut health = match result {
            Some(h) => h,
            None => PatternHealth::default(),
        };

        // Calculate medium reward count
        health.medium_reward_count = health.total_patterns
            .saturating_sub(health.high_reward_count)
            .saturating_sub(health.low_reward_count);

        // Count stale patterns
        let stale_sql = r#"
            SELECT COUNT(*) FROM reasoning_patterns
            WHERE julianday('now') - julianday(updated_at) > ?
            AND reuse_count < 5
        "#;
        let stale_threshold = self.config.stale_threshold_days as f64;
        let stale_count = self.db.query_one(
            stale_sql,
            &[&stale_threshold],
            |row| row.get::<_, i64>(0),
        ).await?.unwrap_or(0);
        health.stale_count = stale_count as usize;

        Ok(health)
    }

    /// Analyze domain coverage
    #[instrument(skip(self))]
    async fn analyze_domain_coverage(&self) -> Result<HashMap<String, DomainMetrics>, NagualError> {
        debug!("Analyzing domain coverage");

        let sql = r#"
            SELECT
                category,
                COUNT(*) as pattern_count,
                COALESCE(AVG(reward), 0.0) as avg_reward,
                COALESCE(AVG(effectiveness), 0.0) as avg_effectiveness,
                COALESCE(SUM(reuse_count), 0) as total_reuse,
                MAX(updated_at) as last_activity
            FROM reasoning_patterns
            GROUP BY category
            ORDER BY pattern_count DESC
        "#;

        let domains = self.db.query(
            sql,
            &[],
            |row| {
                let domain: String = row.get(0)?;
                let pattern_count: i64 = row.get(1)?;
                let avg_reward: f64 = row.get(2)?;
                let avg_effectiveness: f64 = row.get(3)?;
                let total_reuse: i64 = row.get(4)?;
                let last_activity: Option<String> = row.get(5)?;

                let last_activity_dt = last_activity.and_then(|s| {
                    chrono::DateTime::parse_from_rfc3339(&s)
                        .ok()
                        .map(|dt| dt.with_timezone(&Utc))
                });

                // Estimate coverage score based on pattern count and quality
                // More patterns with higher rewards = better coverage
                let coverage_score = (pattern_count as f64 / 100.0).min(1.0)
                    * (0.5 + avg_reward * 0.5);

                Ok(DomainMetrics {
                    domain: domain.clone(),
                    pattern_count: pattern_count as usize,
                    coverage_score,
                    avg_reward,
                    avg_effectiveness,
                    last_activity: last_activity_dt,
                    gap_areas: vec![], // TODO: detect gaps
                    total_reuse_count: total_reuse as usize,
                })
            },
        ).await?;

        let mut map = HashMap::new();
        for metrics in domains {
            map.insert(metrics.domain.clone(), metrics);
        }

        Ok(map)
    }

    /// Analyze temporal trends
    #[instrument(skip(self))]
    async fn analyze_trends(&self) -> Result<TemporalTrends, NagualError> {
        debug!("Analyzing temporal trends");

        // Patterns created in last 7 days
        let sql_7d = r#"
            SELECT
                COUNT(*) as count,
                COALESCE(AVG(reward), 0.0) as avg_reward
            FROM reasoning_patterns
            WHERE julianday('now') - julianday(timestamp) <= 7
        "#;
        let (count_7d, avg_7d) = self.db.query_one(
            sql_7d,
            &[],
            |row| Ok((row.get::<_, i64>(0)?, row.get::<_, f64>(1)?)),
        ).await?.unwrap_or((0, 0.0));

        // Patterns created in last 30 days
        let sql_30d = r#"
            SELECT
                COUNT(*) as count,
                COALESCE(AVG(reward), 0.0) as avg_reward
            FROM reasoning_patterns
            WHERE julianday('now') - julianday(timestamp) <= 30
        "#;
        let (count_30d, avg_30d) = self.db.query_one(
            sql_30d,
            &[],
            |row| Ok((row.get::<_, i64>(0)?, row.get::<_, f64>(1)?)),
        ).await?.unwrap_or((0, 0.0));

        // Overall average reward for comparison
        let sql_overall = "SELECT COALESCE(AVG(reward), 0.0) FROM reasoning_patterns";
        let overall_avg = self.db.query_one(
            sql_overall,
            &[],
            |row| row.get::<_, f64>(0),
        ).await?.unwrap_or(0.0);

        // Determine trends
        let reward_trend_7d = if avg_7d > overall_avg + 0.05 {
            TrendDirection::Improving
        } else if avg_7d < overall_avg - 0.05 {
            TrendDirection::Declining
        } else {
            TrendDirection::Stable
        };

        let reward_trend_30d = if avg_30d > overall_avg + 0.03 {
            TrendDirection::Improving
        } else if avg_30d < overall_avg - 0.03 {
            TrendDirection::Declining
        } else {
            TrendDirection::Stable
        };

        // Calculate growth rate (patterns per week)
        let pattern_growth_rate = count_30d as f64 / 4.0; // 4 weeks

        // Calculate decay rate (newly stale patterns per week)
        let sql_decay = r#"
            SELECT COUNT(*) FROM reasoning_patterns
            WHERE julianday('now') - julianday(updated_at) BETWEEN 30 AND 37
            AND reuse_count < 5
        "#;
        let decay_count = self.db.query_one(
            sql_decay,
            &[],
            |row| row.get::<_, i64>(0),
        ).await?.unwrap_or(0);
        let decay_rate = decay_count as f64;

        Ok(TemporalTrends {
            reward_trend_7d,
            reward_trend_30d,
            pattern_growth_rate,
            decay_rate,
            patterns_created_7d: count_7d as usize,
            patterns_created_30d: count_30d as usize,
            avg_reward_7d: avg_7d,
            avg_reward_30d: avg_30d,
            estimated_time_to_stale: None, // TODO: calculate from decay rate
        })
    }

    /// Detect vulnerabilities based on analyzed data
    fn detect_vulnerabilities(
        &self,
        health: &PatternHealth,
        coverage: &HashMap<String, DomainMetrics>,
        trends: &TemporalTrends,
    ) -> Vec<Vulnerability> {
        let mut vulnerabilities = Vec::new();

        // Knowledge decay detection
        if health.total_patterns > 0 {
            let stale_ratio = health.stale_count as f64 / health.total_patterns as f64;
            if stale_ratio > 0.3 {
                vulnerabilities.push(Vulnerability {
                    id: uuid::Uuid::new_v4().to_string(),
                    category: VulnerabilityCategory::KnowledgeDecay,
                    severity: if stale_ratio > 0.5 { Severity::Critical } else { Severity::High },
                    description: format!(
                        "{} patterns ({:.0}%) are stale and need refresh",
                        health.stale_count,
                        stale_ratio * 100.0
                    ),
                    affected_domains: vec![],
                    estimated_impact: stale_ratio,
                });
            } else if stale_ratio > 0.2 {
                vulnerabilities.push(Vulnerability {
                    id: uuid::Uuid::new_v4().to_string(),
                    category: VulnerabilityCategory::KnowledgeDecay,
                    severity: Severity::Medium,
                    description: format!(
                        "{} patterns ({:.0}%) are stale",
                        health.stale_count,
                        stale_ratio * 100.0
                    ),
                    affected_domains: vec![],
                    estimated_impact: stale_ratio,
                });
            }
        }

        // Low embedding coverage
        if health.total_patterns > 0 {
            let embedding_ratio = health.with_embeddings as f64 / health.total_patterns as f64;
            if embedding_ratio < self.config.min_embedding_coverage {
                let missing = health.total_patterns - health.with_embeddings;
                vulnerabilities.push(Vulnerability {
                    id: uuid::Uuid::new_v4().to_string(),
                    category: VulnerabilityCategory::LowEmbeddingCoverage,
                    severity: if embedding_ratio < 0.5 { Severity::High } else { Severity::Medium },
                    description: format!(
                        "{} patterns ({:.0}%) missing embeddings for semantic search",
                        missing,
                        (1.0 - embedding_ratio) * 100.0
                    ),
                    affected_domains: vec![],
                    estimated_impact: 1.0 - embedding_ratio,
                });
            }
        }

        // Coverage gap detection
        for (domain, metrics) in coverage {
            if metrics.coverage_score < self.config.min_coverage_score && metrics.pattern_count > 5 {
                vulnerabilities.push(Vulnerability {
                    id: uuid::Uuid::new_v4().to_string(),
                    category: VulnerabilityCategory::CoverageGap,
                    severity: if metrics.coverage_score < 0.3 { Severity::High } else { Severity::Medium },
                    description: format!(
                        "Domain '{}' has low coverage score ({:.1}%)",
                        domain,
                        metrics.coverage_score * 100.0
                    ),
                    affected_domains: vec![domain.clone()],
                    estimated_impact: 1.0 - metrics.coverage_score,
                });
            }
        }

        // Quality degradation detection
        if matches!(trends.reward_trend_7d, TrendDirection::Declining) {
            vulnerabilities.push(Vulnerability {
                id: uuid::Uuid::new_v4().to_string(),
                category: VulnerabilityCategory::QualityDegradation,
                severity: Severity::Medium,
                description: format!(
                    "Pattern quality declining: 7d avg reward ({:.2}) below overall",
                    trends.avg_reward_7d
                ),
                affected_domains: vec![],
                estimated_impact: 0.2,
            });
        }

        if matches!(trends.reward_trend_30d, TrendDirection::Declining) {
            vulnerabilities.push(Vulnerability {
                id: uuid::Uuid::new_v4().to_string(),
                category: VulnerabilityCategory::QualityDegradation,
                severity: Severity::High,
                description: format!(
                    "Sustained quality decline: 30d avg reward ({:.2}) below overall",
                    trends.avg_reward_30d
                ),
                affected_domains: vec![],
                estimated_impact: 0.4,
            });
        }

        // Overspecialization detection
        if coverage.len() > 3 {
            let total_patterns: usize = coverage.values().map(|m| m.pattern_count).sum();
            let max_domain = coverage.values().max_by_key(|m| m.pattern_count);
            if let Some(max) = max_domain {
                let concentration = max.pattern_count as f64 / total_patterns as f64;
                if concentration > 0.7 {
                    vulnerabilities.push(Vulnerability {
                        id: uuid::Uuid::new_v4().to_string(),
                        category: VulnerabilityCategory::Overspecialization,
                        severity: Severity::Low,
                        description: format!(
                            "Knowledge concentrated in '{}' ({:.0}% of patterns)",
                            max.domain,
                            concentration * 100.0
                        ),
                        affected_domains: vec![max.domain.clone()],
                        estimated_impact: concentration - 0.5,
                    });
                }
            }
        }

        // Low quality patterns
        if health.total_patterns > 0 {
            let low_quality_ratio = health.low_reward_count as f64 / health.total_patterns as f64;
            if low_quality_ratio > 0.3 {
                vulnerabilities.push(Vulnerability {
                    id: uuid::Uuid::new_v4().to_string(),
                    category: VulnerabilityCategory::QualityDegradation,
                    severity: if low_quality_ratio > 0.5 { Severity::High } else { Severity::Medium },
                    description: format!(
                        "{} patterns ({:.0}%) have low reward scores",
                        health.low_reward_count,
                        low_quality_ratio * 100.0
                    ),
                    affected_domains: vec![],
                    estimated_impact: low_quality_ratio,
                });
            }
        }

        // Sort by severity
        vulnerabilities.sort_by(|a, b| b.severity.cmp(&a.severity));

        vulnerabilities
    }

    /// Generate recommendations based on vulnerabilities
    fn generate_recommendations(&self, vulnerabilities: &[Vulnerability]) -> Vec<Recommendation> {
        let mut recommendations = Vec::new();

        for vuln in vulnerabilities {
            match &vuln.category {
                VulnerabilityCategory::KnowledgeDecay => {
                    recommendations.push(Recommendation {
                        id: uuid::Uuid::new_v4().to_string(),
                        action: RecommendedAction::RefreshStalePatterns {
                            domain: "all".into(),
                            count: 10,
                        },
                        priority: match vuln.severity {
                            Severity::Critical => 10,
                            Severity::High => 8,
                            Severity::Medium => 5,
                            Severity::Low => 3,
                        },
                        reason: vuln.description.clone(),
                        estimated_benefit: vuln.estimated_impact,
                        goap_goal: Some("refresh stale knowledge".into()),
                    });
                }

                VulnerabilityCategory::CoverageGap => {
                    for domain in &vuln.affected_domains {
                        recommendations.push(Recommendation {
                            id: uuid::Uuid::new_v4().to_string(),
                            action: RecommendedAction::FillCoverageGap {
                                domain: domain.clone(),
                                topic: "general".into(),
                            },
                            priority: 7,
                            reason: vuln.description.clone(),
                            estimated_benefit: vuln.estimated_impact,
                            goap_goal: Some(format!("improve {} knowledge coverage", domain)),
                        });
                    }
                }

                VulnerabilityCategory::QualityDegradation => {
                    recommendations.push(Recommendation {
                        id: uuid::Uuid::new_v4().to_string(),
                        action: RecommendedAction::ArchiveLowQuality {
                            count: 20,
                        },
                        priority: match vuln.severity {
                            Severity::Critical => 9,
                            Severity::High => 7,
                            Severity::Medium => 4,
                            Severity::Low => 2,
                        },
                        reason: vuln.description.clone(),
                        estimated_benefit: vuln.estimated_impact,
                        goap_goal: Some("improve pattern quality".into()),
                    });
                }

                VulnerabilityCategory::LowEmbeddingCoverage => {
                    recommendations.push(Recommendation {
                        id: uuid::Uuid::new_v4().to_string(),
                        action: RecommendedAction::GenerateEmbeddings {
                            count: 100,
                        },
                        priority: match vuln.severity {
                            Severity::Critical => 9,
                            Severity::High => 7,
                            Severity::Medium => 5,
                            Severity::Low => 3,
                        },
                        reason: vuln.description.clone(),
                        estimated_benefit: vuln.estimated_impact,
                        goap_goal: Some("generate missing embeddings".into()),
                    });
                }

                VulnerabilityCategory::Overspecialization => {
                    for domain in &vuln.affected_domains {
                        recommendations.push(Recommendation {
                            id: uuid::Uuid::new_v4().to_string(),
                            action: RecommendedAction::ResearchTopic {
                                topic: format!("diversify beyond {}", domain),
                            },
                            priority: 3,
                            reason: vuln.description.clone(),
                            estimated_benefit: vuln.estimated_impact,
                            goap_goal: Some("diversify knowledge base".into()),
                        });
                    }
                }

                VulnerabilityCategory::Fragmentation => {
                    recommendations.push(Recommendation {
                        id: uuid::Uuid::new_v4().to_string(),
                        action: RecommendedAction::ConsolidatePatterns {
                            domain: "all".into(),
                            threshold: 0.85,
                        },
                        priority: 5,
                        reason: vuln.description.clone(),
                        estimated_benefit: vuln.estimated_impact,
                        goap_goal: Some("consolidate fragmented knowledge".into()),
                    });
                }
            }
        }

        // Sort by priority
        recommendations.sort_by(|a, b| b.priority.cmp(&a.priority));

        // Deduplicate similar recommendations
        recommendations.dedup_by(|a, b| {
            std::mem::discriminant(&a.action) == std::mem::discriminant(&b.action)
        });

        recommendations
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vulnerability_detection_stale() {
        let engine = create_test_engine();
        let health = PatternHealth {
            total_patterns: 100,
            stale_count: 35,
            ..Default::default()
        };
        let coverage = HashMap::new();
        let trends = TemporalTrends::default();

        let vulns = engine.detect_vulnerabilities(&health, &coverage, &trends);
        assert!(vulns.iter().any(|v| matches!(v.category, VulnerabilityCategory::KnowledgeDecay)));
    }

    #[test]
    fn test_recommendation_generation() {
        let engine = create_test_engine();
        let vulns = vec![Vulnerability {
            id: "1".into(),
            category: VulnerabilityCategory::KnowledgeDecay,
            severity: Severity::High,
            description: "Test".into(),
            affected_domains: vec![],
            estimated_impact: 0.5,
        }];

        let recs = engine.generate_recommendations(&vulns);
        assert!(!recs.is_empty());
        assert!(recs[0].goap_goal.is_some());
    }

    fn create_test_engine() -> IntrospectionEngine {
        // Create a mock engine for testing
        // In real tests, we'd use a test database
        let db = Arc::new(SqliteDb::open_in_memory().unwrap());
        IntrospectionEngine::new(db, IntrospectionConfig::default())
    }
}
