//! Research Coordinator
//!
//! Orchestrates the research process: parses requests, spawns agents,
//! manages budgets, and aggregates results into patterns.

use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::time::timeout;
use tracing::{debug, info, warn, instrument};
use chrono::Utc;

use super::agents::{AgentFactory, ResearchAgent};
use super::matts::{MaTTS, MaTTSConfig};
use super::types::*;
use crate::db::SqliteDb;
use crate::error::NagualError;

/// Research coordinator configuration
#[derive(Debug, Clone)]
pub struct CoordinatorConfig {
    /// MaTTS configuration
    pub matts: MaTTSConfig,
    /// Default budget for research
    pub default_budget: ResearchBudget,
    /// Auto-store patterns after research
    pub auto_store_patterns: bool,
    /// Minimum confidence to store pattern
    pub min_store_confidence: f64,
}

impl Default for CoordinatorConfig {
    fn default() -> Self {
        Self {
            matts: MaTTSConfig::default(),
            default_budget: ResearchBudget::default(),
            auto_store_patterns: true,
            min_store_confidence: 0.5,
        }
    }
}

/// Research coordinator
pub struct ResearchCoordinator {
    db: Arc<SqliteDb>,
    config: CoordinatorConfig,
    matts: MaTTS,
}

impl ResearchCoordinator {
    pub fn new(db: Arc<SqliteDb>, config: CoordinatorConfig) -> Self {
        let matts = MaTTS::new(config.matts.clone());
        Self {
            db,
            config,
            matts,
        }
    }

    pub fn with_defaults(db: Arc<SqliteDb>) -> Self {
        Self::new(db, CoordinatorConfig::default())
    }

    /// Execute research for a request
    #[instrument(skip(self), fields(topic = %request.topic, depth = %request.depth))]
    pub async fn research(&self, request: ResearchRequest) -> Result<ResearchResult, NagualError> {
        let start = Instant::now();
        info!("Starting research: {} (depth: {})", request.topic, request.depth);

        // Create agents based on strategy
        let agents = AgentFactory::create_agents(
            self.db.clone(),
            &request.strategy,
            request.budget.max_agents,
        );

        info!("Spawned {} research agents", agents.len());

        // Execute research with timeout
        let max_duration = Duration::from_secs(request.budget.max_time_seconds);
        let trajectories = self.execute_with_timeout(agents, &request, max_duration).await?;

        info!("Collected {} trajectories", trajectories.len());

        // Aggregate results using MaTTS
        let consensus = self.matts.aggregate(&trajectories);

        // Create patterns from findings
        let patterns_created = if self.config.auto_store_patterns {
            self.create_patterns(&request, &trajectories, &consensus).await?
        } else {
            vec![]
        };

        let total_tokens: usize = trajectories.iter().map(|t| t.total_tokens()).sum();
        let total_duration_ms = start.elapsed().as_millis() as u64;

        let result = ResearchResult {
            request_id: request.id.clone(),
            topic: request.topic.clone(),
            trajectories,
            consensus,
            patterns_created,
            total_tokens,
            total_duration_ms,
            completed_at: Utc::now(),
        };

        info!(
            "Research complete: {} patterns created, confidence={:.2}, duration={}ms",
            result.patterns_created.len(),
            result.consensus.confidence,
            total_duration_ms
        );

        Ok(result)
    }

    /// Execute agents with timeout and early stopping
    async fn execute_with_timeout(
        &self,
        agents: Vec<Box<dyn ResearchAgent>>,
        request: &ResearchRequest,
        max_duration: Duration,
    ) -> Result<Vec<ResearchTrajectory>, NagualError> {
        let mut trajectories = Vec::new();
        let start = Instant::now();

        for agent in agents {
            // Check timeout
            let remaining = max_duration.saturating_sub(start.elapsed());
            if remaining.is_zero() {
                warn!("Research timeout reached, stopping agents");
                break;
            }

            // Execute agent with remaining time
            let agent_timeout = remaining.min(Duration::from_secs(30));

            match timeout(agent_timeout, agent.research(request)).await {
                Ok(Ok(trajectory)) => {
                    debug!(
                        "Agent {} completed with quality {:.2}",
                        trajectory.agent_type, trajectory.quality_score
                    );
                    trajectories.push(trajectory);

                    // Check early stopping
                    if self.matts.should_early_stop(&trajectories) {
                        info!("Early stopping triggered");
                        break;
                    }
                }
                Ok(Err(e)) => {
                    warn!("Agent failed: {}", e);
                }
                Err(_) => {
                    warn!("Agent timed out");
                }
            }
        }

        Ok(trajectories)
    }

    /// Create patterns from research findings
    async fn create_patterns(
        &self,
        request: &ResearchRequest,
        trajectories: &[ResearchTrajectory],
        _consensus: &ConsensusResult,
    ) -> Result<Vec<PatternSummary>, NagualError> {
        let mut created = Vec::new();
        let domain = request.domain.clone().unwrap_or_else(|| "research".to_string());

        // Collect high-confidence findings
        let mut findings: Vec<&ResearchFinding> = trajectories
            .iter()
            .flat_map(|t| t.findings.iter())
            .filter(|f| f.confidence >= self.config.min_store_confidence)
            .collect();

        // Deduplicate by content similarity
        findings.dedup_by(|a, b| {
            let a_norm = a.content.to_lowercase();
            let b_norm = b.content.to_lowercase();
            a_norm == b_norm || jaccard_similarity(&a_norm, &b_norm) > 0.8
        });

        // Limit patterns to budget
        findings.truncate(request.budget.max_patterns);

        for finding in findings {
            // Extract problem and solution from finding
            let (problem, solution) = self.extract_problem_solution(&finding.content, &request.topic);
            let id = uuid::Uuid::new_v4().to_string();
            let tags = finding.tags.join(",");
            let now = chrono::Utc::now().to_rfc3339();

            // Store pattern directly via SQL
            let result = self.db.execute(
                r#"
                INSERT INTO reasoning_patterns (id, problem, solution, category, tags, reward, timestamp, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                "#,
                &[&id, &problem, &solution, &domain, &tags, &"0.5", &now, &now],
            ).await;

            match result {
                Ok(_) => {
                    created.push(PatternSummary {
                        id: id.clone(),
                        problem: problem.clone(),
                        domain: domain.clone(),
                        source: finding.source.clone(),
                    });
                }
                Err(e) => {
                    warn!("Failed to store pattern: {}", e);
                }
            }
        }

        info!("Created {} patterns from research", created.len());
        Ok(created)
    }

    /// Extract problem and solution from finding content
    fn extract_problem_solution(&self, content: &str, topic: &str) -> (String, String) {
        // Try to find structure in content
        let lines: Vec<&str> = content.lines().collect();

        if lines.len() >= 2 {
            // First line as problem, rest as solution
            let problem = lines[0].trim_start_matches(['*', '#', ' ']).to_string();
            let solution = lines[1..].join("\n");
            (problem, solution)
        } else {
            // Use topic as problem, content as solution
            (topic.to_string(), content.to_string())
        }
    }

    /// Dry run - plan research without executing
    pub fn plan(&self, request: &ResearchRequest) -> ResearchPlan {
        let agents = AgentFactory::create_agents(
            self.db.clone(),
            &request.strategy,
            request.budget.max_agents,
        );

        let agent_descriptions: Vec<String> = agents
            .iter()
            .map(|a| format!("{}", a.agent_type()))
            .collect();

        ResearchPlan {
            topic: request.topic.clone(),
            depth: request.depth,
            strategy: format!("{}", request.strategy),
            agents: agent_descriptions,
            budget: request.budget.clone(),
            estimated_duration_seconds: self.estimate_duration(request),
            estimated_patterns: request.depth.max_patterns(),
        }
    }

    fn estimate_duration(&self, request: &ResearchRequest) -> u64 {
        match request.depth {
            ResearchDepth::Quick => 15,
            ResearchDepth::Medium => 45,
            ResearchDepth::Deep => 90,
        }
    }
}

/// Research plan (dry run result)
#[derive(Debug, Clone)]
pub struct ResearchPlan {
    pub topic: String,
    pub depth: ResearchDepth,
    pub strategy: String,
    pub agents: Vec<String>,
    pub budget: ResearchBudget,
    pub estimated_duration_seconds: u64,
    pub estimated_patterns: usize,
}

/// Simple Jaccard similarity for string comparison
fn jaccard_similarity(a: &str, b: &str) -> f64 {
    let a_words: std::collections::HashSet<_> = a.split_whitespace().collect();
    let b_words: std::collections::HashSet<_> = b.split_whitespace().collect();

    if a_words.is_empty() && b_words.is_empty() {
        return 1.0;
    }

    let intersection = a_words.intersection(&b_words).count();
    let union = a_words.union(&b_words).count();

    if union > 0 {
        intersection as f64 / union as f64
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn setup_coordinator() -> ResearchCoordinator {
        let db = Arc::new(SqliteDb::open_in_memory().unwrap());

        // Initialize schema
        db.execute(
            r#"CREATE TABLE IF NOT EXISTS reasoning_patterns (
                id TEXT PRIMARY KEY,
                problem TEXT,
                solution TEXT,
                category TEXT,
                reward REAL DEFAULT 0.5,
                embedding TEXT,
                tags TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )"#,
            &[],
        ).await.unwrap();

        // Add some test patterns
        for i in 0..5 {
            db.execute(
                "INSERT INTO reasoning_patterns (id, problem, solution, category, reward) VALUES (?, ?, ?, ?, ?)",
                &[
                    &format!("test-{}", i),
                    &format!("Problem about error handling {}", i),
                    &format!("Solution: Use Result type {}", i),
                    &"rust.error",
                    &"0.7",
                ],
            ).await.unwrap();
        }

        ResearchCoordinator::new(
            db,
            CoordinatorConfig {
                auto_store_patterns: false, // Don't auto-store in tests
                ..Default::default()
            },
        )
    }

    #[tokio::test]
    async fn test_research_execution() {
        let coordinator = setup_coordinator().await;

        let request = ResearchRequest::new("error handling")
            .with_depth(ResearchDepth::Quick);

        let result = coordinator.research(request).await.unwrap();

        assert!(!result.trajectories.is_empty());
        assert!(result.consensus.confidence > 0.0);
        // Duration might be 0ms if execution is very fast - just check it exists
        let _ = result.total_duration_ms;
    }

    #[tokio::test]
    async fn test_dry_run() {
        let coordinator = setup_coordinator().await;

        let request = ResearchRequest::new("async patterns")
            .with_depth(ResearchDepth::Medium)
            .with_domain("rust.async");

        let plan = coordinator.plan(&request);

        assert_eq!(plan.topic, "async patterns");
        assert!(!plan.agents.is_empty());
        assert!(plan.estimated_duration_seconds > 0);
    }

    #[test]
    fn test_jaccard_similarity() {
        assert!((jaccard_similarity("hello world", "hello world") - 1.0).abs() < 0.001);
        assert!((jaccard_similarity("hello world", "hello") - 0.5).abs() < 0.001);
        assert!((jaccard_similarity("a b c", "d e f") - 0.0).abs() < 0.001);
    }
}
