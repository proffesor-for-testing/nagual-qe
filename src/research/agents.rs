//! Research agents for knowledge acquisition
//!
//! Implements different agent types: WebSearch, DocFetch, CodeAnalysis,
//! KnowledgeBase, and Synthesis agents.

use async_trait::async_trait;
use std::sync::Arc;
use std::time::Instant;
use tracing::info;

use super::types::*;
use crate::db::SqliteDb;
use crate::error::NagualError;

/// Trait for research agents
#[async_trait(?Send)]
pub trait ResearchAgent {
    /// Execute research for a request
    async fn research(&self, request: &ResearchRequest) -> Result<ResearchTrajectory, NagualError>;

    /// Get the agent type
    fn agent_type(&self) -> AgentType;

    /// Check if this agent can handle the request
    fn can_handle(&self, request: &ResearchRequest) -> bool;
}

/// Knowledge base search agent - searches existing patterns
pub struct KnowledgeBaseAgent {
    db: Arc<SqliteDb>,
}

impl KnowledgeBaseAgent {
    pub fn new(db: Arc<SqliteDb>) -> Self {
        Self { db }
    }

    async fn search_patterns(&self, query: &str, limit: usize) -> Result<Vec<PatternMatch>, NagualError> {
        let sql = r#"
            SELECT id, problem, solution, category, reward
            FROM reasoning_patterns
            WHERE problem LIKE ? OR solution LIKE ? OR category LIKE ?
            ORDER BY reward DESC
            LIMIT ?
        "#;

        let pattern = format!("%{}%", query);
        let limit_str = limit.to_string();

        let results: Vec<PatternMatch> = self.db.query(
            sql,
            &[&pattern, &pattern, &pattern, &limit_str],
            |row| {
                Ok(PatternMatch {
                    id: row.get(0)?,
                    problem: row.get(1)?,
                    solution: row.get(2)?,
                    category: row.get(3)?,
                    reward: row.get(4)?,
                })
            },
        ).await?;

        Ok(results)
    }
}

#[derive(Debug)]
struct PatternMatch {
    id: String,
    problem: String,
    solution: String,
    category: String,
    reward: f64,
}

#[async_trait(?Send)]
impl ResearchAgent for KnowledgeBaseAgent {
    async fn research(&self, request: &ResearchRequest) -> Result<ResearchTrajectory, NagualError> {
        let start = Instant::now();
        let mut trajectory = ResearchTrajectory::new(AgentType::KnowledgeBase);

        info!("KnowledgeBaseAgent researching: {}", request.topic);

        // Search for existing patterns
        let search_start = Instant::now();
        let patterns = self.search_patterns(&request.topic, request.depth.source_count() * 2).await?;
        let search_duration = search_start.elapsed();

        trajectory.add_step(ResearchStep {
            action: ResearchAction::QueryKnowledge {
                query: request.topic.clone(),
            },
            result: Some(format!("Found {} relevant patterns", patterns.len())),
            tokens_used: 0, // No API tokens used for local search
            duration_ms: search_duration.as_millis() as u64,
            success: true,
        });

        // Convert patterns to findings
        for pattern in &patterns {
            let content = format!(
                "**{}**\n\nProblem: {}\n\nSolution: {}",
                pattern.category, pattern.problem, pattern.solution
            );

            trajectory.add_finding(
                ResearchFinding::new(content, format!("pattern:{}", pattern.id))
                    .with_confidence(pattern.reward.min(1.0))
                    .with_tags(vec![pattern.category.clone()]),
            );
        }

        // Calculate quality based on matches and relevance
        let quality = if patterns.is_empty() {
            0.2 // Low quality if no matches
        } else {
            let avg_reward: f64 = patterns.iter().map(|p| p.reward).sum::<f64>() / patterns.len() as f64;
            (0.3 + avg_reward * 0.7).min(1.0)
        };

        trajectory.complete(quality);

        info!(
            "KnowledgeBaseAgent completed: {} findings, quality={:.2}, duration={:?}",
            trajectory.findings.len(),
            quality,
            start.elapsed()
        );

        Ok(trajectory)
    }

    fn agent_type(&self) -> AgentType {
        AgentType::KnowledgeBase
    }

    fn can_handle(&self, _request: &ResearchRequest) -> bool {
        true // Can always search the knowledge base
    }
}

/// Web search simulation agent
/// Note: In production, this would integrate with actual search APIs
pub struct WebSearchAgent {
    db: Arc<SqliteDb>,
}

impl WebSearchAgent {
    pub fn new(db: Arc<SqliteDb>) -> Self {
        Self { db }
    }

    /// Simulate web search by extracting keywords and searching patterns
    async fn simulate_web_search(&self, query: &str) -> Result<Vec<WebResult>, NagualError> {
        // Extract keywords for search
        let keywords: Vec<&str> = query
            .split_whitespace()
            .filter(|w| w.len() > 3)
            .take(5)
            .collect();

        let mut results = Vec::new();

        for keyword in keywords {
            let sql = r#"
                SELECT DISTINCT category, problem, solution
                FROM reasoning_patterns
                WHERE problem LIKE ? OR solution LIKE ?
                ORDER BY reward DESC
                LIMIT 3
            "#;

            let pattern = format!("%{}%", keyword);
            let matches: Vec<(String, String, String)> = self.db.query(
                sql,
                &[&pattern, &pattern],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            ).await?;

            for (category, problem, solution) in matches {
                results.push(WebResult {
                    title: format!("{} - {}", category, &problem[..problem.len().min(50)]),
                    snippet: solution[..solution.len().min(200)].to_string(),
                    url: format!("kb://{}", category),
                });
            }
        }

        // Deduplicate by title
        results.dedup_by(|a, b| a.title == b.title);
        results.truncate(10);

        Ok(results)
    }
}

#[derive(Debug)]
struct WebResult {
    title: String,
    snippet: String,
    url: String,
}

#[async_trait(?Send)]
impl ResearchAgent for WebSearchAgent {
    async fn research(&self, request: &ResearchRequest) -> Result<ResearchTrajectory, NagualError> {
        let start = Instant::now();
        let mut trajectory = ResearchTrajectory::new(AgentType::WebSearch);

        info!("WebSearchAgent researching: {}", request.topic);

        // Perform simulated web search
        let search_start = Instant::now();
        let results = self.simulate_web_search(&request.topic).await?;
        let search_duration = search_start.elapsed();

        trajectory.add_step(ResearchStep {
            action: ResearchAction::Search {
                query: request.topic.clone(),
            },
            result: Some(format!("Found {} web results", results.len())),
            tokens_used: 50, // Simulated token cost
            duration_ms: search_duration.as_millis() as u64,
            success: !results.is_empty(),
        });

        // Process results
        for result in &results {
            let content = format!("**{}**\n\n{}", result.title, result.snippet);

            trajectory.add_finding(
                ResearchFinding::new(content, &result.url)
                    .with_confidence(0.6),
            );
        }

        let quality = if results.is_empty() {
            0.3
        } else {
            (0.5 + results.len() as f64 * 0.05).min(0.9)
        };

        trajectory.complete(quality);

        info!(
            "WebSearchAgent completed: {} findings, quality={:.2}, duration={:?}",
            trajectory.findings.len(),
            quality,
            start.elapsed()
        );

        Ok(trajectory)
    }

    fn agent_type(&self) -> AgentType {
        AgentType::WebSearch
    }

    fn can_handle(&self, request: &ResearchRequest) -> bool {
        matches!(
            request.strategy,
            ResearchStrategy::WebSearch | ResearchStrategy::Combined | ResearchStrategy::Auto
        )
    }
}

/// Code analysis agent - analyzes local code for patterns
pub struct CodeAnalysisAgent {
    db: Arc<SqliteDb>,
}

impl CodeAnalysisAgent {
    pub fn new(db: Arc<SqliteDb>) -> Self {
        Self { db }
    }

    async fn analyze_patterns_in_domain(&self, domain: &str) -> Result<Vec<CodePattern>, NagualError> {
        let sql = r#"
            SELECT id, problem, solution, category, tags
            FROM reasoning_patterns
            WHERE category LIKE ?
            ORDER BY reward DESC
            LIMIT 20
        "#;

        let pattern = format!("{}%", domain);
        let results: Vec<CodePattern> = self.db.query(
            sql,
            &[&pattern],
            |row| {
                Ok(CodePattern {
                    id: row.get(0)?,
                    problem: row.get(1)?,
                    solution: row.get(2)?,
                    category: row.get(3)?,
                    tags: row.get::<_, Option<String>>(4)?.unwrap_or_default(),
                })
            },
        ).await?;

        Ok(results)
    }
}

#[derive(Debug)]
struct CodePattern {
    id: String,
    problem: String,
    solution: String,
    category: String,
    tags: String,
}

#[async_trait(?Send)]
impl ResearchAgent for CodeAnalysisAgent {
    async fn research(&self, request: &ResearchRequest) -> Result<ResearchTrajectory, NagualError> {
        let start = Instant::now();
        let mut trajectory = ResearchTrajectory::new(AgentType::CodeAnalysis);

        // Determine domain to analyze
        let domain = request.domain.as_deref().unwrap_or("rust");

        info!("CodeAnalysisAgent analyzing domain: {}", domain);

        let analysis_start = Instant::now();
        let patterns = self.analyze_patterns_in_domain(domain).await?;
        let analysis_duration = analysis_start.elapsed();

        trajectory.add_step(ResearchStep {
            action: ResearchAction::ExtractInfo {
                content_summary: format!("Analyzed {} patterns in {}", patterns.len(), domain),
            },
            result: Some(format!("Extracted {} code patterns", patterns.len())),
            tokens_used: 0,
            duration_ms: analysis_duration.as_millis() as u64,
            success: !patterns.is_empty(),
        });

        // Convert to findings
        for pattern in &patterns {
            let content = format!(
                "**Pattern: {}**\n\n{}\n\n**Solution:**\n{}",
                pattern.category, pattern.problem, pattern.solution
            );

            let tags: Vec<String> = pattern.tags
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();

            trajectory.add_finding(
                ResearchFinding::new(content, format!("code:{}", pattern.id))
                    .with_confidence(0.75)
                    .with_tags(tags),
            );
        }

        let quality = if patterns.is_empty() {
            0.2
        } else {
            (0.4 + patterns.len() as f64 * 0.03).min(0.85)
        };

        trajectory.complete(quality);

        info!(
            "CodeAnalysisAgent completed: {} findings, quality={:.2}, duration={:?}",
            trajectory.findings.len(),
            quality,
            start.elapsed()
        );

        Ok(trajectory)
    }

    fn agent_type(&self) -> AgentType {
        AgentType::CodeAnalysis
    }

    fn can_handle(&self, request: &ResearchRequest) -> bool {
        matches!(
            request.strategy,
            ResearchStrategy::CodeAnalysis { .. } | ResearchStrategy::Combined | ResearchStrategy::Auto
        )
    }
}

/// Synthesis agent - combines findings into coherent patterns
pub struct SynthesisAgent {
    db: Arc<SqliteDb>,
}

impl SynthesisAgent {
    pub fn new(db: Arc<SqliteDb>) -> Self {
        Self { db }
    }

    fn synthesize_findings(&self, findings: &[ResearchFinding], topic: &str) -> Vec<ResearchFinding> {
        if findings.is_empty() {
            return vec![];
        }

        // Group findings by tags/themes
        let mut synthesized = Vec::new();

        // Create a summary finding
        let summary_content = findings
            .iter()
            .take(5)
            .map(|f| format!("- {}", f.content.lines().next().unwrap_or(&f.content)))
            .collect::<Vec<_>>()
            .join("\n");

        let summary = format!(
            "**Summary: {}**\n\nKey findings:\n{}",
            topic, summary_content
        );

        synthesized.push(
            ResearchFinding::new(summary, "synthesis")
                .with_confidence(
                    findings.iter().map(|f| f.confidence).sum::<f64>() / findings.len() as f64
                )
                .with_tags(vec!["synthesized".to_string()]),
        );

        synthesized
    }
}

#[async_trait(?Send)]
impl ResearchAgent for SynthesisAgent {
    async fn research(&self, request: &ResearchRequest) -> Result<ResearchTrajectory, NagualError> {
        let start = Instant::now();
        let mut trajectory = ResearchTrajectory::new(AgentType::Synthesis);

        info!("SynthesisAgent processing: {}", request.topic);

        // This agent is typically used after other agents have gathered findings
        // For standalone use, it searches and synthesizes
        let kb_agent = KnowledgeBaseAgent::new(self.db.clone());
        let kb_result = kb_agent.research(request).await?;

        trajectory.add_step(ResearchStep {
            action: ResearchAction::Synthesize {
                source_count: kb_result.findings.len(),
            },
            result: Some(format!("Synthesizing {} findings", kb_result.findings.len())),
            tokens_used: 100, // Simulated synthesis cost
            duration_ms: start.elapsed().as_millis() as u64,
            success: true,
        });

        let synthesized = self.synthesize_findings(&kb_result.findings, &request.topic);
        for finding in synthesized {
            trajectory.add_finding(finding);
        }

        let quality = if trajectory.findings.is_empty() {
            0.3
        } else {
            0.7
        };

        trajectory.complete(quality);

        info!(
            "SynthesisAgent completed: {} synthesized findings, quality={:.2}",
            trajectory.findings.len(),
            quality
        );

        Ok(trajectory)
    }

    fn agent_type(&self) -> AgentType {
        AgentType::Synthesis
    }

    fn can_handle(&self, _request: &ResearchRequest) -> bool {
        true
    }
}

/// Factory for creating research agents
pub struct AgentFactory;

impl AgentFactory {
    pub fn create_agents(
        db: Arc<SqliteDb>,
        strategy: &ResearchStrategy,
        max_agents: usize,
    ) -> Vec<Box<dyn ResearchAgent>> {
        let mut agents: Vec<Box<dyn ResearchAgent>> = Vec::new();

        match strategy {
            ResearchStrategy::WebSearch => {
                agents.push(Box::new(WebSearchAgent::new(db.clone())));
            }
            ResearchStrategy::CodeAnalysis { .. } => {
                agents.push(Box::new(CodeAnalysisAgent::new(db.clone())));
            }
            ResearchStrategy::DocFetch { .. } => {
                // DocFetch uses web search as fallback
                agents.push(Box::new(WebSearchAgent::new(db.clone())));
            }
            ResearchStrategy::Combined | ResearchStrategy::Auto => {
                agents.push(Box::new(KnowledgeBaseAgent::new(db.clone())));
                if max_agents > 1 {
                    agents.push(Box::new(WebSearchAgent::new(db.clone())));
                }
                if max_agents > 2 {
                    agents.push(Box::new(CodeAnalysisAgent::new(db.clone())));
                }
            }
        }

        // Always add synthesis agent if we have room
        if agents.len() < max_agents {
            agents.push(Box::new(SynthesisAgent::new(db)));
        }

        agents.truncate(max_agents);
        agents
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_knowledge_base_agent() {
        let db = Arc::new(SqliteDb::open_in_memory().unwrap());

        // Initialize schema
        db.execute(
            r#"CREATE TABLE IF NOT EXISTS reasoning_patterns (
                id TEXT PRIMARY KEY,
                problem TEXT,
                solution TEXT,
                category TEXT,
                reward REAL DEFAULT 0.5,
                tags TEXT
            )"#,
            &[],
        ).await.unwrap();

        // Insert test pattern with "error" keyword that will match
        db.execute(
            "INSERT INTO reasoning_patterns (id, problem, solution, category, reward) VALUES (?, ?, ?, ?, ?)",
            &[&"test-1", &"error handling in Rust", &"Use Result type for error handling", &"rust.error", &"0.8"],
        ).await.unwrap();

        let agent = KnowledgeBaseAgent::new(db);
        let request = ResearchRequest::new("error");

        let result = agent.research(&request).await.unwrap();

        assert_eq!(result.agent_type, AgentType::KnowledgeBase);
        assert!(!result.findings.is_empty());
        assert!(result.completed_at.is_some());
    }

    #[test]
    fn test_agent_factory() {
        let db = Arc::new(SqliteDb::open_in_memory().unwrap());

        let agents = AgentFactory::create_agents(db.clone(), &ResearchStrategy::Auto, 3);
        assert_eq!(agents.len(), 3);

        let agents = AgentFactory::create_agents(db, &ResearchStrategy::WebSearch, 2);
        assert_eq!(agents.len(), 2);
    }
}
