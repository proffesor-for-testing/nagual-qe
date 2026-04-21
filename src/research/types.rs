//! Research types for knowledge acquisition
//!
//! Defines core types for the research swarm system including requests,
//! strategies, budgets, and results.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::fmt;

/// A research request specifying what to research and how
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResearchRequest {
    pub id: String,
    pub topic: String,
    pub depth: ResearchDepth,
    pub strategy: ResearchStrategy,
    pub budget: ResearchBudget,
    pub domain: Option<String>,
    pub context: Option<String>,
    pub created_at: DateTime<Utc>,
}

impl ResearchRequest {
    pub fn new(topic: impl Into<String>) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            topic: topic.into(),
            depth: ResearchDepth::Medium,
            strategy: ResearchStrategy::Auto,
            budget: ResearchBudget::default(),
            domain: None,
            context: None,
            created_at: Utc::now(),
        }
    }

    pub fn with_depth(mut self, depth: ResearchDepth) -> Self {
        self.depth = depth;
        self
    }

    pub fn with_strategy(mut self, strategy: ResearchStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    pub fn with_budget(mut self, budget: ResearchBudget) -> Self {
        self.budget = budget;
        self
    }

    pub fn with_domain(mut self, domain: impl Into<String>) -> Self {
        self.domain = Some(domain.into());
        self
    }

    pub fn with_context(mut self, context: impl Into<String>) -> Self {
        self.context = Some(context.into());
        self
    }
}

/// Research depth determines thoroughness vs speed tradeoff
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ResearchDepth {
    /// Single source, fast (10-20s)
    Quick,
    /// 2-3 sources, balanced (30-60s)
    #[default]
    Medium,
    /// 5+ sources, thorough (60-120s)
    Deep,
}

impl ResearchDepth {
    pub fn source_count(&self) -> usize {
        match self {
            ResearchDepth::Quick => 1,
            ResearchDepth::Medium => 3,
            ResearchDepth::Deep => 5,
        }
    }

    pub fn max_patterns(&self) -> usize {
        match self {
            ResearchDepth::Quick => 2,
            ResearchDepth::Medium => 5,
            ResearchDepth::Deep => 10,
        }
    }
}

impl fmt::Display for ResearchDepth {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ResearchDepth::Quick => write!(f, "quick"),
            ResearchDepth::Medium => write!(f, "medium"),
            ResearchDepth::Deep => write!(f, "deep"),
        }
    }
}

impl std::str::FromStr for ResearchDepth {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "quick" | "fast" | "q" => Ok(ResearchDepth::Quick),
            "medium" | "balanced" | "m" => Ok(ResearchDepth::Medium),
            "deep" | "thorough" | "d" => Ok(ResearchDepth::Deep),
            _ => Err(format!("Unknown depth: {}. Use quick, medium, or deep", s)),
        }
    }
}

/// Research strategy determines how to gather information
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum ResearchStrategy {
    /// Search the web for information
    WebSearch,
    /// Fetch and parse specific documentation URLs
    DocFetch { urls: Vec<String> },
    /// Analyze a local codebase for patterns
    CodeAnalysis { repo_path: String },
    /// Use all available strategies
    Combined,
    /// Let the coordinator decide based on topic
    #[default]
    Auto,
}

impl fmt::Display for ResearchStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ResearchStrategy::WebSearch => write!(f, "web"),
            ResearchStrategy::DocFetch { .. } => write!(f, "docs"),
            ResearchStrategy::CodeAnalysis { .. } => write!(f, "code"),
            ResearchStrategy::Combined => write!(f, "combined"),
            ResearchStrategy::Auto => write!(f, "auto"),
        }
    }
}

/// Budget constraints for research
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResearchBudget {
    /// Maximum tokens to spend on research
    pub max_tokens: usize,
    /// Maximum time in seconds
    pub max_time_seconds: u64,
    /// Maximum parallel agents
    pub max_agents: usize,
    /// Maximum patterns to create
    pub max_patterns: usize,
}

impl Default for ResearchBudget {
    fn default() -> Self {
        Self {
            max_tokens: 10000,
            max_time_seconds: 120,
            max_agents: 3,
            max_patterns: 5,
        }
    }
}

impl ResearchBudget {
    pub fn quick() -> Self {
        Self {
            max_tokens: 3000,
            max_time_seconds: 30,
            max_agents: 1,
            max_patterns: 2,
        }
    }

    pub fn deep() -> Self {
        Self {
            max_tokens: 20000,
            max_time_seconds: 180,
            max_agents: 5,
            max_patterns: 10,
        }
    }
}

/// Type of research agent
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AgentType {
    WebSearch,
    DocFetch,
    CodeAnalysis,
    Synthesis,
    KnowledgeBase,
}

impl fmt::Display for AgentType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AgentType::WebSearch => write!(f, "WebSearch"),
            AgentType::DocFetch => write!(f, "DocFetch"),
            AgentType::CodeAnalysis => write!(f, "CodeAnalysis"),
            AgentType::Synthesis => write!(f, "Synthesis"),
            AgentType::KnowledgeBase => write!(f, "KnowledgeBase"),
        }
    }
}

/// A single step in a research trajectory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResearchStep {
    pub action: ResearchAction,
    pub result: Option<String>,
    pub tokens_used: usize,
    pub duration_ms: u64,
    pub success: bool,
}

/// Actions that can be performed during research
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResearchAction {
    /// Search the web for a query
    Search { query: String },
    /// Fetch content from a URL
    FetchUrl { url: String },
    /// Extract key information from content
    ExtractInfo { content_summary: String },
    /// Query existing knowledge base
    QueryKnowledge { query: String },
    /// Synthesize findings into patterns
    Synthesize { source_count: usize },
}

impl fmt::Display for ResearchAction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ResearchAction::Search { query } => write!(f, "Search: {}", query),
            ResearchAction::FetchUrl { url } => write!(f, "Fetch: {}", url),
            ResearchAction::ExtractInfo { content_summary } => {
                write!(f, "Extract: {}", content_summary)
            }
            ResearchAction::QueryKnowledge { query } => write!(f, "KB Query: {}", query),
            ResearchAction::Synthesize { source_count } => {
                write!(f, "Synthesize {} sources", source_count)
            }
        }
    }
}

/// A complete research trajectory from one agent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResearchTrajectory {
    pub id: String,
    pub agent_type: AgentType,
    pub steps: Vec<ResearchStep>,
    pub findings: Vec<ResearchFinding>,
    pub quality_score: f64,
    pub started_at: DateTime<Utc>,
    pub completed_at: Option<DateTime<Utc>>,
}

impl ResearchTrajectory {
    pub fn new(agent_type: AgentType) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            agent_type,
            steps: Vec::new(),
            findings: Vec::new(),
            quality_score: 0.0,
            started_at: Utc::now(),
            completed_at: None,
        }
    }

    pub fn add_step(&mut self, step: ResearchStep) {
        self.steps.push(step);
    }

    pub fn add_finding(&mut self, finding: ResearchFinding) {
        self.findings.push(finding);
    }

    pub fn complete(&mut self, quality_score: f64) {
        self.quality_score = quality_score;
        self.completed_at = Some(Utc::now());
    }

    pub fn total_tokens(&self) -> usize {
        self.steps.iter().map(|s| s.tokens_used).sum()
    }

    pub fn total_duration_ms(&self) -> u64 {
        self.steps.iter().map(|s| s.duration_ms).sum()
    }
}

/// A finding from research
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResearchFinding {
    pub content: String,
    pub source: String,
    pub confidence: f64,
    pub tags: Vec<String>,
}

impl ResearchFinding {
    pub fn new(content: impl Into<String>, source: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            source: source.into(),
            confidence: 0.5,
            tags: Vec::new(),
        }
    }

    pub fn with_confidence(mut self, confidence: f64) -> Self {
        self.confidence = confidence.clamp(0.0, 1.0);
        self
    }

    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.tags = tags;
        self
    }
}

/// Consensus result from MaTTS aggregation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusResult {
    pub summary: String,
    pub confidence: f64,
    pub key_findings: Vec<String>,
    pub sources: Vec<String>,
    pub agreement_score: f64,
}

impl Default for ConsensusResult {
    fn default() -> Self {
        Self {
            summary: String::new(),
            confidence: 0.0,
            key_findings: Vec::new(),
            sources: Vec::new(),
            agreement_score: 0.0,
        }
    }
}

/// Complete research result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResearchResult {
    pub request_id: String,
    pub topic: String,
    pub trajectories: Vec<ResearchTrajectory>,
    pub consensus: ConsensusResult,
    pub patterns_created: Vec<PatternSummary>,
    pub total_tokens: usize,
    pub total_duration_ms: u64,
    pub completed_at: DateTime<Utc>,
}

impl ResearchResult {
    pub fn success_rate(&self) -> f64 {
        if self.trajectories.is_empty() {
            return 0.0;
        }
        let successful = self
            .trajectories
            .iter()
            .filter(|t| t.quality_score > 0.5)
            .count();
        successful as f64 / self.trajectories.len() as f64
    }
}

/// Summary of a created pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternSummary {
    pub id: String,
    pub problem: String,
    pub domain: String,
    pub source: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_research_request_builder() {
        let request = ResearchRequest::new("Rust async patterns")
            .with_depth(ResearchDepth::Deep)
            .with_domain("rust.async")
            .with_budget(ResearchBudget::deep());

        assert_eq!(request.topic, "Rust async patterns");
        assert_eq!(request.depth, ResearchDepth::Deep);
        assert_eq!(request.domain, Some("rust.async".to_string()));
        assert_eq!(request.budget.max_agents, 5);
    }

    #[test]
    fn test_depth_parsing() {
        assert_eq!(
            "quick".parse::<ResearchDepth>().unwrap(),
            ResearchDepth::Quick
        );
        assert_eq!(
            "deep".parse::<ResearchDepth>().unwrap(),
            ResearchDepth::Deep
        );
        assert!("invalid".parse::<ResearchDepth>().is_err());
    }

    #[test]
    fn test_trajectory_tracking() {
        let mut traj = ResearchTrajectory::new(AgentType::WebSearch);

        traj.add_step(ResearchStep {
            action: ResearchAction::Search {
                query: "test".into(),
            },
            result: Some("Found results".into()),
            tokens_used: 100,
            duration_ms: 500,
            success: true,
        });

        traj.add_finding(ResearchFinding::new("Key insight", "web search").with_confidence(0.8));

        traj.complete(0.85);

        assert_eq!(traj.total_tokens(), 100);
        assert_eq!(traj.findings.len(), 1);
        assert!(traj.completed_at.is_some());
    }
}
