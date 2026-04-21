//! Research CLI command
//!
//! Provides the `nagual research` command for autonomous knowledge acquisition.

use clap::{Args, Subcommand};
use std::sync::Arc;
use tracing::info;

use crate::db::SqliteDb;
use crate::error::NagualError;
use crate::research::{
    ResearchBudget, ResearchCoordinator, ResearchDepth, ResearchRequest,
    ResearchStrategy,
};

#[derive(Debug, Args)]
pub struct ResearchCommand {
    /// Topic to research (shortcut for 'research topic')
    #[arg(index = 1)]
    pub topic: Option<String>,

    /// Research depth: quick, medium, deep
    #[arg(long, short = 'd', default_value = "medium")]
    pub depth: String,

    /// Research strategy: web, docs, code, combined, auto
    #[arg(long, short = 's', default_value = "auto")]
    pub strategy: String,

    /// Domain to categorize resulting patterns
    #[arg(long)]
    pub domain: Option<String>,

    /// Maximum tokens to spend
    #[arg(long, default_value = "10000")]
    pub max_tokens: usize,

    /// Maximum time in seconds
    #[arg(long, default_value = "120")]
    pub max_time: u64,

    /// Maximum patterns to create
    #[arg(long, default_value = "5")]
    pub max_patterns: usize,

    /// URLs to fetch (for docs strategy)
    #[arg(long)]
    pub urls: Vec<String>,

    /// Repository path (for code strategy)
    #[arg(long)]
    pub repo: Option<String>,

    /// Dry run - show plan but don't execute
    #[arg(long)]
    pub dry_run: bool,

    /// External research - spawn Claude agents with WebSearch for real web research
    #[arg(long)]
    pub external: bool,

    /// Output as JSON
    #[arg(long)]
    pub json: bool,

    /// Enable verbose output
    #[arg(short, long)]
    pub verbose: bool,

    /// Path to config file
    #[arg(short, long)]
    pub config: Option<String>,

    /// Path to the SQLite database
    #[arg(long, default_value = "nagual.db")]
    pub db_path: String,

    #[command(subcommand)]
    pub command: Option<ResearchSubcommand>,
}

#[derive(Debug, Subcommand)]
pub enum ResearchSubcommand {
    /// Research a topic
    Topic {
        /// The topic to research
        topic: String,
    },
    /// Fill knowledge gaps automatically
    FillGaps {
        /// Target domain for gap filling
        #[arg(long)]
        domain: Option<String>,
        /// Maximum gaps to fill
        #[arg(long, default_value = "3")]
        max_gaps: usize,
    },
    /// Show research history
    History {
        /// Number of recent research sessions to show
        #[arg(long, default_value = "10")]
        limit: usize,
    },
}

pub async fn run(args: ResearchCommand) -> Result<(), NagualError> {
    let db = Arc::new(SqliteDb::open(&args.db_path)?);

    // Determine what to research
    let topic = match (&args.topic, &args.command) {
        (Some(t), _) => t.clone(),
        (None, Some(ResearchSubcommand::Topic { topic })) => topic.clone(),
        (None, Some(ResearchSubcommand::FillGaps { domain, max_gaps })) => {
            return fill_gaps(db, domain.clone(), *max_gaps, &args).await;
        }
        (None, Some(ResearchSubcommand::History { limit })) => {
            return show_history(db, *limit, args.json).await;
        }
        (None, None) => {
            println!("Usage: nagual research <TOPIC>");
            println!();
            println!("Examples:");
            println!("  nagual research \"Rust async error handling\"");
            println!("  nagual research \"tokio vs async-std\" --depth deep");
            println!("  nagual research \"GraphQL patterns\" --domain graphql");
            println!("  nagual research fill-gaps --domain rust");
            println!();
            println!("Run 'nagual research --help' for more options.");
            return Ok(());
        }
    };

    // Parse depth
    let depth: ResearchDepth = args.depth.parse().map_err(|e| NagualError::Internal {
        message: e,
    })?;

    // Parse strategy
    let strategy = parse_strategy(&args.strategy, &args.urls, &args.repo)?;

    // Build budget
    let budget = ResearchBudget {
        max_tokens: args.max_tokens,
        max_time_seconds: args.max_time,
        max_agents: depth.source_count(),
        max_patterns: args.max_patterns,
    };

    // Build request
    let mut request = ResearchRequest::new(&topic)
        .with_depth(depth)
        .with_strategy(strategy)
        .with_budget(budget);

    if let Some(ref domain) = args.domain {
        request = request.with_domain(domain);
    }

    // Create coordinator
    let coordinator = ResearchCoordinator::with_defaults(db.clone());

    if args.dry_run {
        // Show plan only
        let plan = coordinator.plan(&request);
        print_plan(&plan, args.json);
        return Ok(());
    }

    // External research mode - output request for Claude to spawn Task agents
    if args.external {
        return run_external_research(&topic, &args, &request).await;
    }

    info!("Starting research: {}", topic);

    // Execute local research (searches local KB)
    let result = coordinator.research(request).await?;

    // If no good results from local KB, suggest external research
    if result.patterns_created.is_empty() && result.consensus.confidence < 0.5 {
        println!();
        println!("💡 Tip: Local knowledge base has limited information on this topic.");
        println!("   Run with --external flag for web research via Claude agents:");
        println!("   nagual research \"{}\" --external", topic);
        println!();
    }

    // Print results
    if args.json {
        println!("{}", serde_json::to_string_pretty(&result).unwrap_or_default());
    } else {
        print_result(&result);
    }

    Ok(())
}

fn parse_strategy(
    strategy: &str,
    urls: &[String],
    repo: &Option<String>,
) -> Result<ResearchStrategy, NagualError> {
    match strategy.to_lowercase().as_str() {
        "web" => Ok(ResearchStrategy::WebSearch),
        "docs" => {
            if urls.is_empty() {
                Ok(ResearchStrategy::WebSearch) // Fallback to web search
            } else {
                Ok(ResearchStrategy::DocFetch { urls: urls.to_vec() })
            }
        }
        "code" => {
            let repo_path = repo.clone().unwrap_or_else(|| ".".to_string());
            Ok(ResearchStrategy::CodeAnalysis { repo_path })
        }
        "combined" => Ok(ResearchStrategy::Combined),
        "auto" => Ok(ResearchStrategy::Auto),
        _ => Err(NagualError::Internal {
            message: format!(
                "Unknown strategy: {}. Use web, docs, code, combined, or auto",
                strategy
            ),
        }),
    }
}

fn print_plan(plan: &crate::research::ResearchPlan, json: bool) {
    if json {
        println!(
            "{}",
            serde_json::json!({
                "topic": plan.topic,
                "depth": format!("{}", plan.depth),
                "strategy": plan.strategy,
                "agents": plan.agents,
                "budget": {
                    "max_tokens": plan.budget.max_tokens,
                    "max_time_seconds": plan.budget.max_time_seconds,
                    "max_agents": plan.budget.max_agents,
                    "max_patterns": plan.budget.max_patterns,
                },
                "estimated_duration_seconds": plan.estimated_duration_seconds,
                "estimated_patterns": plan.estimated_patterns,
            })
        );
        return;
    }

    println!("Research Plan (Dry Run)");
    println!("=======================");
    println!();
    println!("Topic: {}", plan.topic);
    println!("Depth: {}", plan.depth);
    println!("Strategy: {}", plan.strategy);
    println!();
    println!("Agents to spawn:");
    for (i, agent) in plan.agents.iter().enumerate() {
        println!("  {}. {}", i + 1, agent);
    }
    println!();
    println!("Budget:");
    println!("  Max tokens:   {}", plan.budget.max_tokens);
    println!("  Max time:     {}s", plan.budget.max_time_seconds);
    println!("  Max agents:   {}", plan.budget.max_agents);
    println!("  Max patterns: {}", plan.budget.max_patterns);
    println!();
    println!("Estimated duration: ~{}s", plan.estimated_duration_seconds);
    println!("Estimated patterns: up to {}", plan.estimated_patterns);
    println!();
    println!("Run without --dry-run to execute research.");
}

fn print_result(result: &crate::research::ResearchResult) {
    println!("Research Results");
    println!("================");
    println!();
    println!("Topic: {}", result.topic);
    println!("Duration: {}ms", result.total_duration_ms);
    println!("Tokens used: {}", result.total_tokens);
    println!();

    // Consensus
    println!("Consensus (confidence: {:.0}%)", result.consensus.confidence * 100.0);
    println!("─────────────────────────────");
    if result.consensus.key_findings.is_empty() {
        println!("  No significant findings");
    } else {
        for (i, finding) in result.consensus.key_findings.iter().enumerate() {
            let truncated: String = finding.chars().take(100).collect();
            println!("  {}. {}{}", i + 1, truncated, if finding.len() > 100 { "..." } else { "" });
        }
    }
    println!();

    // Trajectories
    println!("Trajectories ({})", result.trajectories.len());
    println!("─────────────────────────────");
    for traj in &result.trajectories {
        println!(
            "  • {} - {} findings, quality: {:.0}%",
            traj.agent_type,
            traj.findings.len(),
            traj.quality_score * 100.0
        );
    }
    println!();

    // Patterns created
    if !result.patterns_created.is_empty() {
        println!("Patterns Created ({})", result.patterns_created.len());
        println!("─────────────────────────────");
        for pattern in &result.patterns_created {
            let problem_short: String = pattern.problem.chars().take(60).collect();
            println!("  • [{}] {}", pattern.domain, problem_short);
            println!("    ID: {}", pattern.id);
        }
    } else {
        println!("No patterns created (auto-store disabled or no high-confidence findings)");
    }
}

/// Run external research via Claude Task agents
/// Outputs a structured research request that Claude can use to spawn agents with WebSearch
async fn run_external_research(
    topic: &str,
    args: &ResearchCommand,
    request: &ResearchRequest,
) -> Result<(), NagualError> {
    let domain = args.domain.clone().unwrap_or_else(|| {
        // Try to infer domain from topic
        let topic_lower = topic.to_lowercase();
        if topic_lower.contains("rust") { "rust".to_string() }
        else if topic_lower.contains("python") { "python".to_string() }
        else if topic_lower.contains("kubernetes") || topic_lower.contains("k8s") { "kubernetes".to_string() }
        else if topic_lower.contains("docker") { "docker".to_string() }
        else if topic_lower.contains("api") { "api".to_string() }
        else if topic_lower.contains("database") || topic_lower.contains("sql") { "database".to_string() }
        else { "general".to_string() }
    });

    if args.json {
        // Output JSON for programmatic use
        let request_json = serde_json::json!({
            "type": "external_research_request",
            "version": "1.0",
            "topic": topic,
            "domain": domain,
            "depth": format!("{}", request.depth),
            "max_patterns": request.budget.max_patterns,
            "instructions": {
                "task_type": "researcher",
                "tools_required": ["WebSearch", "WebFetch"],
                "store_command": format!(
                    "nagual knowledge store \"PROBLEM\" --solution \"SOLUTION\" --domain {} --tags \"TAGS\"",
                    domain
                ),
                "focus_areas": generate_focus_areas(topic),
            }
        });
        println!("{}", serde_json::to_string_pretty(&request_json).unwrap());
        return Ok(());
    }

    // Human-readable output with Claude Task instructions
    println!("╔═══════════════════════════════════════════════════════════════════════════════╗");
    println!("║                     EXTERNAL RESEARCH REQUEST                                  ║");
    println!("╠═══════════════════════════════════════════════════════════════════════════════╣");
    println!("║ Topic:  {:<68} ║", truncate_str(topic, 68));
    println!("║ Domain: {:<68} ║", domain);
    println!("║ Depth:  {:<68} ║", format!("{}", request.depth));
    println!("║ Max Patterns: {:<62} ║", request.budget.max_patterns);
    println!("╚═══════════════════════════════════════════════════════════════════════════════╝");
    println!();
    println!("📋 Instructions for Claude:");
    println!("─────────────────────────────────────────────────────────────────────────────────");
    println!();
    println!("Use the Task tool to spawn a researcher agent:");
    println!();
    println!("```");
    println!("Task({{");
    println!("  description: \"Research {}\",", truncate_str(topic, 40));
    println!("  prompt: `Research \"{}\" using WebSearch.", topic);
    println!("Find authoritative information and store findings as patterns using:");
    println!();
    println!("nagual knowledge store \"PROBLEM\" --solution \"SOLUTION\" --domain {} --tags \"TAGS\"", domain);
    println!();
    println!("Focus on:");
    for area in generate_focus_areas(topic) {
        println!("- {}", area);
    }
    println!();
    println!("Create up to {} high-quality patterns from your research.`,", request.budget.max_patterns);
    println!("  subagent_type: \"researcher\"");
    println!("}})");
    println!("```");
    println!();
    println!("─────────────────────────────────────────────────────────────────────────────────");
    println!("💡 The researcher agent will use WebSearch to find external information");
    println!("   and store findings in Nagual's knowledge base.");
    println!();

    Ok(())
}

/// Generate focus areas based on the research topic
fn generate_focus_areas(topic: &str) -> Vec<String> {
    let topic_lower = topic.to_lowercase();
    let mut areas = vec![];

    // Add topic-specific focus areas
    if topic_lower.contains("error") || topic_lower.contains("exception") {
        areas.push("Common error patterns and handling strategies".to_string());
        areas.push("Best practices for error recovery".to_string());
    }
    if topic_lower.contains("async") || topic_lower.contains("concurrent") {
        areas.push("Async patterns and concurrency models".to_string());
        areas.push("Common pitfalls and how to avoid them".to_string());
    }
    if topic_lower.contains("test") {
        areas.push("Testing strategies and patterns".to_string());
        areas.push("Test coverage best practices".to_string());
    }
    if topic_lower.contains("api") {
        areas.push("API design principles".to_string());
        areas.push("Common API patterns and anti-patterns".to_string());
    }
    if topic_lower.contains("security") {
        areas.push("Security vulnerabilities and mitigations".to_string());
        areas.push("Secure coding practices".to_string());
    }

    // Generic areas if none matched
    if areas.is_empty() {
        areas.push("Key concepts and definitions".to_string());
        areas.push("Best practices and patterns".to_string());
        areas.push("Common pitfalls and solutions".to_string());
        areas.push("Real-world examples and use cases".to_string());
    }

    areas
}

/// Truncate string to max length with ellipsis
fn truncate_str(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len.saturating_sub(3)])
    }
}

async fn fill_gaps(
    db: Arc<SqliteDb>,
    _domain: Option<String>,
    max_gaps: usize,
    args: &ResearchCommand,
) -> Result<(), NagualError> {
    use crate::introspection::IntrospectionEngine;

    println!("Analyzing knowledge gaps...");

    // Run introspection to find gaps
    let engine = IntrospectionEngine::with_defaults(db.clone());
    let report = engine.introspect().await?;

    use crate::introspection::VulnerabilityCategory;

    // Filter to coverage gap vulnerabilities
    let gaps: Vec<_> = report
        .vulnerabilities
        .iter()
        .filter(|v| v.category == VulnerabilityCategory::CoverageGap)
        .take(max_gaps)
        .collect();

    if gaps.is_empty() {
        println!("No knowledge gaps found!");
        return Ok(());
    }

    println!("Found {} gaps to fill:", gaps.len());
    for (i, gap) in gaps.iter().enumerate() {
        let domain = gap.affected_domains.first().map(|s| s.as_str()).unwrap_or("unknown");
        println!("  {}. {} ({:?} severity)", i + 1, domain, gap.severity);
    }
    println!();

    // Research each gap
    let coordinator = ResearchCoordinator::with_defaults(db);

    for gap in gaps {
        let domain = gap.affected_domains.first().map(|s| s.as_str()).unwrap_or("general");
        let topic = format!("{} best practices", domain);
        println!("Researching: {}", topic);

        let request = ResearchRequest::new(&topic)
            .with_depth(ResearchDepth::Medium)
            .with_domain(domain)
            .with_budget(ResearchBudget {
                max_tokens: args.max_tokens / max_gaps,
                max_time_seconds: args.max_time / max_gaps as u64,
                max_agents: 2,
                max_patterns: 3,
            });

        match coordinator.research(request).await {
            Ok(result) => {
                println!(
                    "  ✓ Created {} patterns (confidence: {:.0}%)",
                    result.patterns_created.len(),
                    result.consensus.confidence * 100.0
                );
            }
            Err(e) => {
                println!("  ✗ Failed: {}", e);
            }
        }
    }

    println!();
    println!("Gap filling complete!");
    Ok(())
}

async fn show_history(db: Arc<SqliteDb>, limit: usize, json: bool) -> Result<(), NagualError> {
    // For now, show recent patterns created by research
    let sql = r#"
        SELECT id, problem, category, created_at
        FROM reasoning_patterns
        WHERE problem LIKE '%research%' OR tags LIKE '%research%'
        ORDER BY created_at DESC
        LIMIT ?
    "#;

    let limit_str = limit.to_string();
    let patterns: Vec<(String, String, String, String)> = db.query(
        sql,
        &[&limit_str],
        |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
    ).await?;

    if json {
        let items: Vec<_> = patterns
            .iter()
            .map(|(id, problem, category, created)| {
                serde_json::json!({
                    "id": id,
                    "problem": problem,
                    "category": category,
                    "created_at": created,
                })
            })
            .collect();
        println!("{}", serde_json::to_string_pretty(&items).unwrap_or_default());
        return Ok(());
    }

    println!("Research History");
    println!("================");

    if patterns.is_empty() {
        println!("No research history found.");
    } else {
        for (id, problem, category, created) in &patterns {
            let problem_short: String = problem.chars().take(50).collect();
            println!("  [{}] {} - {}", category, problem_short, created);
            println!("    ID: {}", id);
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_strategy() {
        assert!(matches!(
            parse_strategy("web", &[], &None).unwrap(),
            ResearchStrategy::WebSearch
        ));
        assert!(matches!(
            parse_strategy("auto", &[], &None).unwrap(),
            ResearchStrategy::Auto
        ));
        assert!(matches!(
            parse_strategy("code", &[], &Some("/path".to_string())).unwrap(),
            ResearchStrategy::CodeAnalysis { .. }
        ));
    }
}
